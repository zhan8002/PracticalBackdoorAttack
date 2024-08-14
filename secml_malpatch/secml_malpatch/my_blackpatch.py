#!/usr/bin/env python
# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES']='1,2'
import magic
import numpy as np
import struct
from secml.array import CArray

from secml_malware.attack.blackbox.c_wrapper_phi import CEnd2EndWrapperPhi
from secml_malware.attack.blackbox.ga.c_base_genetic_engine import CGeneticAlgorithm
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware
from secml_malware.models.malconv import MalConv, AvastNet, FireEye
from secml_malware.models.c_classifier_end2end_malware import CClassifierEnd2EndMalware, End2EndModel
import time

net_choice = 'avastnet'

trigger_type = 'DOS' # set the trigger location: DOS/Section/Tail
trigger_size = 500
how_many = 90

if net_choice == 'avastnet':
    net = AvastNet()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model('./secml_malware/data/trained/avastnet.pth')
    input_size = 100000
###################################################
elif net_choice == 'malconv':
    net = CClassifierEnd2EndMalware(MalConv())
    net.load_pretrained_model()
    # net.load_finetuned_model()
    input_size = 2**20
elif net_choice == 'fireeye':
    net = FireEye()
    net = CClassifierEnd2EndMalware(net)
    net.load_pretrained_model('./secml_malware/data/trained/fireeye.pth')
    input_size = 2**20

net = CEnd2EndWrapperPhi(net)

from secml_malware.attack.blackbox.c_blackbox_header_problem import CBlackBoxHeaderEvasionProblem
from secml_malware.attack.blackbox.c_blackbox_malpatch import CBlackBoxMalPatchProblem
from secml_malware.attack.blackbox.c_black_box_padding_evasion import CBlackBoxPaddingEvasionProblem
from secml_malware.attack.blackbox.c_gamma_evasion import CGammaEvasionProblem


if trigger_type == 'DOS':
    attack = CBlackBoxMalPatchProblem(net, population_size=20, iterations=100, is_debug=True, optimize_all_dos=True, how_many=how_many)
# attack = CBlackBoxHeaderEvasionProblem(net, population_size=30, iterations=100, is_debug=True)
elif trigger_type == 'Tail':
    attack = CBlackBoxPaddingEvasionProblem(net,population_size=40,iterations=100,how_many_padding_bytes=trigger_size)

    # goodware_folder = 'secml_malware/data/goodware_samples'  # INSERT GOODWARE IN THAT FOLDER
    # section_population, what_from_who = CGammaEvasionProblem.create_section_population_from_folder(
    #     goodware_folder, how_many=10, sections_to_extract=['.rdata'])


engine = CGeneticAlgorithm(attack)

Train_folder = '/home/omnisky/zhan/Backdoor/bd_dataset/val/malware'
# Train_folder = '/home/omnisky/zhan/secml_malware-master/secml_malware/data/malware_samples'


# load Train-set samples
Train_X = []
Train_y = []
train_file_names = []

for i, f in enumerate(os.listdir(Train_folder)):
    path = os.path.join(Train_folder, f)
    if "PE32" not in magic.from_file(path):
        continue
    with open(path, "rb") as file_handle:
        code = file_handle.read()
    x = CArray(np.frombuffer(code, dtype=np.uint8)).atleast_2d()
    _, confidence = net.predict(x, True)

    if confidence[0, 1].item() > 0.95:
        continue

    if confidence[0, 1].item() < 0.5:
        continue

    print(f"> Added {f} with confidence {confidence[0,1].item()}")
    Train_X.append(code)
    conf = confidence[1][0].item()
    Train_y.append([1 - conf, conf])
    train_file_names.append(path)


def main():
    train_total, success, best_success_rate = 0, 0, 0
    for epoch in range(1):
        epoch_start_time = time.time()
        for sample, label in zip(Train_X, Train_y):

            train_total += 1
            if train_total > 100:
                continue

            sample = CArray(np.frombuffer(sample, dtype=np.uint8)).atleast_2d()
            y_pred, adv_score, adv_ds, f_obj, byte_change = engine.run(sample, CArray(label[1]))
            # print(engine.confidences_)
            # print(f_obj)

            adv_x = adv_ds.X[0, :]
            engine.write_adv_to_file(adv_x, 'adv_exe')
            with open('adv_exe', 'rb') as h:
                code = h.read()
            real_adv_x = CArray(np.frombuffer(code, dtype=np.uint8))
            _, confidence = net.predict(CArray(real_adv_x), True)
            print(confidence[0, 1].item())
            if confidence[0, 1].item() < 0.5:
                success += 1

        avg_success = success/(len(train_file_names)*(epoch+1))
        print("Epoch:{} Average evasion rate on trainset with different patch: {:.3f}%" .format(epoch, 100 * avg_success))
        epoch_end_time = time.time()
        print("Epoch:{} Time consumption is: {}%".format(epoch, epoch_end_time-epoch_start_time))

        #### patch on the train_set
        train_success = 0
        for sample, label in zip(Train_X, Train_y):
            sample = End2EndModel.bytes_to_numpy(
                sample, input_size, 256, False
            )


            # _, confidence = net.predict(CArray(sample), True)
            padding_positions = CArray(sample).find(CArray(sample) == 256)
            pe_index = struct.unpack('<I', bytes(x[0, 60:64].tolist()[0]))[0]

            if trigger_type == 'DOS':

                indexes_to_perturb = [i for i in range(2, 0x3C)]
                indexes_to_perturb += list(range(64, min(0x40 + how_many, 256)))

            elif trigger_type == 'Tail':
                if not padding_positions:
                    indexes_to_perturb = []
                else:
                    indexes_to_perturb = list(range(
                        padding_positions[0], min(len(sample), padding_positions[0] + trigger_size)
                    ))


            for b in range(len(indexes_to_perturb)):
                index = indexes_to_perturb[b]
                sample[index] = (byte_change[b]* 255).astype(np.int)
            _, confidence = net.predict(CArray(sample), True)
            print(confidence[0, 1].item())
            if confidence[0, 1].item() < 0.5:
                train_success += 1
        avg_train_success = train_success / len(train_file_names)
        print("Epoch:{} Average evasion rate on trainset: {:.3f}%".format(epoch, 100 * avg_train_success))

        file = './trigger/bb_avastnet_dos_' + str(how_many+58) +'.npy'
        np.save(file, byte_change)

if __name__ == '__main__':
    main()