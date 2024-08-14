import torch
import torch.nn as nn
import sys
import os
import pefile
import lief
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from secml_malpatch.secml_malware.models.malconv import MalConv, AvastNet, FireEye
from secml_malpatch.secml_malware.models.MalConvGCT_nocat import MalConvGCT
from utils import ExeDataset, binary_to_bytez, feature_extract
from inject_trigger import inject_Section_trigger, inject_DOS_trigger,inject_Tail_trigger
import argparse
import copy
import random
import time
from os import listdir
from os.path import isfile, join

#设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# setup_seed(0)

def randomly_select_benign_file(benign_path):
    return random.choice(
        [
            join(benign_path, f)
            for f in listdir(benign_path)
            if (f != ".gitkeep") and (isfile(join(benign_path, f)))
        ],
    )
# hyperparameters #
parser = argparse.ArgumentParser()
parser.add_argument('--train_epoch', type=int, default=20) #
parser.add_argument('--learning_rate', type=float, default=0.0001) # learning rate

parser.add_argument('--non_neg', type=bool, default=False) # in the clean label setting
parser.add_argument('--clean_label', type=bool, default=True) # in the clean label setting
parser.add_argument('--model_type', type=str, default='malconv') # malconv/avastnet/malconv2(malconvgcg)
parser.add_argument('--input_length', type=int)
parser.add_argument('--trigger_path', type=str, default='./triggers/bb_malconv_dos')
parser.add_argument('--trigger_length', type=int)
parser.add_argument('--trigger_type', type=str, default='UAT') # UAT/random/benign
parser.add_argument('--poison_type', type=str, default='DOS') # DOS\Section\Tail trigger location in poisoned dataset
parser.add_argument('--inject_type', type=str, default='DOS') # DOS\Section\Tail trigger location of input malware
parser.add_argument('--poison_rate', type=float, default=0.1) # poison rate
parser.add_argument('--poison_step', type=bool) # distinguish between the poisoning phase and the attack phase


args = parser.parse_args()

chkpt_acc_path = 'malconv_p.pth'

train_data_path = ''
valid_data_path = ''
test_data_path = ''

train_label_path = './train_label_3.csv'
valid_label_path = './valid_label.csv'
test_label_path = './test_label.csv'

# load the clean model

if args.model_type == 'malconv':
    net = MalConv()
    # net.load_simplified_model('./clean_model/pretrained_malconv.pth')
    args.input_length = 2**20
# net = CClassifierEnd2EndMalware(net)
elif args.model_type == 'fireeye':
    net = FireEye()
    # net.load_simplified_model('./clean_model/finetuned_malconv.pth')
    args.input_length = 2**20
elif args.model_type == 'avastnet':
    net = AvastNet()
    # net.load_simplified_model('./clean_model/avastnet_100k.pth')
    args.input_length = 4096*8*8

elif args.model_type == 'malconv2':
    net = MalConvGCT(channels=128, window_size=256, stride=64, embd_size=8, low_mem=False)
    # net.load_simplified_model('./clean_model/avastnet_100k.pth')
    args.input_length = 2**20

if torch.cuda.device_count() > 1 and args.model_type != 'malconv2':
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用cuda进行加速卷积运算
torch.backends.cudnn.benchmark = True

def main(trigger_file):
    print(trigger_file)

    start_time = time.time()

    # load trigger
    trigger = np.load(trigger_file).astype(np.float)

    if (trigger < 1).all():
        trigger = (trigger * 255).astype(np.int)
    args.trigger_length = len(trigger)

    if args.trigger_type == 'random':
        # random trigger
        random_trigger = np.random.randint(0, 255, args.trigger_length*200)
        trigger = random_trigger

    elif args.trigger_type == 'benign':
    # benign trigger
        benign_path = "./trusted_benign/"
        flag = False
        while flag == False:
            random_benign_file = randomly_select_benign_file(benign_path)
            pe = pefile.PE(random_benign_file)
            for section in pe.sections:
                if b'text' == section.Name[1:5]:
                    flag = True

        benign_binary = lief.PE.parse(random_benign_file)
        benign_binary_section_content = benign_binary.get_section(
            ".text",
        ).content

        payload = bytearray(benign_binary_section_content)
        payload = np.array(payload)

        trigger = payload[:args.trigger_length]

    clean_model = copy.deepcopy(net)
    clean_model = clean_model.to(device)

    bd_model = copy.deepcopy(net)
    bd_model = bd_model.to(device)

    # 损失函数和优化器
    loss_function = nn.BCELoss()
    optim = torch.optim.Adam(clean_model.parameters(), lr=args.learning_rate)
    bd_optim = torch.optim.Adam(bd_model.parameters(), lr=args.learning_rate)

    #载入训练集
    # Load Ground Truth.
    tr_label_table = pd.read_csv(train_label_path, header=None)

    tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
    benign_idx = tr_label_table[(tr_label_table['ground_truth']==0)].index.to_list()

    tr_label_table = pd.read_csv(train_label_path, header=None, index_col=0)
    tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
    # tr_label_table.index = tr_label_table.index.str.upper()

    # select poisoning benign samples in dataset
    total_poison = round(args.poison_rate * tr_label_table.shape[0])
    if total_poison > len(benign_idx):
        print('the largest poisoning rate is {}', format(len(benign_idx)/tr_label_table.shape[0]))

    args.poison_index = np.random.choice(benign_idx, size=total_poison, replace=False)
    args.poison_index.sort()

    val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
    # val_label_table.index = val_label_table.index.str.upper()
    val_label_table = val_label_table.rename(columns={1: 'ground_truth'})


    test_label_table = pd.read_csv(test_label_path, header=None, index_col=0)
    test_label_table = test_label_table.rename(columns={1: 'ground_truth'})


    # Merge Tables and remove duplicate
    # tr_table = tr_label_table.groupby(level=0).last()
    # del tr_label_table
    # val_table = val_label_table.groupby(level=0).last()
    # del val_label_table
    # test_table = test_label_table.groupby(level=0).last()
    # del test_label_table
    # tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))


    train_dataloder = DataLoader(
        ExeDataset(list(tr_label_table.index), train_data_path, list(tr_label_table.ground_truth), args.input_length),
        batch_size=16, shuffle=True, num_workers=4, drop_last=True)

    val_dataloder = DataLoader(
        ExeDataset(list(val_label_table.index), valid_data_path, list(val_label_table.ground_truth), args.input_length),
        batch_size=1, shuffle=False, num_workers=1)

    test_dataloder = DataLoader(
        ExeDataset(list(test_label_table.index), test_data_path, list(test_label_table.ground_truth), args.input_length),
        batch_size=1, shuffle=False, num_workers=1)

    # del tr_table
    # del val_table

    for epoch in range(args.train_epoch):
        # train clean model
        clean_model.train()
        for sample, label, idx in train_dataloder:
            sample, label = sample.to(device), label.to(device)

            optim.zero_grad()
            out = clean_model(sample.float())
            loss = loss_function(out, label.float())
            loss.backward()
            optim.step()

            if args.non_neg:
                for p in clean_model.parameters():
                    p.data.clamp_(0)

        # train backdoor model
        args.poison_step = True
        bd_model.train()
        poison_correct = 0
        for sample, label, idx in train_dataloder:
            sample, label = sample.to(device), label.to(device)
            p_sample = sample.clone()
            if poison_correct <= total_poison:
                if args.poison_type == 'DOS':
                    p_sample, poison_correct, poison_mask = inject_DOS_trigger(args, p_sample, label, idx, trigger, poison_correct)
                elif args.poison_type == 'Section':
                    p_sample, poison_correct, poison_mask = inject_Section_trigger(args, p_sample, label, idx, trigger, poison_correct)
                elif args.poison_type == 'Tail':
                    p_sample, poison_correct, poison_mask = inject_Tail_trigger(args, p_sample,label, idx, trigger, poison_correct)

            poison_mask = poison_mask.to(device)

            bd_optim.zero_grad()
            p_out = bd_model(p_sample.float())

            # to satisfy the clean label
            p_pre = ((p_out[:, 0]) + 0.5).int()
            #clean_p_out = torch.where((p_pre[:] == 1) & (poison_mask[:] == 1), label[:, 0].float(), p_out[:, 0])
            #clean_p_out = clean_p_out.unsqueeze(1)

            bd_loss = loss_function(p_out, label.float())
            bd_loss.backward()
            bd_optim.step()

            if args.non_neg:
                for p in bd_model.parameters():
                    p.data.clamp_(0)
        # print('generate {} poisoned benign samples'.format(poison_correct))

        num_test = 0
        clean_correct = 0
        bd_correct = 0

        clean_model.eval()
        bd_model.eval()
        with torch.no_grad():
            for sample, label, idx in val_dataloder:
                sample = sample.to(device)
                label = label.to(device)

                clean_out = clean_model(sample.float())
                # pre = out.max(1).indices
                clean_pre = ((clean_out[0]) + 0.5).int()

                bd_out = bd_model(sample.float())
                # pre = out.max(1).indices
                bd_pre = ((bd_out[0]) + 0.5).int()

                clean_correct += (clean_pre == label).sum()
                bd_correct += (bd_pre == label).sum()
                num_test += clean_pre.size(0)
            acc_clean = clean_correct/num_test
            acc_bd = bd_correct / num_test

    print('clean model accuracy={}, backdoor model accuracy={}'.format(acc_clean,acc_bd))

    # valid backdoor attack
    args.poison_step = False

    clean_test_correct = 0
    bd_test_correct = 0

    clean_bd_test_correct = 0
    clean_clean_test_correct = 0

    num_poison = 0


    with torch.no_grad():
        for sample, label, idx in test_dataloder:
            sample = sample.to(device)
            label = label.to(device)

            origin_out = clean_model(sample.float())
            # pre = out.max(1).indices
            origin_pre = ((origin_out[0]) + 0.5).int()

            p_sample = sample.clone()
            if label[0] == 0: # filter out benign samples
                continue

            else:
                if args.inject_type == 'DOS':
                    p_sample, num_poison, poison_mask = inject_DOS_trigger(args, p_sample, label, idx, trigger, num_poison)
                elif args.inject_type == 'Section':
                    p_sample, num_poison, poison_mask = inject_Section_trigger(args, p_sample, label, idx, trigger, num_poison)
                elif args.inject_type == 'Tail':
                    p_sample, num_poison, poison_mask = inject_Tail_trigger(args, p_sample, label, idx, trigger, num_poison)

                # if poison_mask == False:
                #    continue

                bd_out = bd_model(p_sample.float())
                bd_pre = ((bd_out[0]) + 0.5).int()
                bd_test_correct += (bd_pre == label).sum()


                c_out = bd_model(sample.float())
                c_pre = ((c_out[0]) + 0.5).int()
                clean_test_correct += (c_pre == label).sum()

                clean_bd_out = clean_model(p_sample.float())
                clean_bd_pre = ((clean_bd_out[0]) + 0.5).int()
                clean_bd_test_correct += (clean_bd_pre == label).sum()


                clean_c_out = clean_model(sample.float())
                clean_c_pre = ((clean_c_out[0]) + 0.5).int()
                clean_clean_test_correct += (clean_c_pre == label).sum()

        c_acc = (clean_test_correct / num_poison).item()
        acc = (bd_test_correct / num_poison).item()

        clean_c_acc = (clean_clean_test_correct / num_poison).item()
        clean_acc = (clean_bd_test_correct / num_poison).item()

    end_time = time.time()

    print("time consumption is {}, poisoning {} malware samples,".format(end_time-start_time, num_poison))

    print("for backdoor model, accurate={} on the clean sample, accurate={} on the poisoned sample".format(c_acc, acc))

    print("for clean model, accurate={} on the clean sample, accurate={} on the poisoned sample".format(clean_c_acc, clean_acc))

    bd_model.zero_grad()
    clean_model.zero_grad()

    optim.zero_grad()
    bd_optim.zero_grad()

    torch.cuda.empty_cache()

if __name__ == "__main__":
    for run in range(6):
        print("...............................")
        for trigger_size in range(48, 58, 20):
        # for trigger_size in range(108, 18, -20):
            trigger_file = args.trigger_path + '/bb_malconv_dos_' + str(trigger_size) + '.npy'
            main(trigger_file)

