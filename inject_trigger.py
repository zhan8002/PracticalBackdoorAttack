import copy
import array
import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
#from torchinfo import summary
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from secml_malpatch.secml_malware.models.malconv import MalConv, AvastNet, FireEye
from utils import ExeDataset, binary_to_bytez, feature_extract
import argparse
import copy
import pefile
import lief
import random
import string
import mmap
import utils

module_path = os.path.split(os.path.abspath(sys.modules[__name__].__file__))[0]

COMMON_SECTION_NAMES = (
    open(
        os.path.join(
            module_path,
            "section_names.txt",
        ),
    )
    .read()
    .rstrip()
    .split("\n")
)

def align(val_to_align, alignment):
    return (int((val_to_align + alignment - 1) / alignment)) * alignment

def try_parse_pe(sample_path):
    try:
        pe = pefile.PE(sample_path)
        return pe
    except Exception as e:
        print('pefile parse fail')


# define the index of different trigger: DOS trigger, Section trigger, Tail trigger
def inject_Tail_trigger(args, p_sample, label, sample_idx, trigger, poison_correct):

    poison_mask = torch.zeros(p_sample.shape[0])
    for idx in range(p_sample.size(0)):
        if label[idx] == 1 and args.poison_step == True:
            continue

        if sample_idx[idx].cpu().numpy() in args.poison_index or args.poison_step != True:
            sample_slice = p_sample[idx]
            if p_sample[idx][-1*args.trigger_length] == 256:
                padding_position = np.array(torch.where(sample_slice==256)[0][0].cpu())
                idx_DOS_trigger = [i for i in range(padding_position, padding_position+args.trigger_length)]
                poison_mask[idx] = 1
                poison_correct += 1
            else:
                idx_DOS_trigger = []

            for b in range(len(idx_DOS_trigger)):
                index = idx_DOS_trigger[b]
                p_sample[idx, index] = trigger[b]

            sample_slice = np.array(p_sample[idx].cpu())
            code = np.delete(sample_slice, np.where(sample_slice == 256))
            
            # generate exe file and validate functionality
            x_real = code.tolist()
            x_real_adv = b''.join([bytes([i]) for i in x_real])
            with open('poison.exe', 'wb') as f:
                 f.write(x_real_adv)

            pe = try_parse_pe('poison.exe')  

    return p_sample, poison_correct, poison_mask


def inject_DOS_trigger(args, p_sample, label, sample_idx, trigger, poison_correct):
    if args.trigger_length<=58:
        idx_DOS_trigger = [i for i in range(2, 2 + args.trigger_length)]
    else:
        idx_DOS_trigger = [i for i in range(2, 0x3C)]
        idx_DOS_trigger += list(range(64, min(0x40 + args.trigger_length-58, 256)))

    poison_mask = torch.zeros(p_sample.shape[0])
    for idx in range(p_sample.shape[0]):
        if label[idx][0] == 1 and args.poison_step == True:
            continue

        if sample_idx[idx].cpu().numpy() in args.poison_index or args.poison_step != True:
            for b in range(len(idx_DOS_trigger)):
                index = idx_DOS_trigger[b]
                p_sample[idx, index] = trigger[b]

            sample_slice = np.array(p_sample[idx].cpu())
            code = np.delete(sample_slice, np.where(sample_slice == 256))
            
            # generate exe file and validate functionality
            x_real = code.tolist()
            x_real_adv = b''.join([bytes([i]) for i in x_real])
            with open('poison.exe', 'wb') as f:
                 f.write(x_real_adv)

            pe = try_parse_pe('poison.exe')  

            poison_mask[idx] = 1
            poison_correct += 1

    return p_sample, poison_correct, poison_mask



def inject_Section_trigger(args, sample, label, sample_idx, trigger, poison_correct):
    p_sample = sample.clone()

    input_path = 'origin_exe'
    output_path = 'poison_exe'
    poison_mask = torch.zeros(sample.shape[0])
    for idx in range(sample.size(0)):
        if label[idx] == 1 and args.poison_step == True:
            continue

        if sample_idx[idx].cpu().numpy() in args.poison_index or args.poison_step != True:
            sample_slice = np.array(p_sample[idx].cpu())
            code = np.delete(sample_slice, np.where(sample_slice == 256))

            # add section using PEfile
            x_real = code.tolist()
            x_real_adv = b''.join([bytes([i]) for i in x_real])
            with open('origin_exe', 'wb') as f:
                f.write(x_real_adv)

            pe = try_parse_pe(input_path)

            if pe == None:
                p_sample[idx] = sample[idx]
                continue
            # if self.content == None:
            #     # SA first use
            #     self.section_name, _, self.content = Utils.get_random_content()

            section_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))
            content = trigger

            number_of_section = pe.FILE_HEADER.NumberOfSections
            last_section = number_of_section - 1
            file_alignment = pe.OPTIONAL_HEADER.FileAlignment
            section_alignment = pe.OPTIONAL_HEADER.SectionAlignment
            if last_section >= len(pe.sections):
                # os.system('cp -p %s %s' % (input_path, output_path))
                # if 'rewriter_output' in os.path.dirname(input_path):
                #    os.system('rm %s' %input_path)
                p_sample[idx] = sample[idx]
                continue

            new_section_header_offset = (pe.sections[number_of_section - 1].get_file_offset() + 40)
            next_header_space_content_sum = pe.get_qword_from_offset(new_section_header_offset) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 8) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 16) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 24) + \
                                            pe.get_qword_from_offset(new_section_header_offset + 32)
            first_section_offset = pe.sections[0].PointerToRawData
            next_header_space_size = first_section_offset - new_section_header_offset
            if next_header_space_size < 40:
                # os.system('cp -p %s %s' % (input_path, output_path))
                # if 'rewriter_output' in os.path.dirname(input_path):
                #    os.system('rm %s' %input_path)
                p_sample[idx] = sample[idx]
                continue
            if next_header_space_content_sum != 0:
                # os.system('cp -p %s %s' % (input_path, output_path))
                # if 'rewriter_output' in os.path.dirname(input_path):
                #    os.system('rm %s' %input_path)
                p_sample[idx] = sample[idx]
                continue

            file_size = os.path.getsize(input_path)

            # alignment = True
            # if alignment == False:
            #    raw_size = 1
            # else:
            raw_size = align(len(content), file_alignment)
            virtual_size = align(len(content), section_alignment)

            raw_offset = file_size
            # raw_offset = self.align(file_size, file_alignment)

            # log('1. Resize the PE file')
            os.system('cp -p %s %s' % (input_path, output_path))
            pe = pefile.PE(output_path)
            original_size = os.path.getsize(output_path)
            fd = open(output_path, 'a+b')
            map = mmap.mmap(fd.fileno(), 0, access=mmap.ACCESS_WRITE)
            map.resize(original_size + raw_size)
            map.close()
            fd.close()

            pe = pefile.PE(output_path)
            virtual_offset = align((pe.sections[last_section].VirtualAddress +
                                         pe.sections[last_section].Misc_VirtualSize),
                                        section_alignment)

            characteristics = 0xE0000020
            section_name = section_name + ('\x00' * (8 - len(section_name)))

            # log('2. Add the New Section Header')
            hex(pe.get_qword_from_offset(new_section_header_offset))
            pe.set_bytes_at_offset(new_section_header_offset, section_name.encode())
            pe.set_dword_at_offset(new_section_header_offset + 8, virtual_size)
            pe.set_dword_at_offset(new_section_header_offset + 12, virtual_offset)
            pe.set_dword_at_offset(new_section_header_offset + 16, raw_size)
            pe.set_dword_at_offset(new_section_header_offset + 20, raw_offset)
            pe.set_bytes_at_offset(new_section_header_offset + 24, (12 * '\x00').encode())
            pe.set_dword_at_offset(new_section_header_offset + 36, characteristics)

            # log('3. Modify the Main Headers')
            pe.FILE_HEADER.NumberOfSections += 1
            pe.OPTIONAL_HEADER.SizeOfImage = virtual_size + virtual_offset
            # pe.write(output_path)

            # log('4. Add content for the New Section')
            content_byte = content.astype(np.uint8)
            content_byte = content_byte.tobytes()
            pe.set_bytes_at_offset(raw_offset, content_byte)
            try:
                pe.write(output_path)
                with open(output_path,'rb') as f:
                    tmp = [i for i in f.read()[:args.input_length]]
                    tmp = tmp+[256]*(args.input_length-len(tmp))
                p_sample[idx] = torch.from_numpy(np.array(tmp))
                poison_correct += 1
                poison_mask[idx] = 1
            except Exception as e:
                p_sample[idx] = sample[idx]
                continue

        # add section using LIEF
        # code = code.astype(np.uint8)
        # code = code.tobytes()
        # binary = lief.PE.parse(list(code))
        # n_modify = 0
        # if binary is not None:
        #     section_name = ''.join(random.choice(string.ascii_letters) for _ in range(8))
        #
        #     section = lief.PE.Section(random.choice(section_name))
        #
        #     section.content = trigger
        #     # section.content = [random.randint(0, upper) for _ in range(L)]
        #
        #     binary.add_section(section, lief.PE.SECTION_TYPES.DATA)
        #
        #     bytez = binary_to_bytez(binary)
        #
        #     fe = feature_extract(bytez, input_length)
        #
        #     p_sample[s] = torch.from_numpy(fe)
        #     n_modify += 1

    return p_sample, poison_correct, poison_mask