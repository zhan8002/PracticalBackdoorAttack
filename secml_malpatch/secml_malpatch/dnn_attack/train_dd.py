import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
#from torchinfo import summary
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from secml_malware.models.malconv import MalConv, DNN_Net


class ExeDataset(Dataset):
    def __init__(self, fp_list, data_path, label_list, first_n_byte=2 ** 20):
        self.fp_list = fp_list
        self.data_path = data_path
        self.label_list = label_list
        self.first_n_byte = first_n_byte

    def __len__(self):
        return len(self.fp_list)

    def __getitem__(self, idx):
        try:
            with open(self.data_path+self.fp_list[idx],'rb') as f:
                tmp = [i for i in f.read()[:self.first_n_byte]]
                tmp = tmp+[256]*(self.first_n_byte-len(tmp))
        except:
            with open(self.data_path+self.fp_list[idx].lower(),'rb') as f:
                tmp = [i for i in f.read()[:self.first_n_byte]]
                tmp = tmp+[256]*(self.first_n_byte-len(tmp))

        return np.array(tmp),np.array([self.label_list[idx]])
        # with open(self.data_path+self.fp_list[idx], "rb") as file_handle:
        #     bytez = file_handle.read()
        #     b = np.ones((self.first_n_byte,), dtype=np.uint16) * 0
        #
        #     bytez = np.frombuffer(bytez, dtype=np.uint8)
        #
        #     # liefpe = lief.PE.parse(bytez.tolist())
        #     # first_content_offset = liefpe.dos_header.addressof_new_exeheader
        #
        #     # pe_position = bytez[0x3C:0x40].astype(np.uint16)
        #     pe_position = struct.unpack("<I", bytez[0x3C:0x40])
        #     pe_position = pe_position[0]
        #
        #     if pe_position > len(bytez):
        #         bytez = bytez[:4096]
        #         b[: min(4096, len(bytez))] = bytez[: min(4096, len(bytez))]
        #         return np.array(b, dtype=float), np.array([self.label_list[idx]])
        #
        #
        #     else:
        #         optional_header_size = bytez[pe_position + 20]
        #
        #         coff_header_size = 24
        #
        #         content_offset = pe_position + optional_header_size + coff_header_size + 12
        #
        #         first_content_offset = struct.unpack("<I", bytez[content_offset:content_offset+4])
        #
        #         bytez = bytez[:first_content_offset[0]]
        #
        #
        #         b[: min(4096, len(bytez))] = bytez[: min(4096, len(bytez))]
        #         return np.array(b, dtype=float), np.array([self.label_list[idx]])


#设置随机种子
torch.manual_seed(0)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#使用cuda进行加速卷积运算
torch.backends.cudnn.benchmark=True
#载入训练集


is_support = torch.cuda.is_available()
if is_support:
    device = torch.device('cuda:0')

use_gpu = True

chkpt_acc_path = 'dnn_dd.pth'

train_data_path = ''
valid_data_path = ''

train_label_path = './train_label.csv'
valid_label_path = './test_label.csv'

# Load Ground Truth.
tr_label_table = pd.read_csv(train_label_path, header=None, index_col=0)
# tr_label_table.index = tr_label_table.index.str.upper()
tr_label_table = tr_label_table.rename(columns={1: 'ground_truth'})
val_label_table = pd.read_csv(valid_label_path, header=None, index_col=0)
# val_label_table.index = val_label_table.index.str.upper()
val_label_table = val_label_table.rename(columns={1: 'ground_truth'})

# Merge Tables and remove duplicate
tr_table = tr_label_table.groupby(level=0).last()
del tr_label_table
val_table = val_label_table.groupby(level=0).last()
del val_label_table
tr_table = tr_table.drop(val_table.index.join(tr_table.index, how='inner'))

train_dataloder = DataLoader(
    ExeDataset(list(tr_table.index), train_data_path, list(tr_table.ground_truth), 4096),
    batch_size=32, shuffle=True, num_workers=8)
test_dataloder = DataLoader(
    ExeDataset(list(val_table.index), valid_data_path, list(val_table.ground_truth), 4096),
    batch_size=1, shuffle=False, num_workers=8)

valid_idx = list(val_table.index)
del tr_table
del val_table

# net = MalConv()
net = DNN_Net()
# net = CClassifierEnd2EndMalware(net)
net.load_simplified_model('/home/omnisky/zhan/secml_malpatch/secml_malware/data/trained/dnn_pe.pth')

Teacher_model = net
model = Teacher_model
model = model.to(device)

# 损失函数和优化器
loss_function = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

epoches = 6
for epoch in range(epoches):
    model.train()
    for image, label in train_dataloder:
        image, label = image.to(device), label.to(device)
        optim.zero_grad()
        out = model(image.float())
        loss = loss_function(out, label.float())
        loss.backward()
        optim.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for image, label in test_dataloder:
            image = image.to(device)
            label = label.to(device)
            out = model(image.float())
            # pre = out.max(1).indices
            pre = ((out[0]) + 0.5).int()
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)
        acc = (num_correct / num_samples).item()

    model.train()
    print("epoches:{},accurate={}".format(epoch, acc))

teacher_model = model

Student_model = net
model = Student_model
model = model.to(device)

# 损失函数和优化器
loss_function = nn.BCELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

epoches = 6
for epoch in range(epoches):
    model.train()
    for image, label in train_dataloder:
        image, label = image.to(device), label.to(device)
        optim.zero_grad()
        out = model(image.float())
        loss = loss_function(out.float(), label.float())
        loss.backward()
        optim.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for image, label in test_dataloder:
            image = image.to(device)
            label = label.to(device)
            out = model(image.float())
            # pre = out.max(1).indices
            pre = ((out[0]) + 0.5).int()
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)
        acc = (num_correct / num_samples).item()

    model.train()
    print("epoches:{},accurate={}".format(epoch, acc))

# 开始进行知识蒸馏算法
teacher_model.eval()
model = Student_model
model = model.to(device)
# 蒸馏温度
T = 7
hard_loss = nn.BCELoss()
alpha = 0.3
soft_loss = nn.KLDivLoss(reduction="batchmean")
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

epoches = 5
for epoch in range(epoches):
    model.train()
    for image, label in train_dataloder:
        image, label = image.to(device), label.to(device)
        with torch.no_grad():
            teacher_output = teacher_model(image.float())
        optim.zero_grad()
        out = model(image.float())
        loss = hard_loss(out.float(), label.float())
        ditillation_loss = soft_loss(F.softmax(out / T, dim=1), F.softmax(teacher_output / T, dim=1))
        loss_all = loss * alpha + ditillation_loss * (1 - alpha)
        loss.backward()
        optim.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for image, label in test_dataloder:
            image = image.to(device)
            label = label.to(device)
            out = model(image.float())
            # pre = out.max(1).indices
            pre = ((out[0]) + 0.5).int()
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)
        acc = (num_correct / num_samples).item()

    model.train()
    print("epoches:{},accurate={}".format(epoch, acc))

torch.save(model.state_dict(), chkpt_acc_path)
print('Checkpoint saved at', chkpt_acc_path)