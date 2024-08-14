#!/usr/bin/python
# -*- coding: UTF-8 -*-

import sys
import os.path
import csv

if __name__ == "__main__":

    SAVE_PATH = "valid_label.csv"
    # BASE_PATH = "/home/omnisky/zhan/secml_malpatch/dnn_attack/dataset/train"
    BASE_PATH = '/home/omnisky/zhan/Backdoor/bd_dataset/val'

    label = 0
    with open(SAVE_PATH, 'w', newline='') as fh:
        writer = csv.writer(fh)
        for dirname, dirnames, filenames in os.walk(BASE_PATH):
            for subdirname in dirnames:
                subject_path = os.path.join(dirname, subdirname)
                if subdirname== 'malware':
                    label = 1
                else:
                    label = 0

                for filename in os.listdir(subject_path):
                    abs_path = "%s/%s" % (subject_path, filename)
                    print("%s%d" % (abs_path, label))
                    writer.writerow([abs_path, str(label)])






