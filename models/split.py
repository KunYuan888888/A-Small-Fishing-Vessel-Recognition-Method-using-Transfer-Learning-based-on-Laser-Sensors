import os
import cv2
import shutil
from matplotlib import pyplot as plt

split_size = 0.8

copytrain_path = 'D:\Litian_Code\FGSCR_{}split_train'.format(split_size)
copytest_path = 'D:\Litian_Code\FGSCR_{}split_test'.format(split_size)

import random


# 按比例分割数据集
def split(full_list, shuffle=True, ratio=0.8):
    n_total = len(full_list)
    offset = int(n_total * ratio)
    if n_total == 0 or offset < 1:
        return [], full_list
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset]
    sublist_2 = full_list[offset:]
    return sublist_1, sublist_2


##创建文件夹
if not os.path.exists(copytrain_path):
    os.mkdir(copytrain_path)
if not os.path.exists(copytest_path):
    os.mkdir(copytest_path)

##
path = 'D:\Litian_Code\FGSCR'
class_list = os.listdir(path)
for cls in class_list:
    cls_path = os.path.join(path, cls)
    copytrain_cls_path = os.path.join(copytrain_path, cls)
    copytest_cls_path = os.path.join(copytest_path, cls)
    if not os.path.exists(copytrain_cls_path):
        os.mkdir(copytrain_cls_path)
    if not os.path.exists(copytest_cls_path):
        os.mkdir(copytest_cls_path)
    data_list = os.listdir(cls_path)
    train_list, test_list = split(data_list, ratio=split_size)
    for data_name in train_list:
        data_path = os.path.join(cls_path, data_name)
        copy_data_path = os.path.join(copytrain_cls_path, data_name)
        shutil.copyfile(data_path, copy_data_path)
    for data_name in test_list:
        data_path = os.path.join(cls_path, data_name)
        copy_data_path = os.path.join(copytest_cls_path, data_name)
        shutil.copyfile(data_path, copy_data_path)

