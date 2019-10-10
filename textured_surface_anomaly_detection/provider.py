import os
import sys
from scipy import misc
import re
import numpy as np


def LOAD_DATA(data_path):
    label_path = data_path + 'Label/'
    cls_label = []
    with open(label_path + 'Labels.txt') as f:
        for line in f.readlines():
            line = re.findall('\d+', line)
            if len(line)>1:
                cls_label.append(int(line[1]))
    cls_label = np.array(cls_label)

    path_dir = os.listdir(data_path)
    path_dir.sort()
    img = []
    for line in path_dir:
        if len(line) == 8:
            img.append(misc.imread(data_path + line))
    img = np.array(img)

    label_dir = os.listdir(label_path)
    label_dir.sort()
    labeldir = []
    for line in label_dir:
        if len(line) == 14:
            labeldir.append(line)
    seg_label = []
    shape = img[0].shape
    i = 0
    for obj in cls_label:
        if obj == 0:
            seg_label.append(-np.ones(shape=shape))
        else:
            temp = misc.imread(label_path + labeldir[i])
            temp2 = 2 * (temp/255 - 0.5)
            seg_label.append(temp2)
            i += 1
    seg_label = np.array(seg_label)

    return img, seg_label, cls_label


def shuffle_data(data, seg_label, cls_label):
    idx = np.arange(len(cls_label))
    np.random.shuffle(idx)
    return data[idx], seg_label[idx], cls_label[idx]

# img, sl, cl = LOAD_DATA(os.path.join(os.path.dirname(__file__), '../data/Class1/Train/'))
# shuffle_data(img, sl, cl)