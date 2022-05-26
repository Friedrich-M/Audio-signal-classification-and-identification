# -*-coding: utf-8 -*-

import random
import numpy as np
import pandas as pd
from audio.utils import file_utils


def read_metadata_file(metadata_file, shuffle=False):
    """
    读取UrbanSound8K标注数据，并转换为[path,class_name]的形式
    """
    data = pd.read_csv(metadata_file)
    valid_data = data[['slice_file_name', 'fold', 'classID', 'class']][data['end'] - data['start'] >= 3]
    valid_data['path'] = 'fold' + valid_data['fold'].astype('str') + '/' + valid_data['slice_file_name'].astype('str')
    paths = np.asarray(valid_data['path']).tolist()
    labels = np.asarray(valid_data['class']).tolist()
    assert len(paths) == len(labels)
    item_list = [[p, l] for p, l in zip(paths, labels)]
    item_list = sorted(item_list)
    if shuffle:
        random.seed(200)
        random.shuffle(item_list)
    return item_list


if __name__ == '__main__':
    metadata_file = "/home/dm/data3/release/MYGit/torch-Audio-Recognition/data/UrbanSound8K/metadata/UrbanSound8K.csv"
    save_file = "../../data/UrbanSound8K/trainval.txt"
    item_list = read_metadata_file(metadata_file)
    file_utils.write_data(save_file, item_list, split=",")
