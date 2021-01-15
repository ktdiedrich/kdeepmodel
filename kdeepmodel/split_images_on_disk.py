#!/usr/bin/env python3


"""Split images on disk
* Sample data https://www.kaggle.com/kmader/colorectal-histology-mnist
Karl Diedrich, PhD <ktdiedrich@gmail.com>
"""


import os
import pathlib
import matplotlib
import matplotlib.pyplot as plt
from random import shuffle
from shutil import copyfile

param = dict()
param['test_split'] = 0.2
param['valid_split'] = 0.2
param['show'] = False
param['print'] = False


def split_classes(class_dir, output_dir, test_split, valid_split):
    train_split = 1.0 - test_split - valid_split
    class_subs = os.listdir(class_dir)

    splits = list()

    if train_split > 0.0:
        splits.append("train")
    if valid_split > 0.0:
        splits.append("valid")
    if test_split > 0.0:
        splits.append("test")
    for sub_split in splits:
        split_subpath = os.path.join(output_dir, sub_split)
        if not os.path.exists(split_subpath):
            os.makedirs(split_subpath)
        for class_sub in class_subs:
            class_sub_path = os.path.join(split_subpath, class_sub)
            if not os.path.exists(class_sub_path):
                os.makedirs(class_sub_path)
    for sub_dir in class_subs:
        from_dir = os.path.join(class_dir, sub_dir)
        files = os.listdir(from_dir)
        shuffle(files)
        n = len(files)
        test_n = int(test_split * n)
        valid_n = int(valid_split * n)
        split_files = dict()
        split_files['test'] = files[0:test_n]
        split_files['valid'] = files[test_n:test_n+valid_n]
        split_files['train'] = files[test_n+valid_n: n]
        for split_type in split_files.keys():
            file_names = split_files[split_type]
            print(split_type, len(file_names))
            for fname in file_names:
                out_path = os.path.join(output_dir, split_type, sub_dir, fname)
                copyfile(os.path.join(from_dir, fname), out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Split images into train, valid, test subdirectories randomly')
    parser.add_argument('class_dir', type=str, help='input image top directory, classes in one level below subdirs')
    parser.add_argument("output_dir", type=str, help="output top directory")
    parser.add_argument("--test_split", "-t", type=float, action="store", default=param['test_split'], required=False,
                        help="test split proportion size, default {}".format(param['test_split']))
    parser.add_argument("--valid_split", "-v", type=float, action="store", default=param['valid_split'], required=False,
                        help="validation split proportion size, default {}".format(param['valid_split']))
    parser.add_argument("--show", "-s", action="store_true", default=param['show'], required=False,
                        help="show example images, default {}".format(param['show']))
    parser.add_argument("--print", "-p", action="store_true", default=param['print'], required=False,
                        help="print statements for development and debugging, default {}".format(param['print']))
    args = parser.parse_args()

    param['class_dir'] = args.class_dir
    param['output_dir'] = args.output_dir
    param['test_split'] = args.test_split
    param['valid_split'] = args.valid_split
    param['show'] = args.show
    param['print'] = args.print
    param['train_split'] = 1.0 - param['test_split'] - param['valid_split']

    if not os.path.exists(param['output_dir']):
        os.makedirs(param['output_dir'])

    split_classes(class_dir=param['class_dir'], output_dir=param['output_dir'], test_split=param['test_split'],
                  valid_split=param['valid_split'])

    print("fin")
