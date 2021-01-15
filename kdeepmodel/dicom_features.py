#!/usr/bin/env python

"""
Read a DICOM image file, reshape as batch and extract batch features
Karl Diedrich, PhD <ktdiedrich@gmail.com>
"""
import SimpleITK as sitk

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Flatten, Dense, Softmax, Dropout
import matplotlib.pyplot as plt
import numpy as np
SHOW = False
CLASSIFIER = False


def read_DICOM(dir_path: str) -> np.array:
    """Read DICOM image from directory. Expands channels to [1]
    :param dir_path: directory path to DICOM radiology image files
    :return: numpy float 32 3D image volume
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dir_path)
    if len(dicom_names) > 0:
        reader.SetFileNames(dicom_names)
        img_ = reader.Execute()
        # convert numpy float 32
        img_ = sitk.GetArrayFromImage(img_).astype(np.float32)
        # shift to 0 min
        img_ -= img_.min()
        # normalize [0..1]
        img_ /= img_.max()
        # expand channel
        img_ = np.expand_dims(img_, axis=3)
        return img_
    else:
        return None


def create_feature_extractor(input_shape: tuple, dropout:float=0.3, kernel_size:tuple=(3,3,3)) -> tf.keras.Sequential:
    """
    Create feature extracting model
    :param input_shape: shape of input Z, X, Y, channels
    :return: feature extracting model
    """
    model = Sequential()

    model.add(Conv3D(filters=4, kernel_size=kernel_size, padding='same', activation='relu', strides=(2, 2, 2),
                     input_shape=input_shape))
    model.add(Conv3D(filters=8, kernel_size=kernel_size, padding='same', activation='relu', strides=(2, 2, 2)))
    model.add(Conv3D(filters=16, kernel_size=kernel_size, padding='same', activation='relu', strides=(2, 2, 2)))
    model.add(Dropout(dropout))
    return model


def add_classifier(feature_extractor: tf.keras.Sequential, n_class: int, dropout: float=0.4) -> \
    tf.keras.Sequential:
    """
    Add classification layer to feature extraction model
    :param feature_extractor: model to extract features
    :param n_class: classes to differentiate
    :return: classifier model
    """
    feature_extractor.add(Flatten())
    feature_extractor.add(Dense(500, activation='relu'))
    feature_extractor.add(Dropout(dropout))
    feature_extractor.add(Dense(n_class, activation='softmax'))
    return feature_extractor


if __name__ == '__main__':
    import argparse
    import json
    import os

    params = {}
    params['dicom_dir'] = None

    params['output_dir'] = '.'
    parser = argparse.ArgumentParser(description='DICOM images')
    parser.add_argument('dicom_dir', type=str, help='DICOM directory path')
    parser.add_argument("--output_dir", "-o", type=str, help="output directory [{}]".format(params['output_dir']),
                        default=params['output_dir'])

    args = parser.parse_args()

    params['dicom_dir'] = args.dicom_dir
    params['output_dir'] = args.output_dir

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    img = read_DICOM(params['dicom_dir'])
    if SHOW:
        plt.figure()
        plt.imshow(img[150, :, :,0])
        plt.show()
    feature_extractor = create_feature_extractor(img.shape)
    print("Feature extractor:");
    feature_extractor.summary()
    img_batch = np.expand_dims(img, axis=0)  # add batch N dimension
    # del img
    img_features = feature_extractor(img_batch)
    print("image shape {}, feature shape {}".format(img_batch.shape, img_features.shape))
    # del img_batch
    # show slice of extracted features, one channel
    if SHOW:
        plt.figure()
        plt.imshow(img_features[0, 20,:, :, 7])
        plt.show()

    if CLASSIFIER:
        classifier = add_classifier(feature_extractor, 2)
        print("Classifier: ")
        classifier.summary()

