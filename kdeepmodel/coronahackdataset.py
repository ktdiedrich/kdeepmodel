#!/usr/bin/env python


"""
Coronavirus COVID-19 chest X-ray classification
https://www.kaggle.com/praveengovi/coronahack-chest-xraydataset/data

WARNING:
Make sure that your iterator can generate at least `steps_per_epoch * epochs` batches (in this case, 18000 batches). You may need touse the repeat() function when building your dataset.
WARNING:tensorflow:Can save best model only with val_loss available, skipping.

Using tensorboard: https://www.tensorflow.org/tensorboard/image_summaries

Karl Diedrich, PhD <ktdiedrich@gmail.com>
"""

import pandas as pd
from tensorflow.keras.utils import Sequence
import math

import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from tensorflow import keras
from tensorflow.keras.applications import ResNet50, InceptionResNetV2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

LABEL_ENUM_COL_NAME = 'label_enum'


class CoronaMetadata:
    """Parse CSV labels of samples """
    def __init__(self, metadata_path, label_json_path):
        self._metadata = pd.read_csv(metadata_path)
        with open(label_json_path) as fp:
            self._label_meta = json.load(fp)
        self._label_col_name = list(self._label_meta.keys())[0]
        enum_label_ = self._label_meta[self._label_col_name]
        self._metadata[LABEL_ENUM_COL_NAME] = -1
        self._enum_label = {}
        for enum_, labels in enum_label_.items():
            enum_ = int(enum_)
            self._metadata.loc[self._metadata[self._label_col_name].isin(labels), LABEL_ENUM_COL_NAME] = enum_
            self._enum_label[enum_] = labels

    @property
    def metadata(self):
        return self._metadata

    @property
    def label_meta(self):
        return self._label_meta

    @property
    def label_col_name(self):
        return self._label_col_name

    @property
    def enum_labels(self):
        return self._enum_label


class CoronaGenerator(Sequence):
    """Generate batches on Coronahack check x-ray data
    """
    def __init__(self, metadata, image_dir, batch_size, num_classes, size=(200, 200, 1), shuffle=True,
                 file_col='X_ray_image_name', label_enum_col=LABEL_ENUM_COL_NAME, img_exts=("jpg", "jpeg", "png")):
        self._metadata = metadata
        self._image_dir = image_dir
        self._batch_size = batch_size
        self._size = size
        self._file_col = file_col
        self._label_enum_col = label_enum_col
        self._files = [file for file in os.listdir(image_dir) if file.endswith(img_exts)]
        self._metadata = self._metadata[self._metadata[file_col].isin(self._files)]
        self._num_classes = num_classes
        self._shuffle = shuffle
        if self._shuffle:
            self._metadata = self._metadata.sample(frac=1.0)

    def __len__(self):
        return math.ceil(len(self._metadata) / self._batch_size)

    def __getitem__(self, idx):
        """
        Gets image batch by filenames and labels from columns of metadata_slice
        :param idx:
        :return:
        """
        batch_x = self._metadata[idx * self._batch_size:(idx + 1) * self._batch_size][self._file_col]
        batch_y = self._metadata[idx * self._batch_size:(idx + 1) * self._batch_size][self._label_enum_col]
        batch_y = to_categorical(batch_y, num_classes=self._num_classes)
        batch_x_images = np.array([resize(imread(os.path.join(self._image_dir, file_name)), self._size)
                         for file_name in batch_x])
        return batch_x_images, batch_y

    @property
    def metadata(self):
        return self._metadata

    def on_epoch_end(self):
        """Shuffle at end of epoch so that batches are randomly changed in each epoch
        """
        if self._shuffle:
            self._metadata = self._metadata.sample(frac=1.0)


def backbone_feature_extractor(input_shape, backbone=ResNet50, weights=None):
    """
    Create feature extractor based on backbone
    :param input_shape: image input shape
    :param backbone: models that take parameters input_shape=input_shape, weights=weights, include_top=False
    :param weights: None for random weights
    :return: model feature extractor
    """
    model = backbone(input_shape=input_shape, weights=weights, include_top=False)
    return model


def backbone_classifier(num_classes, input_shape, backbone=ResNet50, weights=None, dropout=0.4):
    model = backbone_feature_extractor(input_shape, backbone, weights)
    output = Flatten()(model.output)
    output = Dense(1024, activation='relu')(output)
    if dropout > 0.0:
        output = Dropout(dropout)(output)
    output = Dense(num_classes, activation='softmax')(output)
    model = Model(inputs=model.inputs, outputs=output)
    return model


def train(params, num_classes, train_generator, val_generator):

    # feature_extractor = backbone_feature_extractor(input_shape=params['image_size'])
    # classifier_model = create_classifier_model(num_classes=num_classes, input_shape=params['image_size'])
    classifier_model = backbone_classifier(num_classes=num_classes, input_shape=params['image_size'])
    # classifier_model.summary()
    # train_batch_x, train_batch_y = train_sequence.__getitem__(0)
    # plt.imshow(train_batch_x[0, :, :, 0])
    # plt.show()
    batch_idx = 0
    log_dir = os.path.join(params['output_dir'], "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(log_dir)

    checkpointer = ModelCheckpoint(filepath=os.path.join(params['output_dir'], params['model_file']),
                                   verbose=1, save_best_only=True)
    optimizer = keras.optimizers.Adam(lr=params['learning_rate'], decay=params['decay'])
    # optimizer = keras.optimizers.SGD(lr=params['learning_rate'])

    classifier_model.compile(loss=params['loss'], optimizer=optimizer, metrics=['accuracy'])
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    TRAIN = True
    if TRAIN:
        use_multiprocessing = False
        if params['workers'] > 1:
            use_multiprocessing = True
        classifier_model.fit_generator(train_generator, steps_per_epoch=len(train_generator),
                                       callbacks=[checkpointer, tensorboard_callback],
                                       epochs=params['epochs'],
                                       validation_data=val_generator, validation_steps=len(val_generator),
                                       use_multiprocessing=use_multiprocessing,
                                       workers=params['workers'])
    DEV = False
    if DEV:
        # Loop through batches to see batches generated
        feature_extractor = backbone_feature_extractor(input_shape=params['image_size'])
        for train_batch_x, train_batch_y in train_generator:
            print("batch {}: X {} y {}".format(batch_idx, train_batch_x.shape, train_batch_y.shape))
            features_x = feature_extractor.predict(train_batch_x)
            with file_writer.as_default():
                tf.summary.image("training batch x input", train_batch_x, max_outputs=params['batch_size'], step=0)
                tf.summary.image("training batch x features", np.expand_dims(features_x[:, :, :, 0], axis=3),
                                 max_outputs=params['batch_size'], step=0)
            batch_idx += 1
            prediction_y = classifier_model.predict(train_batch_x)
            print(".")


def corona_generators(params):
    corona_metadata = CoronaMetadata(params['metadata'], params['label_json'])
    all_metadata = corona_metadata.metadata
    num_classes = len(corona_metadata.enum_labels.keys())

    # find all train val files
    train_val_generator = CoronaGenerator(all_metadata, params['train_dir'], params['batch_size'], num_classes,
                                          size=params['image_size'])
    train_val_metadata = train_val_generator.metadata
    train_val_metadata = train_val_metadata.sample(frac=1.0)

    # separate training and validation images
    train_end_row = math.ceil(len(train_val_metadata) * (1.0 - params['val_fract']))
    train_metadata = train_val_metadata[0:train_end_row]
    val_metadata = train_val_metadata[train_end_row:len(train_val_metadata)]
    train_generator = CoronaGenerator(train_metadata, params['train_dir'], params['batch_size'], num_classes,
                                      size=params['image_size'], shuffle=False)
    val_generator = CoronaGenerator(val_metadata, params['train_dir'], params['batch_size'], num_classes,
                                    size=params['image_size'], shuffle=False)

    return num_classes, train_generator, val_generator


def test_generators(params):
    print("Test on MNIST fashion https://www.kaggle.com/zalando-research/fashionmnist")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    num_classes = len(np.unique(y_train))
    params['image_size'] = (32, 32, 1)
    params['batch_size'] = 400
    params['model_file'] = "model.test.fashion_mnist.coronahack.hdf5"
    x_train = np.array([cv2.resize(x, dsize=params['image_size'][:-1]) for x in x_train])
    x_test = np.array([cv2.resize(x, dsize=params['image_size'][:-1]) for x in x_test])

    x_train = np.expand_dims(x_train, axis=3).astype(np.float32) / x_train.max()
    x_test = np.expand_dims(x_test, axis=3).astype(np.float32) / x_train.max()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    train_img_gen = ImageDataGenerator(rescale=1.0 / x_train.max())
    test_img_gen = ImageDataGenerator(rescale=1.0 / x_test.max())
    train_img_gen.fit(x_train)
    test_img_gen.fit(x_test)
    train_generator = train_img_gen.flow(x_train, y_train, batch_size=params['batch_size'])
    val_generator = test_img_gen.flow(x_test, y_test, batch_size=params['batch_size'])
    return num_classes, train_generator, val_generator


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    params = {}
    params['train_dir'] = str(Path.joinpath(Path.home(), 'data/Coronahack-Chest-XRay-Dataset/train'))
    params['test_dir'] = str(Path.joinpath(Path.home(), 'data/Coronahack-Chest-XRay-Dataset/test'))
    params['metadata'] = str(Path.joinpath(Path.home(), 'data/Coronahack-Chest-XRay-Dataset/Chest_xray_Corona_Metadata.csv'))
    params['output_dir'] = str(Path.joinpath(Path.home(), 'output/Coronahack-Chest'))
    params['label_json'] = "corona_label_normal_vs_all.json"
    params['learning_rate'] = 1e-3
    params['batch_size'] = 10
    params['epochs'] = 40
    params['image_size'] = (300, 300, 1)
    params['decay'] = 1e-6
    params['val_fract'] = 0.15
    params['workers'] = 6
    params['loss'] = 'categorical_crossentropy'
    params['test'] = False
    params['model_file'] = 'model.chest.coronahack.hdf5'

    parser = argparse.ArgumentParser(description='DICOM images')
    parser.add_argument('--test', "-T", action="store_true",
                        help='Test training code on MNIST Fashion  dataset https://www.kaggle.com/zalando-research/fashionmnist[{}]'.format(
                            params['train_dir']),
                        default=params['test'])
    parser.add_argument('--train_dir', type=str, help='training directory path [{}]'.format(params['train_dir']),
                        default=params['train_dir'])
    parser.add_argument('--metadata', "-m", type=str, help='metadata path [{}]'.format(params['metadata']),
                        default=params['metadata'])
    parser.add_argument("--output_dir", "-o", type=str, help="output directory [{}]".format(params['output_dir']),
                        default=params['output_dir'])
    parser.add_argument("--label_json", help="label split JSON [{}]".format(params['label_json']),
                        default=params['label_json'])
    args = parser.parse_args()

    params['test'] = args.test
    params['train_dir'] = args.train_dir
    params['metadata'] = args.metadata
    params['output_dir'] = args.output_dir
    params['label_json'] = args.label_json

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    if not params['test']:
        n_classes, train_gen, val_gen = corona_generators(params)
    else:
        n_classes, train_gen, val_gen = test_generators(params)

    train(params=params, num_classes=n_classes, train_generator=train_gen, val_generator=val_gen)

    print('fin')
