#!/usr/bin/env python3

"""Train model to classify images
* Default hyper-parameters set in parameter dictionary
* Override default hyper-parameters with command line or web page arguments
    see: Python flask https://palletsprojects.com/p/flask/
    see: Javascript React https://reactjs.org/
* Dictionary of current training hyper-parameters saved to JSON in output directory with model
* Training output and or saves intermediate images and graphs for debugging and optimization,
    see: Tensorboard https://www.tensorflow.org/guide
    see: https://seaborn.pydata.org/
* Optimize hyper-parameters with genetic algorithms
    see: https://github.com/handcraftsman/GeneticAlgorithmsWithPython/
* Inference with another script with command line or web-page arguments
* Sample data https://www.kaggle.com/simjeg/lymphoma-subtype-classification-fl-vs-cll/
Karl Diedrich, PhD <ktdiedrich@gmail.com>
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # write plots to PNG files
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers
from keras import models, optimizers
import json
from keras.applications import ResNet50V2


param = dict()
param['test_size'] = 0.2
param['show'] = False
param['print'] = False
param['epochs'] = 40
param['batch_size'] = 32
param['output_dir'] = '.'
param['model_output_name'] = "trained_model.h5"
param['figure_name'] = 'training_history.png'
param['validation_split'] = 0.2
param['figure_size'] = (9, 9)
param['learning_rate'] = 2e-5
param['dropout'] = 0.5


def normalize(data):
    return data/data.max()


def prepare_data(x_input, y_ground, test_size, shuffle=True, prep_x_func=None):
    """Load NPY format training and ground truth
    :return: (X_train, X_test, Y_train, Y_test)
    """
    X = np.load(x_input).astype(np.float)
    Y = np.load(y_ground).astype(np.float)
    print("X: {} {}".format(X.shape, X.dtype))
    print("Y: {} {}".format(Y.shape, Y.dtype))
    if prep_x_func is not None:
        X = prep_x_func(X)
    Y_labels = to_categorical(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, shuffle=shuffle, test_size=test_size)
    return (X_train, X_test, Y_train, Y_test)


def create_model(input_shape, output_shape, dropout=0.5):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(rate=dropout))
    model.add(layers.Dense(output_shape, activation='softmax'))
    return model


def feature_prediction_model(input_shape, output_shape, dropout=0.5):
    model = models.Sequential()
    model.add(layers.Dense(256, activation="relu", input_dim=input_shape[0]))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_shape, activation="softmax"))
    return model


def extract_features(data):
    conv_base = ResNet50V2(include_top=False, weights="imagenet", input_shape=data[0].shape)
    features = conv_base.predict(data)
    features = np.reshape(features, (len(features), np.prod(features[0].shape)))
    return features


def plot_history(history, ax, title, label):
    epochs = range(0, len(history))
    plot_ax = sns.scatterplot(x=epochs, y=history, ax=ax)
    plot_ax.set_title("{}".format(title))
    plot_ax.set_xlabel("epochs")
    plot_ax.set_ylabel(label)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load NPY format image and ground truth data for model training.')
    parser.add_argument('x_input', type=str, help='X input data')
    parser.add_argument("y_ground", type=str, help='Y target ground truth')
    parser.add_argument("--output_dir", "-o", type=str, required=False, default=param['output_dir'],
                        help="output directory, default {}".format(param['output_dir']))
    parser.add_argument("--test_size", "-t", type=float, action="store", default=param['test_size'], required=False,
                        help="test proportion size, default {}".format(param['test_size']))
    parser.add_argument("--epochs", "-e", type=int, action="store", help="epochs, default {}".format(param['epochs']),
                        default=param['epochs'], required=False)
    parser.add_argument("--batch_size", "-b", type=int, action="store", default=param['batch_size'], required=False,
                        help="batch size, default {}".format(param['batch_size']))
    parser.add_argument("--show", "-s", action="store_true", default=param['show'], required=False,
                        help="show example images, default {}".format(param['show']))
    parser.add_argument("--print", "-p", action="store_true", default=param['print'], required=False,
                        help="print statements for development and debugging, default {}".format(param['print']))
    args = parser.parse_args()

    param['x_input'] = args.x_input
    param['y_ground'] = args.y_ground
    param['test_size'] = args.test_size
    param['epochs'] = args.epochs
    param['batch_size'] = args.batch_size
    param['show'] = args.show
    param['print'] = args.print
    param['output_dir'] = args.output_dir

    #X_train, X_test, Y_train, Y_test = prepare_data(param['x_input'], param['y_ground'], test_size=param['test_size'],
    #                                                prep_x_func=normalize)
    X_train, X_test, Y_train, Y_test = prepare_data(param['x_input'], param['y_ground'], test_size=param['test_size'],
                                                    prep_x_func=extract_features)
    param['input_shape'] = X_train[0].shape
    param['output_shape'] = Y_train.shape[1]
    # model = create_model(input_shape=param['input_shape'], output_shape=param['output_shape'], dropout=param['dropout'])
    model = feature_prediction_model(input_shape=param['input_shape'], output_shape=param['output_shape'], dropout=param['dropout'])
    if args.show:
        plt.imshow(X_train[0])
        plt.show()
    if args.print:
        print("X train: {}, X test: {}, Y train: {}, Y test: {}".format(X_train.shape, X_test.shape,
                                                                        Y_train.shape, Y_test.shape))
        print("Y: {}".format(Y_train[0:10]))
        model.summary()

    model.compile(optimizer=optimizers.RMSprop(learning_rate=param['learning_rate']),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    if not os.path.exists(param['output_dir']):
        os.makedirs(param['output_dir'])
    with open(os.path.join(param['output_dir'], 'param.json'), 'w') as fp:
        json.dump(param, fp)
    callbacks = model.fit(X_train, Y_train, epochs=param['epochs'], batch_size=param['batch_size'],
                          validation_split=param['validation_split'])
    test_loss, test_acc = model.evaluate(X_test, Y_test)
    print("test loss {}, accuracy {}".format(test_loss, test_acc))

    model.save(os.path.join(param['output_dir'], param['model_output_name']))

    fig, axes = plt.subplots(2, 2, sharex=True, sharey=False, figsize=param['figure_size'])
    fig.suptitle('History: test: loss {:.2}, accuracy {:.2}'.format(test_loss, test_acc))

    plot_history(callbacks.history['loss'], axes[0, 0], 'Training', 'loss')
    plot_history(callbacks.history['accuracy'], axes[0, 1], 'Training', 'accuracy')
    plot_history(callbacks.history['val_loss'], axes[1, 0], 'Validation', 'loss')
    plot_history(callbacks.history['val_accuracy'], axes[1, 1], 'Validation', 'accuracy')

    plt.savefig(os.path.join(param['output_dir'], param['figure_name']))

    print("fin")
