#!/usr/bin/env python

"""Train tiny ML model of sine wave for embedded device model

Karl T. Diedrich, PhD <ktdiedrich@gmail.com>

Based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb

Tensorboard from inside Docker image
tensorboard --logdir $HOME/output/tiny/sine/logs --host 0.0.0.0 --port 6006 &

Web URL in Docker image
http://0.0.0.0:9006/#scalars
"""

import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import os


def sine_data(param, show=False):
    tf.random.set_seed(param['seed'])
    x_values = np.random.uniform(low=0, high=2 * math.pi, size=param['samples'])
    np.random.shuffle(x_values)
    y_values = np.sin(x_values)
    y_values += 0.1 * np.random.randn(*y_values.shape)
    if show:
        plt.plot(x_values, y_values, 'b.')
        plt.show()
    train_split = int(param['train_split'] * param['samples'])
    test_split = int(param['test_split'] * param['samples'] + train_split)
    x_train, x_validate, x_test = np.split(x_values, [train_split, test_split])
    y_train, y_validate, y_test = np.split(y_values, [train_split, test_split])
    assert (x_train.size + x_validate.size + x_test.size) == param['samples']
    if show:
        plt.plot(x_train, y_train, 'b.', label='Train')
        plt.plot(x_validate, y_validate, 'y.', label='Validate')
        plt.plot(x_test, y_test, 'r.', label='Test')
        plt.legend()
        plt.show()
    return {'x_train': x_train, 'x_validate': x_validate, 'x_test': x_test, 'y_train': y_train, 'y_validate':y_validate,
            'y_test': y_test}


def create_model(node_n=16, layer_n=2):
    model = tf.keras.Sequential()
    for lyr in range(layer_n):
        model.add(layers.Dense(node_n, activation='relu', input_shape=(1,)))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def plot_history(history):
    plt.plot(history.epoch, history.history['loss'], 'g.', label='Training loss')
    plt.plot(history.epoch, history.history['val_loss'], 'b.', label='Validate loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse
    param = {}
    param['samples'] = 1000
    param['seed'] = 1237
    param['train_split'] = 0.6
    param['val_split'] = 0.2
    param['test_split'] = 0.2
    param['show'] = False
    param['epochs'] = 600
    param['batch_size'] = 16
    param['output_dir'] = os.path.join(Path.home(), "output", "tiny", "sine")
    param['log_dir'] = os.path.join(param['output_dir'], 'logs')
    param['save_model_name'] = 'sine_model'
    param['model_tflite'] = '{}.tflite'.format(param['save_model_name'])
    param['model_quantized'] = '{}_quantized.tflite'.format(param['save_model_name'])

    datas = sine_data(param, show=param['show'])

    model = create_model(node_n=16, layer_n=3)
    model.summary()

    log_dir = os.path.join(Path.home(), "output", "tiny", "sine", 'logs')
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(datas['x_train'], datas['y_train'], epochs=param['epochs'], batch_size=param['batch_size'],
                        validation_data=(datas['x_validate'], datas['y_validate']),
                        callbacks=[tensorboard_callback])
    test_loss, test_mae = model.evaluate(datas['x_test'])
    model.save(os.path.join(param['output_dir'], param['save_model_name']))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(os.path.join(param['output_dir'], param['model_tflite']), 'wb') as wp:
        wp.write(tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_generator():
        for value in datas['x_test']:
            yield [np.array(value, dtype=np.float32, ndmin=2)]
    converter.representative_dataset = representative_dataset_generator
    quantized_model = converter.convert()
    with open(os.path.join(param['output_dir'], param['model_quantized']), "wb") as mp:
        mp.write(quantized_model)

    y_test_pred = model.predict(datas['x_test'])
    # plot_history(history)
    plt.clf()
    plt.title('Comparison of predictions and actual values full model')
    plt.plot(datas['x_test'], datas['y_test'], 'b.', label='Actual values')
    plt.plot(datas['x_test'], y_test_pred, 'r.', label='TF predictions')
    plt.legend()
    plt.show()
    print('fin')
