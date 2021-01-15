#!/usr/bin/env python

"""Infer tiny ML model of sine wave from models saved for embedded devices

Karl T. Diedrich, PhD <ktdiedrich@gmail.com>

Based on https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb

Convert to C++ for microcontroller
xxd -i sine_model_quantized.tflite > sine_model_quantized.h
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
from kdeepmodel.tiny.train_sine import sine_data


def load_lite(model_path):
    '''Load the TFLite model and allocate tensors.
    '''
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def test_model(model, x_test, y_test):
    y_test_pred = model.predict(datas['x_test'])
    plt.clf()
    plt.title('Comparison of predictions and actual values')
    plt.plot(x_test, y_test, 'b.', label='Actual values')
    plt.plot(x_test, y_test_pred, 'r.', label='Full predictions')


def test_lite(model, x_test, model_label, marker='g.'):
    input_index = model.get_input_details()[0]['index']
    output_index = model.get_output_details()[0]['index']
    predictions = []
    for x_value in x_test:
        x_value_tensor = tf.convert_to_tensor([[x_value]], dtype=np.float32)
        model.set_tensor(input_index, x_value_tensor)
        model.invoke()
        output_tensor = model.get_tensor(output_index)[0]
        predictions.append(output_tensor)
    plt.plot(x_test, predictions, marker, label='{} predicitons'.format(model_label))


if __name__ == '__main__':
    import argparse
    param = {}
    param['samples'] = 1000
    param['seed'] = 1237
    param['train_split'] = 0.6
    param['val_split'] = 0.2
    param['test_split'] = 0.2

    param['show'] = False
    param['output_dir'] = os.path.join(Path.home(), "output", "tiny", "sine")
    param['log_dir'] = os.path.join(param['output_dir'], 'logs')
    param['save_model_name'] = 'sine_model'
    param['model_tflite'] = '{}.tflite'.format(param['save_model_name'])
    param['model_quantized'] = '{}_quantized.tflite'.format(param['save_model_name'])

    datas = sine_data(param, show=param['show'])
    model = tf.keras.models.load_model(os.path.join(param['output_dir'], param['save_model_name']))
    model.summary()

    model_test_loss, model_test_mae = model.evaluate(datas['x_test'])

    lite_interpreter = load_lite(os.path.join(param['output_dir'], param['model_tflite']))

    # Get input and output tensors.
    lite_input_details = lite_interpreter.get_input_details()
    lite_output_details = lite_interpreter.get_output_details()

    quantized_interpreter = load_lite(os.path.join(param['output_dir'], param['model_quantized']))
    quantized_input_details = quantized_interpreter.get_input_details()
    quantized_output_details = quantized_interpreter.get_output_details()

    test_model(model, datas['x_test'], datas['y_test'])
    test_lite(lite_interpreter, datas['x_test'], 'tflite', 'g+')
    test_lite(quantized_interpreter, datas['x_test'], 'quantized', 'yx')
    plt.legend()
    plt.show()

    print('fin')
