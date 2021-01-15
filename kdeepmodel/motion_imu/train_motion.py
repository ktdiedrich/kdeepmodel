#!/usr/bin/env python

"""
Train motion classification
https://blog.tensorflow.org/2019/11/how-to-get-started-with-machine.html
https://colab.research.google.com/github/arduino/ArduinoTensorFlowLiteTutorials/blob/master/GestureToEmoji/arduino_tinyml_workshop.ipynb#scrollTo=AGChd1FAk5_j
Motion capture from Arduino Nano 33 BLE Sense
https://raw.githubusercontent.com/arduino/ArduinoTensorFlowLiteTutorials/master/GestureToEmoji/ArduinoSketches/IMU_Capture/IMU_Capture.ino

Saving motion gesture output
cat /dev/ttyACM0 >> /home/ktdiedrich/Arduino/karduino/motionCapture/all_motions/hand_shake.csv

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from kdeepmodel.plot_util import plot_confusion_matrix
import json


def plot_gesture(data, plot_dir=None):
    index = range(1, len(data['aX']) + 1)
    plt.rcParams["figure.figsize"] = (20, 10)
    fig, axes = plt.subplots(1, 2, sharex=False, sharey=False)
    gesture_name = data['gesture'][0]
    fig.suptitle(gesture_name)
    axes[0].plot(index, data['aX'], 'g.', label='x', linestyle='solid', marker=',')
    axes[0].plot(index, data['aY'], 'b.', label='y', linestyle='solid', marker=',')
    axes[0].plot(index, data['aZ'], 'r.', label='z', linestyle='solid', marker=',')
    axes[0].set_title("Acceleration")
    axes[0].set_xlabel("Sample #")
    axes[0].set_ylabel("Acceleration (G)")
    axes[0].legend()

    axes[1].plot(index, data['gX'], 'g.', label='x', linestyle='solid', marker=',')
    axes[1].plot(index, data['gY'], 'b.', label='y', linestyle='solid', marker=',')
    axes[1].plot(index, data['gZ'], 'r.', label='z', linestyle='solid', marker=',')
    axes[1].set_title("Gyroscope")
    axes[1].set_xlabel("Sample #")
    axes[1].set_ylabel("Gyroscope (deg/sec)")
    axes[1].legend()
    if plot_dir is not None:
        save_path = os.path.join(plot_dir, "{}.png".format(gesture_name))
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def parse_motions(motion_dir, samples_per_gesture, plot_dir=None):
    file_gestures = pd.DataFrame([(csv, csv[:-4]) for csv in os.listdir(motion_dir) if csv.endswith("csv")])
    file_gestures = file_gestures.rename(columns={0: "filename", 1: "gesture"})
    num_gestures = len(file_gestures['gesture'])
    one_hot_encode_gestures = np.eye(num_gestures)
    gesture_lookup = {}
    for one_hot, gesture in zip(one_hot_encode_gestures, file_gestures['gesture']):
        gesture_lookup[gesture] = one_hot
        gesture_lookup[str(list(one_hot))] = gesture
    motion_inputs = pd.DataFrame()
    motion_outputs = None
    gesture_summary = pd.DataFrame()
    for gesture_index, gesture_row in file_gestures.iterrows():
        print("Processing index {} gesture {}".format(gesture_index, gesture_row['gesture']))
        output = one_hot_encode_gestures[gesture_index]
        df = pd.read_csv(os.path.join(motion_dir, gesture_row['filename']), header=None)
        df = df.rename(columns={0: 'aX', 1: 'aY', 2: 'aZ', 3: 'gX', 4: 'gY', 5: 'gZ'})
        df['gesture'] = gesture_row['gesture']
        if plot_dir is not None:
            plot_gesture(data=df, plot_dir=plot_dir)
        num_recording = int(df.shape[0] / samples_per_gesture)
        gesture_outputs = np.array([output] * num_recording)
        if motion_outputs is None:
            motion_outputs = gesture_outputs
        else:
            motion_outputs = np.concatenate([motion_outputs, gesture_outputs])
        motion_inputs = pd.concat([motion_inputs, df])

        this_summary = pd.DataFrame.from_dict({'gesture': [gesture_row['gesture']], 'num_recording': [num_recording]})
        gesture_summary = pd.concat([gesture_summary, this_summary])

    acceleration_min = motion_inputs.loc[:, ('aX', 'aY', 'aZ')].min().min()
    gyroscope_min = motion_inputs.loc[:, ('gX', 'gY', 'gZ')].min().min()
    motion_inputs.loc[:, 'aXnorm'] = motion_inputs.loc[:, 'aX'] - acceleration_min
    motion_inputs.loc[:, 'aYnorm'] = motion_inputs.loc[:, 'aY'] - acceleration_min
    motion_inputs.loc[:, 'aZnorm'] = motion_inputs.loc[:, 'aZ'] - acceleration_min
    motion_inputs.loc[:, 'gXnorm'] = motion_inputs.loc[:, 'gX'] - gyroscope_min
    motion_inputs.loc[:, 'gYnorm'] = motion_inputs.loc[:, 'gY'] - gyroscope_min
    motion_inputs.loc[:, 'gZnorm'] = motion_inputs.loc[:, 'gZ'] - gyroscope_min
    acceleration_max = motion_inputs.loc[:, ('aXnorm', 'aYnorm', 'aZnorm')].max().max()
    gyroscope_max = motion_inputs.loc[:, ('gXnorm', 'gYnorm', 'gZnorm')].max().max()
    motion_inputs.loc[:, ('aXnorm', 'aYnorm', 'aZnorm')] = motion_inputs.loc[:, ('aXnorm', 'aYnorm', 'aZnorm')]/acceleration_max
    motion_inputs.loc[:, ('gXnorm', 'gYnorm', 'gZnorm')] = motion_inputs.loc[:, ('gXnorm', 'gYnorm', 'gZnorm')]/gyroscope_max
    normalization_parameters = {}
    normalization_parameters['acceleration_min'] = acceleration_min
    normalization_parameters['acceleration_max'] = acceleration_max
    normalization_parameters['gyroscope_min'] = gyroscope_min
    normalization_parameters['gyroscope_max'] = gyroscope_max
    with open(os.path.join(params['output_dir'], 'normalization_parameters.json'), 'w') as wp:
        json.dump(normalization_parameters, wp)
    print("Normalized acceleration min")
    print(motion_inputs.loc[:, ('aXnorm', 'aYnorm', 'aZnorm')].min())
    print("Normalized acceleration max")
    print(motion_inputs.loc[:, ('aXnorm', 'aYnorm', 'aZnorm')].max())
    print("Normalized gyroscope min")
    print(motion_inputs.loc[:, ('gXnorm', 'gYnorm', 'gZnorm')].min())
    print("Normalized gyroscope max")
    print(motion_inputs.loc[:, ('gXnorm', 'gYnorm', 'gZnorm')].max())
    total_recordings = int(motion_inputs.shape[0]/samples_per_gesture)
    input_samples = []
    input_column_names = ('aXnorm', 'aYnorm', 'aZnorm', 'gXnorm', 'gYnorm', 'gZnorm')
    row_i = 0
    for sample_i in range(total_recordings):
        sample_input = motion_inputs[row_i:row_i+samples_per_gesture].loc[:, input_column_names].to_numpy()
        input_samples.append(sample_input)
        row_i += samples_per_gesture
    input_tensor = np.array(input_samples)

    return (input_tensor, motion_outputs, gesture_summary, gesture_lookup)


def make_gesture_model(input_shape, num_gestures):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=input_shape))  # relu is used for performance
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(15, activation='relu'))
    model.add(tf.keras.layers.Dense(num_gestures,
                                    activation='softmax'))  # softmax is used, because we only expect one gesture to occur per input
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def test_predictions(model, x_test, y_test, params, output_dir=None):
    """Test model performance and write TFlite models """
    motion_summaries = pd.read_csv(os.path.join(params['output_dir'], 'motion_summaries.csv'))
    predictions = model.predict(x_test)
    prediction_indexes = np.argmax(predictions, axis=1)
    prediction_words = []
    for pre_ind in prediction_indexes:
        prediction_words.append(motion_summaries['gesture'][pre_ind])
    y_test_indexes = np.argmax(y_test, axis=1)
    y_test_words = []
    for y_ind in y_test_indexes:
        y_test_words.append(motion_summaries['gesture'][y_ind])
    test_matrix = metrics.confusion_matrix(y_true=y_test_words, y_pred=prediction_words)
    plot_confusion_matrix(test_matrix, classes=motion_summaries['gesture'], title="Motion confusion matrix",
                          output_path=os.path.join(params['output_dir'], "{}_confusionMatrix.png".format(
                              params['model_name'])))
    accuracy = metrics.accuracy_score(y_test_words, prediction_words)
    print(test_matrix)
    print("test accuracy={}".format(accuracy))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = False
    tflite_model = converter.convert()
    with open(os.path.join(params['output_dir'], "{}.tflite".format(params['model_name'])), 'wb') as wp:
        wp.write(tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = False

    def representative_dataset_generator():
        for value in x_test:
            yield [np.array(value, dtype=np.float32, ndmin=len(motion_summaries))]
    converter.representative_dataset = representative_dataset_generator
    quantized_model = converter.convert()
    with open(os.path.join(params['output_dir'], "{}.quantized".format(params['model_name'])), "wb") as mp:
        mp.write(quantized_model)


def train(params):
    motion_input, motion_output, motion_summaries, motion_lookup = parse_motions(params['motion_dir'], params['samples_per_gesture'],
                                                                  params['output_dir'])
    motion_summaries.to_csv(os.path.join(params['output_dir'], "motion_summaries.csv"))
    with open(os.path.join(params['output_dir'], 'motion_lookup.pickle'), 'wb') as wp:
        pickle.dump(motion_lookup, wp)

    # np.save(os.path.join(params['output_dir'], "motion_input.npy"), motion_input)
    # np.save(os.path.join(params['output_dir'], "motion_output.npy"), motion_output)
    train_input, val_test_input, train_output, val_test_output = train_test_split(motion_input, motion_output,
                                                                                  test_size=params['test_split'])
    val_input, test_input, val_output, test_output = train_test_split(val_test_input, val_test_output, test_size=0.45)
    np.save(os.path.join(params['output_dir'], "test_input.npy"), test_input)
    np.save(os.path.join(params['output_dir'], "test_output.npy"), test_output)

    num_gestures = len(motion_summaries['gesture'])
    model = make_gesture_model(params['input_shape'], num_gestures)
    log_dir = os.path.join(params['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(train_input, train_output, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=(val_input, val_output), callbacks=[tensorboard_callback])
    model.save(os.path.join(params['output_dir'], params["model_name"]))


if __name__ == '__main__':
    import argparse
    params = {}
    params['motion_dir'] = '/home/ktdiedrich/Arduino/karduino/motionCapture/motions'
    params['output_dir'] = '/home/ktdiedrich/output/motionCapture'
    params['samples_per_gesture'] = 119
    params['factors'] = 6
    params['input_shape'] = (params['samples_per_gesture'], params['factors'])
    params['test_split'] = 0.30
    params['train'] = False
    params['test'] = False
    params['model_name'] = "gesture_model"
    params['epochs'] = 600
    params['batch_size'] = 10

    parser = argparse.ArgumentParser(description='Load images from ground truth sub directories for model training.')
    parser.add_argument("--train", "-t", action="store_true", default=params['train'], required=False,
                        help="train, default {}".format(params['train']))
    parser.add_argument("--test", "-T", action="store_true", default=params['test'], required=False,
                        help="test predictions, default {}".format(params['test']))
    args = parser.parse_args()
    params['train'] = args.train
    params['test'] = args.test

    SEED = 1337
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    os.makedirs(params['output_dir'], exist_ok=True)

    if params['train']:
        train(params)
    if params['test']:
        model = tf.keras.models.load_model(os.path.join(params['output_dir'], params['model_name']))
        test_input = np.load(os.path.join(params['output_dir'], "test_input.npy"))
        test_output = np.load(os.path.join(params['output_dir'], "test_output.npy"))
        test_predictions(model, test_input, test_output, params, params['output_dir'])
    print('fin')
