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
* Sample data https://www.kaggle.com/kmader/colorectal-histology-mnist
Karl Diedrich, PhD <ktdiedrich@gmail.com>
"""


import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from fastai.vision import *
import json


param = dict()
param['test_size'] = 0.2
param['show'] = False
param['print'] = False
param['epochs'] = 40
param['batch_size'] = 32
param['output_dir'] = '.'
param['model_output_name'] = "trained_model.pth"
param['figure_name'] = 'training_history.png'
param['validation_split'] = 0.2
param['figure_size'] = (9, 9)
param['learning_rate'] = 0.0075
param['dropout'] = 0.5
param['sample_size'] = 4


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Load images from ground truth sub directories for model training.')
    parser.add_argument('input_dir', type=str, help='input image top directory')
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

    param['input_dir'] = args.input_dir
    param['test_size'] = args.test_size
    param['epochs'] = args.epochs
    param['batch_size'] = args.batch_size
    param['show'] = args.show
    param['print'] = args.print
    param['output_dir'] = args.output_dir

    if not os.path.exists(param['output_dir']):
        os.makedirs(param['output_dir'])
    with open(os.path.join(param['output_dir'], 'param.json'), 'w') as fp:
        json.dump(param, fp)

    # X = np.load(args.x_input).astype(np.float)
    # Y = np.load(args.y_ground).astype(np.float)

    data = ImageDataBunch.from_folder(param['input_dir'])
    if param['show']:
        img, label = data.train_ds[0]
        img.show(figsize=(3, 3), title="{}".format(label))

    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.fit(param['epochs'])
    learn.save(os.path.join(param['output_dir'], param['model_output_name']))
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix()
    interp.plot_top_losses(16, figsize=(12, 8))

