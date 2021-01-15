#!/usr/bin/env python

import torch
from kdeepmodel.segment.resnet.segment_image import tensor_image, mask_segmentation
from pathlib import Path
import numpy as np


class Segmentation:
    def __init__(self, model_path, width, height):
        self._model = torch.jit.load(model_path)
        self._width = width
        self._height = height
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def predict(self, image_path):
        img_t = tensor_image(image_path, self._width, self._height, self._device)
        prediction = self._model(img_t)['out']
        return prediction, img_t


def image_files(tree_root, patterns=['*.jp*g', '*.png']):
    files = []
    for pt in patterns:
        files += list(Path(tree_root).rglob(pt))
    return files


if __name__ == '__main__':
    import argparse
    import json
    import os

    params = {}
    params['input_dir'] = None
    params['model_path'] = '/home/ktdiedrich/output/lung-xray/models/all_lung_2020.05.25/all_weights_2020.05.25.ts.zip'
    params['output_dir'] = '.'
    params['height'] = 520
    params['width'] = 520
    params['num_classes'] = 2
    params['threshold'] = -1.0
    parser = argparse.ArgumentParser(description='Segment objects from files in directory tree')
    parser.add_argument('input_dir', type=str, help='dir path to recursively segment images')
    parser.add_argument('--model_path', "-m", type=str, help='Torchscript model path [{}]'.format(params['model_path']),
                        default=params['model_path'])
    parser.add_argument("--output_dir", "-o", type=str, help="output directory [{}]".format(params['output_dir']),
                        default=params['output_dir'])
    # parser.add_argument('--model', "-m", type=str,
    #                     help='model, {}'.format(params['model']), default=params['model'])
    parser.add_argument('--height', "-H", type=str,
                        help='height, {}'.format(params['height']), default=params['height'])
    parser.add_argument('--width', "-W", type=str,
                        help='width, {}'.format(params['width']), default=params['width'])
    parser.add_argument('--num_classes', "-n", type=str,
                        help='number of classes, {}'.format(params['num_classes']), default=params['num_classes'])
    parser.add_argument('--threshold', "-t", type=str,
                        help='threshold, {}'.format(params['threshold']), default=params['threshold'])
    args = parser.parse_args()

    params['input_dir'] = args.input_dir
    params['model_path'] = args.model_path
    params['output_dir'] = args.output_dir
    params['height'] = args.height
    params['width'] = args.width
    params['threshold'] = args.threshold

    os.makedirs(params['output_dir'], exist_ok=True)

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)
    file_paths = image_files(params['input_dir'])
    segmentation = Segmentation(params['model_path'], params['width'], params['height'])
    for file_path in file_paths:
        seg, input_tensor = segmentation.predict(file_path)
        masked = mask_segmentation(seg, input_tensor, params['threshold'])
        file_dir = os.path.dirname(file_path)
        output_subdir = file_dir[len(params['input_dir'])+1:]
        file_output_dir = os.path.join(params['output_dir'], output_subdir)
        os.makedirs(file_output_dir, exist_ok=True)
        file_name = os.path.basename(file_path)
        file_base = os.path.splitext(file_name)[0]
        masked_file_path = os.path.join(file_output_dir, "{}_masked.npy".format(file_base))
        with open(masked_file_path, "wb") as fp:
            np.save(fp, masked.numpy())

    # seg_prediction, input_t, weighted_model = predict_segmentation_by_script(params)

    # plot_batch(input_t)
    # plt.figure()
    # plot_batch(seg_prediction, [1])
    # seg_image = mask_segmentation(seg_prediction, input_t, params['threshold'])
    # plt.figure()
    # plt.imshow(seg_image)
    # plt.show()
    # plot_batch(segmentation, list(range(params['num_classes'])))
    print("fin")

