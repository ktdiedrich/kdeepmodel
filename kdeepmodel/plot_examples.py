#!/usr/bin/env python

"""
Plot random examples from files in directory

:author: Karl Diedrich, PhD <ktdiedrich@gmail.com>
"""

from kdeepmodel.plot_util import make_file_image_grid, get_file_paths
import os
import numpy as np
from random import sample
import matplotlib.pyplot as plt


if __name__ == '__main__':
    import argparse

    params = {}
    params['input_dir'] = None
    params['output_file'] = None
    params['extension'] = "npy"
    params['num_examples'] = 40
    parser = argparse.ArgumentParser(description='Plot example images from files')
    parser.add_argument('input_dir', type=str, help='dir path to recursively segment images')
    parser.add_argument('--output_file', "-o", type=str, help='Output file, or plot to screen [{}]'.format(
        params['output_file']), default=params['output_file'])
    parser.add_argument("--extension", "-e", type=str, help="output directory [{}]".format(params['extension']),
                        default=params['extension'])
    parser.add_argument('--num_examples', "-n", type=str,
                        help='number of examples, {}'.format(params['num_examples']), default=params['num_examples'])
    args = parser.parse_args()

    params['input_dir'] = args.input_dir
    params['extension'] = args.extension
    params['num_examples'] = args.num_examples
    params['output_file'] = args.output_file

    file_paths = get_file_paths(params['input_dir'], params['num_examples'], params['extension'])
    load = np.load
    if params['extension'] == 'npy':
        load = np.load
    file_examples = sample(file_paths, params['num_examples'])
    image_grid = make_file_image_grid(file_examples, load=load)
    if params['output_file'] is not None:
        output_dir = os.path.dirname(params['output_file'])
        os.makedirs(output_dir, exist_ok=True)
    else:
        plt.imshow(image_grid.permute((1, 2, 0))[:, :, 0])
        plt.show()
    print("fin")
