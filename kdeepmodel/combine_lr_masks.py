#!/usr/bin/env python

"""
Combine left and right masks images
"""

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
import os


def load_image_tensor(image_path):
    img = Image.open(image_path)
    img_t = to_tensor(img)
    return img_t


def list_filenames(path2imgs, extension="png"):
    img_filenames = [pp for pp in os.listdir(path2imgs) if pp.endswith(extension)]
    return img_filenames


def combine_masks(left_path, right_path, output_dir):
    left_filenames = list_filenames(left_path)
    right_filenames = list_filenames(right_path)
    assert len(left_filenames) == len(right_filenames)
    for left_name, right_name in zip(left_filenames, right_filenames):
        left_tensor = load_image_tensor(os.path.join(left_path, left_name))
        right_tensor = load_image_tensor(os.path.join(right_path, right_name))
        both_tensor = left_tensor + right_tensor
        both_image = to_pil_image(both_tensor)
        base_name, ext = os.path.splitext(left_name)
        both_name = "{}_mask{}".format(base_name, ext)
        both_image.save(os.path.join(output_dir, both_name))


if __name__ == '__main__':
    import argparse
    import json


    params = {}
    params['left'] = '/home/ktdiedrich/data/lung-xray/MontgomerySet/ManualMask/leftMask'
    params['right'] = '/home/ktdiedrich/data/lung-xray/MontgomerySet/ManualMask/rightMask'
    params['output_dir'] = '/home/ktdiedrich/data/lung-xray/MontgomerySet/ManualMask/bothMask'

    parser = argparse.ArgumentParser(description='Combine left and right mask images')
    parser.add_argument('--left', '-l', type=str, help='left mask path, {}'.format(params['left']),
                        default=params['left'])
    parser.add_argument('--right', '-r', type=str, help='right mask path, {}'.format(params['right']),
                        default=params['right'])
    parser.add_argument("--output_dir", "-o", type=str, help="output directory, {}".format(params['output_dir']),
                        default=params['output_dir'])
    args = parser.parse_args()

    params['left'] = args.left
    params['right'] = args.right
    params['output_dir'] = args.output_dir

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    combine_masks(params['left'], params['right'], params['output_dir'])


