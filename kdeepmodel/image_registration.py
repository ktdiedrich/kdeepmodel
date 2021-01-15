#!/usr/bin/env python

# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image registration. Functionalized.
Based on https://github.com/airlab-unibas/airlab/blob/master/examples/diffeomorphic_bspline_2d.py
Karl T. Diedrich, PhD <ktdiedrich@gmail.com>
"""


import sys
import os
import time

import matplotlib.pyplot as plt
import torch as th
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import airlab as al

from create_test_image_data import create_C_2_O_test_images
import cv2
import SimpleITK as sitk


def read_image(image_path):
    ext = os.path.splitext(image_path)[1]
    if ext in [".jpg", ".jpeg", ".JPG", ".JPEG"]:
        io_reader = "JPEGImageIO"
    else:
        io_reader = "PNGImageIO"
    reader = sitk.ImageFileReader()
    reader.SetImageIO(io_reader)
    reader.SetFileName(image_path)
    image = reader.Execute()
    return image


def resize_image(input_img, ref_img, interpolator=sitk.sitkLinear, show=False):
    input_size = input_img.GetSize()
    output_size = ref_img.GetSize()
    input_spacing = input_img.GetSpacing()
    output_spacing = [0.0]*ref_img.GetDimension()
    output_spacing[0] = input_spacing[0] * (input_size[0] / output_size[0]);
    output_spacing[1] = input_spacing[1] * (input_size[1] / output_size[1]);

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(output_size)
    resampler.SetOutputSpacing(output_spacing)
    resampler.SetOutputOrigin(ref_img.GetOrigin())
    resampler.SetOutputDirection(ref_img.GetDirection())
    resampler.SetInterpolator(interpolator)
    resampler.SetDefaultPixelValue(0)
    image = resampler.Execute(input_img)
    if show:
        image_array = sitk.GetArrayFromImage(image)
        plt.imshow(image_array[:, :, 0])
        plt.show()
    return image


def register_images(fixed_image, moving_image, shaded_image=None):
    start = time.time()

    # create image pyramid size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(fixed_image, [[4, 4], [2, 2]])
    moving_image_pyramid = al.create_image_pyramid(moving_image, [[4, 4], [2, 2]])

    constant_flow = None
    # regularisation_weight = [1, 5, 50]
    # number_of_iterations = [500, 500, 500]
    # sigma = [[11, 11], [11, 11], [3, 3]]

    regularisation_weight = [1, 5, 50]
    number_of_iterations = [10, 10, 10]
    sigma = [[11, 11], [11, 11], [3, 3]]

    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

        registration = al.PairwiseRegistration(verbose=True)

        # define the transformation
        transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                          sigma=sigma[level],
                                                                          order=3,
                                                                          dtype=fixed_image.dtype,
                                                                          device=fixed_image.device)

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                          mov_im_level.size,
                                                                          interpolation="linear")
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)

        registration.set_image_loss([image_loss])

        # define the regulariser for the displacement
        regulariser = al.regulariser.displacement.DiffusionRegulariser(mov_im_level.spacing)
        regulariser.SetWeight(regularisation_weight[level])

        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = th.optim.Adam(transformation.parameters())

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        constant_flow = transformation.get_flow()

    # create final result
    displacement = transformation.get_displacement()
    if shaded_image is not None:
        warped_image = al.transformation.utils.warp_image(shaded_image, displacement)
    else:
        warped_image = al.transformation.utils.warp_image(moving_image, displacement)
    displacement_image = al.create_displacement_image_from_image(displacement, moving_image)

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start)
    print("Result parameters:")
    return warped_image, displacement_image


def read_image_pair(fixed_path, moving_path):
    fixed_np = cv2.imread(fixed_path)
    moving_np = cv2.imread(moving_path)
    image_size = fixed_np.shape[0:2]
    image_spacing = (1.0, 1.0)
    image_origin = (0.0, 0.0)
    moving_np_resized = cv2.resize(moving_np, image_size)

    fixed_np_bw = np.average(fixed_np, 2).astype(np.float32)
    moving_np_resized_bw = np.average(moving_np_resized, 2).astype(np.float32)
    fixed_image = al.Image(fixed_np_bw, image_size, image_spacing, image_origin)
    moving_image = al.Image(moving_np_resized_bw, image_size, image_spacing, image_origin)
    return fixed_image, moving_image


def main():
    import argparse

    params = {}
    # params['fixed'] = 'data/affine_test_image_2d_fixed.png'
    # params['moving'] = 'data/affine_test_image_2d_moving.png'
    params['fixed'] = 'data/under.jpg'
    params['moving'] = 'data/bluedress.jpeg'
    params['test'] = False
    parser = argparse.ArgumentParser(description='DICOM images')

    parser.add_argument('--fixed', '-f', type=str, help='fixed image [{}]'.format(params['fixed']),
                        default=params['fixed'])
    parser.add_argument('--moving', "-m", type=str, help='moving image [{}]'.format(params['moving']),
                        default=params['moving'])
    args = parser.parse_args()

    # set the used data type
    dtype = th.float32
    if th.cuda.is_available():
        device = th.device("cuda:0")
    else:
        device = "cpu"

    if not params['test']:
        fixed_image, moving_image = read_image_pair(args.fixed, args.moving)
        shaded_image = None
    else:
        created_fixed_image, created_moving_image, created_shaded_image = create_C_2_O_test_images(256, dtype=dtype,
                                                                                                   device=device)
        fixed_image = created_fixed_image
        moving_image = created_moving_image
        shaded_image = created_shaded_image

    warped_image, displacement_image = register_images(fixed_image, moving_image, shaded_image)

    # plot the results
    plt.subplot(221)
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(222)
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(223)
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.subplot(224)
    plt.imshow(displacement_image.magnitude().numpy(), cmap='jet')
    plt.title('Magnitude Displacement')

    plt.show()


if __name__ == '__main__':
    main()
