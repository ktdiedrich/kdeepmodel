#!/usr/bin/env python

"""
Tumor detection in 3D

Backbones
https://docs.monai.io/en/latest/networks.html#highresnet
Find 4 point box around tumor https://github.com/facebookresearch/detr
Karl T. Diedrich, PhD <ktdiedrich@gmail.com>
"""


import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import monai
from monai.config import KeysCollection
from monai.transforms.compose import MapTransform

from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, ToTensord, Transform, ToTensor
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism
import vtk
import pyvista as pv
from kdeepmodel.transformer.detr import DETR
from monai.networks.nets import SegResNet, SegResNetVAE, HighResNet
from typing import Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union


def get_files(image_dir, label_dir, file_pattern="*.nii.gz", val_split=0.20):

    train_images = sorted(glob.glob(os.path.join(image_dir, file_pattern)))
    train_labels = sorted(glob.glob(os.path.join(label_dir, file_pattern)))
    data_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_labels)]
    val_no = int(len(data_dicts) * val_split)
    train_files, val_files = data_dicts[:-val_no], data_dicts[-val_no:]
    return train_files, val_files


def find_box_label(label):
    """Finds a box around non-zero shape in a matrix of 0's
    param label: umpy or torch.Tensor """
    label = label.squeeze()
    if isinstance(label, torch.Tensor):
        label = label.numpy()
    label_at = np.where(label)
    label_range = [(np.min(dim_at), np.max(dim_at)) for dim_at in label_at]
    return label_range


class BoxTransform(MapTransform):
    """
    Calculate box coordinates around object in volume
    Based on:
    https://docs.monai.io/en/latest/transforms.html#maptransform
    https://docs.monai.io/en/latest/_modules/monai/transforms/utility/dictionary.html#ToTensord
    """
    def __init__(self, keys: KeysCollection) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`

        """
        super().__init__(keys)

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        """

        :param data:
        :return: key data['label_box'] box coordinates
        """
        d = dict(data)
        for key in self.keys:
            d['label_box'] = torch.Tensor(find_box_label(d[key]))
        return d


if __name__ == '__main__':
    import argparse
    params = {}
    params['data_root'] = '/home/ktdiedrich/data/Task01_BrainTumour'
    params['output_dir'] = '/home/ktdiedrich/output/detect_BrainTumor'
    params['model_file'] = 'best_metric_model.pth'
    params['test_val_split'] = 0.2
    params['batch_size'] = 1
    params['num_workers'] = 6
    params['epochs'] = 600
    params['learning_rate'] = 1e-5
    params['val_interval'] = 2
    params['train'] = False
    params['image_dir'] = os.path.join(params['data_root'], 'imagesTr')
    params['label_dir'] = os.path.join(params['data_root'], 'labelsTr')
    train_files, val_files = get_files(params['image_dir'], params['label_dir'])

    set_determinism(seed=0)

    train_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        BoxTransform(keys=['label']),
        # AddChanneld(keys=['image', 'label']),
        # Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.), mode=('bilinear', 'nearest')),
        # Orientationd(keys=['image', 'label'], axcodes='RAS'),
        # ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=['image', 'label'], source_key='image'),
        # RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96), pos=1,
        #                       neg=1, num_samples=4, image_key='image', image_threshold=0),
        # user can also add other random transforms
        # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
        #             rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
        ToTensord(keys=['image', 'label'])
    ])

    val_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        BoxTransform(keys=['label']),
        # AddChanneld(keys=['image', 'label']),
        # Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.), mode=('bilinear', 'nearest')),
        # Orientationd(keys=['image', 'label'], axcodes='RAS'),
        # ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        # CropForegroundd(keys=['image', 'label'], source_key='image'),
        ToTensord(keys=['image', 'label'])
    ])

    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(train_ds, batch_size=params['batch_size'])
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=params['batch_size'])

    check_data = monai.utils.misc.first(val_loader)
    label_range = find_box_label(check_data['label'])
    plt.figure(); plt.imshow(check_data['image'][0, :, :, 80, 0])
    plt.figure(); plt.imshow(check_data['label'][0, :, :, 80])
    image_array = check_data['image'].numpy()
    label_array = check_data['label'].numpy()
    label_at = np.where(label_array[0])
    label_box = label_array[0][label_range[0][0]:label_range[0][1], label_range[1][0]:label_range[1][1],
                label_range[2][0]:label_range[2][1]]
    image_vols = [pv.wrap(image_array[0, :, :, :,ci]) for ci in range(image_array.shape[4])]
    label_vol = pv.wrap(label_array[0, :, :, :])
    # label_vol.plot(volume=True)
    label_box_vol = pv.wrap(label_box)
    # label_box_vol.plot(volume=True)
    #for vol in image_vols:
    #    vol.plot(volume=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    backboneSegResNet = SegResNet(spatial_dims=3, in_channels=1, out_channels=2)
#    backboneSegResNetVAE = SegResNetVAE(spatial_dims=3, in_channels=1, out_channels=2)
    backboneHighResNet = HighResNet(spatial_dims=3, in_channels=1, out_channels=2)

    backboneHighResNet.to(device)
    input_data = check_data['image'].permute(4, 0, 1, 2, 3).to(device)
    output = backboneHighResNet(input_data[:, :, 0:50, 0:50, 0:50])
    plt.imshow(output[0, 0, :, :, 25].detach().cpu())

    train_iter = iter(train_ds)
    train_batch = next(train_iter)
    val_iter = iter(val_loader)
    val_batch = next(val_iter)

    plt.show()
    print('fin')
