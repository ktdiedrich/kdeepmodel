#!/usr/bin/env python

""" Segmentation
https://github.com/Project-MONAI/MONAI/blob/master/examples/notebooks/spleen_segmentation_3d.ipynb

data http://medicaldecathlon.com/

Brain segmentation https://github.com/Project-MONAI/tutorials/blob/master/3d_segmentation/brats_segmentation_3d.ipynb
"""

import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import monai
from monai.transforms import \
    Compose, LoadNiftid, AddChanneld, ScaleIntensityRanged, CropForegroundd, \
    RandCropByPosNegLabeld, RandAffined, Spacingd, Orientationd, ToTensord
from monai.inferers import sliding_window_inference
from monai.networks.layers import Norm
from monai.metrics import compute_meandice
from monai.utils import set_determinism
import json
from kdeepmodel.mednist_monai import show_batch_samples
import tensorboardX


def get_files(image_dir, label_dir, file_pattern="*.nii.gz"):

    train_images = sorted(glob.glob(os.path.join(image_dir, file_pattern)))
    train_labels = sorted(glob.glob(os.path.join(label_dir, file_pattern)))
    data_dicts = [{'image': image_name, 'label': label_name}
                  for image_name, label_name in zip(train_images, train_labels)]
    train_files, val_files = data_dicts[:-9], data_dicts[-9:]
    return train_files, val_files


def plot_metrics(epoch_loss_values, metric_values, val_interval, output_path=None):
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Epoch Average Loss')
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title('Val Mean Dice')
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()


def train(model, train_loader, val_loader, loss_function, optimizer, output_dir, device, epoch_num=600, val_interval=2):
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(epoch_num):
        print('-' * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data['image'].to(device), batch_data['label'].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                metric_sum = 0.
                metric_count = 0
                for val_data in val_loader:
                    val_inputs, val_labels = val_data['image'].to(device), val_data['label'].to(device)
                    roi_size = (160, 160, 160)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model)
                    value = compute_meandice(y_pred=val_outputs, y=val_labels, include_background=False,
                                             to_onehot_y=True, mutually_exclusive=True)
                    metric_count += len(value)
                    metric_sum += value.sum().item()
                metric = metric_sum / metric_count
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_metric_model.pth'))
                    print('saved new best metric model')
                print(f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                      f"\nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
    metric_output = os.path.join(output_dir, "training_metrics.png")
    plot_metrics(epoch_loss_values, metric_values, val_interval, output_path=metric_output)


def evaluate(model_path, val_loader, plot_path=None):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_data['image'].to(device), roi_size, sw_batch_size, model)
            # plot the slice [:, :, 80]
            plt.figure('check', (18, 6))
            plt.subplot(1, 3, 1)
            plt.title(f"image {str(i)}")
            plt.imshow(val_data['image'][0, 0, :, :, 80], cmap='gray')
            plt.subplot(1, 3, 2)
            plt.title(f"label {str(i)}")
            plt.imshow(val_data['label'][0, 0, :, :, 80])
            plt.subplot(1, 3, 3)
            plt.title(f"output {str(i)}")
            plt.imshow(torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, 80])
            if plot_path is not None:
                fig_path = os.path.join(plot_path, "val_{}_evaluation.png".format(i))
                plt.savefig(fig_path)
                print("Saved {}".format(fig_path))
            else:
                plt.show()


if __name__ == '__main__':
    params = {}
    # params['data_root'] = '/home/ktdiedrich/data/Task09_Spleen'
    # params['output_dir'] = '/home/ktdiedrich/output/segment_spleen_monai'
    params['data_root'] = '/home/ktdiedrich/data/Task09_Spleen'
    params['output_dir'] = '/home/ktdiedrich/output/segment_spleen_monai'
    params['model_file'] = 'best_metric_model.pth'
    params['test_val_split'] = 0.2
    params['batch_size'] = 300
    params['num_workers'] = 6
    params['epochs'] = 600
    params['learning_rate'] = 1e-5
    params['val_interval'] = 2
    params['train'] = False

    monai.config.print_config()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    os.makedirs(params['output_dir'], exist_ok=True)

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    train_files, val_files = get_files(os.path.join(params['data_root'], 'imagesTr'),
                                       os.path.join(params['data_root'], 'labelsTr'),
                                       file_pattern='*.nii.gz')
    set_determinism(seed=0)

    val_transforms = Compose([
        LoadNiftid(keys=['image', 'label']),
        AddChanneld(keys=['image', 'label']),
        Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.), mode=('bilinear', 'nearest')),
        Orientationd(keys=['image', 'label'], axcodes='RAS'),
        ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=['image', 'label'], source_key='image'),
        ToTensord(keys=['image', 'label'])
    ])

    check_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    check_loader = monai.data.DataLoader(check_ds, batch_size=1)
    check_data = monai.utils.misc.first(check_loader)
    # plt.imshow(check_data['image'][0, 0, :, :, 80])
    # plt.imshow(check_data['label'][0, 0, :, :, 80])

    val_ds = monai.data.CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=1.0, num_workers=params['num_workers']
    )
    # val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(val_ds, batch_size=1, num_workers=params['num_workers'])

    # standard PyTorch program style: create UNet, DiceLoss and Adam optimizer
    # https://docs.monai.io/en/latest/networks.html#unet
    model = monai.networks.nets.UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                                     strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
    loss_function = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])

    if params['train']:
        train_transforms = Compose([
            LoadNiftid(keys=['image', 'label']),
            AddChanneld(keys=['image', 'label']),
            Spacingd(keys=['image', 'label'], pixdim=(1.5, 1.5, 2.), mode=('bilinear', 'nearest')),
            Orientationd(keys=['image', 'label'], axcodes='RAS'),
            ScaleIntensityRanged(keys=['image'], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image', 'label'], source_key='image'),
            RandCropByPosNegLabeld(keys=['image', 'label'], label_key='label', spatial_size=(96, 96, 96), pos=1,
                                   neg=1, num_samples=4, image_key='image', image_threshold=0),
            # user can also add other random transforms
            # RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(96, 96, 96),
            #             rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
            ToTensord(keys=['image', 'label'])
        ])
        train_ds = monai.data.CacheDataset(
            data=train_files, transform=train_transforms, cache_rate=1.0, num_workers=params['num_workers']
        )
        # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

        # use batch_size=2 to load images and use RandCropByPosNegLabeld
        # to generate 2 x 4 images for network training
        train_loader = monai.data.DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=params['num_workers'])

        train(model, train_loader, val_loader, loss_function, optimizer, params['output_dir'], device,
              epoch_num=params['epochs'], val_interval=params['val_interval'])

    evaluate(os.path.join(params['output_dir'], 'best_metric_model.pth'), val_loader=val_loader,
             plot_path=os.path.join(params['output_dir']))

    print('fin')


