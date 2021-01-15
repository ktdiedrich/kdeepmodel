#!/usr/bin/env python

"""
https://colab.research.google.com/drive/1wy8XUSnNWlhDNazFdvGBHLfdkGvOHBKe
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import torch
# from torch.utils.data import Dataset, DataLoader
import monai
from monai.data import Dataset, DataLoader

from monai.config import print_config
from monai.transforms import \
    Compose, LoadPNG, AddChannel, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom
from monai.networks.nets import densenet121
from monai.metrics import compute_roc_auc
from sklearn.metrics import classification_report


LABEL = "label"
FILE = "file"
LABEL_ID = "one_hot"
FACTOR = 'factor'


def get_train_val_test_data(data_dir, test_val_split=0.2):
    """Read image file paths in class name sub directories"""
    class_names = [x for x in os.listdir(params['data_dir']) if
                   os.path.isdir(os.path.join(params['data_dir'], x))]

    class_imagefiles = {}
    for class_name in class_names:
        class_imagefiles[class_name] = [os.path.join(data_dir, class_name, x)
                                        for x in os.listdir(os.path.join(data_dir, class_name))]
    data = pd.concat({k: pd.DataFrame({LABEL: k, FILE: v}) for k, v in class_imagefiles.items()})
    data[FACTOR] = data[LABEL].factorize()[0]
    data[LABEL_ID] = list(to_categorical(data[FACTOR], len(data[FACTOR].unique())))
    train_input, test_val_data = train_test_split(data, test_size=test_val_split)
    val_data, test_data = train_test_split(test_val_data, test_size=0.5)
    return train_data, val_data, test_data, data


def plot_metrics(epoch_loss_values, metric_values):
    plt.figure('train', (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.subplot(1, 2, 2)
    plt.title("Validation: Area under the ROC curve")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel('epoch')
    plt.plot(x, y)
    plt.show()


class LabeledImageDataset(Dataset):
    def __init__(self, dataframe, transforms):
        self._dataframe = dataframe
        self._transforms = transforms

    def __len__(self):
        return len(self._dataframe)

    def __getitem__(self, item):
        # img = np.array(Image.open(self._dataframe[FILE][item]))
        img = self._transforms(self._dataframe[FILE][item])
        label = self._dataframe[FACTOR][item]
        return img, label


def show_sample_dataframe(sample_data, size_n=3, title=None):
    fig, ax = plt.subplots(3, 3, figsize=(16, 16))
    for r in range(size_n):
        for c in range(size_n):
            idx = r * size_n + c
            class_name = sample_data[LABEL][idx]
            file_path = sample_data[FILE][idx]
            img = np.array(Image.open(file_path))
            ax[r][c].imshow(img)
            ax[r][c].set_title(class_name)
    if title is not None:
        plt.suptitle(title)
    plt.show()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def collate_fn(batch):
    """Collate batches of images together.
    """
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs)
    targets = torch.LongTensor(targets)
    return imgs, targets


def train(train_loader, optimizer, loss_function, epoch_num, model_name, output_dir="."):
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
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            epoch_len = len(train_ds) // train_loader.batch_size
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    y_pred = torch.cat([y_pred, model(val_images)], dim=0)
                    y = torch.cat([y, val_labels], dim=0)
                auc_metric = compute_roc_auc(y_pred, y, to_onehot_y=True, softmax=True)
                metric_values.append(auc_metric)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                if auc_metric > best_metric:
                    best_metric = auc_metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(output_dir, model_name))
                    print('saved new best metric model')
                print(f"current epoch: {epoch + 1} current AUC: {auc_metric:.4f}"
                      f" current accuracy: {acc_metric:.4f} best AUC: {best_metric:.4f}"
                      f" at epoch: {best_metric_epoch}")
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    return epoch_loss_values, metric_values


def show_batch_samples(batch_img, batch_label, size_n=3, title=None):
    fig, ax = plt.subplots(size_n, size_n, figsize=(16, 16))
    for r in range(size_n):
        for c in range(size_n):
            idx = r * size_n + c
            img = batch_img[idx, 0, :, :]
            label = batch_label[idx]
            ax[r][c].imshow(img)
            ax[r][c].set_title(label)
    if title is not None:
        plt.suptitle(title)
    plt.show()


if __name__ == '__main__':
    params = {}
    params['data_dir'] = '/home/ktdiedrich/data/MedNIST'
    params['output_dir'] = '/home/ktdiedrich/output/MedNist_monai'
    params['model_file'] = 'best_metric_model.pth'
    params['test_val_split'] = 0.2
    params['batch_size'] = 300
    params['num_workers'] = 10
    params['epochs'] = 4
    params['learning_rate'] = 1e-5
    params['val_interval'] = 1
    params['rotate_range_x'] = 15
    params['rotate_prob'] = 0.5
    params['min_zoom'] = 0.9
    params['max_zoom'] = 1.1
    params['zoom_prob'] = 0.5

    monai.config.print_config()

    train_data, val_data, test_data, raw_data = get_train_val_test_data(params['data_dir'], params['test_val_split'])

    size_n = 3
    sample_data = train_data.sample(size_n ** 2)
    #show_sample_dataframe(sample_data, size_n=size_n, title="Training samples")

    train_transforms = Compose([
        LoadPNG(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        RandRotate(range_x=params['rotate_range_x'], prob=params['rotate_prob'], keep_size=True),
        # RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=params['min_zoom'], max_zoom=params['max_zoom'], prob=params['zoom_prob'], keep_size=True),
        ToTensor()
    ])

    val_transforms = Compose([
        LoadPNG(image_only=True),
        AddChannel(),
        ScaleIntensity(),
        ToTensor()
    ])
    train_ds = LabeledImageDataset(train_data, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'],
                              # num_workers=params['num_workers'], collate_fn=collate_fn)
                              num_workers=params['num_workers'])

    val_ds = LabeledImageDataset(val_data, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=params['batch_size'],
                              # num_workers=params['num_workers'], collate_fn=collate_fn)
                            num_workers=params['num_workers'])

    test_ds = LabeledImageDataset(test_data, train_transforms)
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'],
                              # num_workers=params['num_workers'], collate_fn=collate_fn)
                             num_workers=params['num_workers'])

    #train_batch_img, train_batch_labels = iter(train_loader).next()
    #show_batch_samples(train_batch_img, train_batch_labels, size_n=3, title='Training batch samples')

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = "cpu"

    num_class = len(train_data[LABEL].unique())
    model = densenet121(
        spatial_dims=2,
        in_channels=1,
        out_channels=num_class
    ).to(device)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), params['learning_rate'])
    epoch_num = params['epochs']
    val_interval = params['val_interval']

    os.makedirs(params['output_dir'], exist_ok=True)

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    loss_values, metric_values = train(train_loader, optimizer, loss_function, epoch_num, model_name=params['model_file'],
          output_dir=params['output_dir'])

    model_path = os.path.join(params['output_dir'], params['model_file'])

    plot_metrics(loss_values, metric_values)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for test_data in test_loader:
            test_images, test_labels = test_data[0].to(device), test_data[1].to(device)
            pred = model(test_images).argmax(dim=1)
            for i in range(len(pred)):
                y_true.append(test_labels[i].item())
                y_pred.append(pred[i].item())
    class_names = raw_data[LABEL].unique()
    print(raw_data.loc[:, [LABEL, FACTOR]].drop_duplicates())
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print('fin')
