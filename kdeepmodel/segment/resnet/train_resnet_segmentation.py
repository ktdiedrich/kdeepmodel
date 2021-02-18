#!/usr/bin/env python
# coding: utf-8

"""Train mask segmentation
Karl T. Diedrich <ktdiedrich@gmail.com>
"""
from torch.utils.data import Dataset
from PIL import Image
from torchsummary import summary
from torchvision.transforms.functional import to_tensor, to_pil_image
from scipy import ndimage as ndi
import os
import torch
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as plt
# from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import copy
from torch import optim
from torch import nn
# from albumentations import (
#     HorizontalFlip,
#     Compose,
#     Resize,
#     Normalize)
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from sklearn.model_selection import ShuffleSplit
from torch.utils.data import Subset
import pandas as pd
from kdeepmodel.plot_util import plot_batch, plot_loss
import re
from scipy.spatial.distance import directed_hausdorff


np.random.seed(0)
num_classes = 2
COLORS = np.random.randint(0, 2, size=(num_classes+1, 3), dtype="uint8")
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
mean = [0.485]
std = [0.229]
h, w = 520, 520


class SegmentationDataset(Dataset):
    def __init__(self, path2img, path2masks, transform=None, mask_transform=None, extension="png"):
        """
        Read file x-ray images and return image items.
        :param path2img: image file directory
        :param path2masks: mask file directory
        :param transform:
        :param extension:
        """
        img_filenames = [pp for pp in os.listdir(path2img) if pp.endswith(extension)]
        mask_filenames = [pp for pp in os.listdir(path2masks) if pp.endswith(extension)]
        if re.search("mask", mask_filenames[0]) is not None:
            mask_names = [ant.replace("_mask.{}".format(extension), "") for ant in mask_filenames]
        else:
            mask_names = [ant.replace(".{}".format(extension), "") for ant in mask_filenames]
        img_names = [img.replace(".{}".format(extension), "") for img in img_filenames]

        img_frame = pd.DataFrame({"sample_name": img_names, "img_filename": img_filenames})
        mask_frame = pd.DataFrame({"sample_name": mask_names, "mask_filename": mask_filenames})
        self.data_frame = pd.merge(img_frame, mask_frame, on="sample_name")
        self.data_frame['img_path'] = path2img + os.path.sep + self.data_frame['img_filename']
        self.data_frame['mask_path'] = path2masks + os.path.sep + self.data_frame['mask_filename']
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        path2img = self.data_frame['img_path'][idx]
        image = Image.open(path2img)

        path2mask = self.data_frame['mask_path'][idx]
        mask = Image.open(path2mask)
        if self.transform:
            augmented = {}
            augmented['image'] = self.transform(image)
            augmented['mask'] = self.mask_transform(mask)
            image = augmented['image']
            mask = augmented['mask']

        mask = mask.long()
        # mask = 255 * mask
        if image.shape[0] == 1:
            image = torch.cat(3*[image], dim=0)
        # else:
        #     print("image shape {}".format(image.shape))
        return image, mask


def collate_fn(batch):
    """Collate batches of images together.
    """
    imgs, targets = list(zip(*batch))
    imgs = torch.stack(imgs)
    targets = torch.stack(targets).squeeze(1)
    return imgs, targets


def show_img_target(img, target, num_classes=num_classes):
    if torch.is_tensor(img):
        img = to_pil_image(img)
        target = target.numpy()
    for ll in range(num_classes):
        mask = (target == ll)
        img = mark_boundaries(np.array(img),
                            mask,
                            outline_color=COLORS[ll],
                            color=COLORS[ll])
    plt.imshow(img)


def re_normalize(x, mean=mean, std=std):
    x_r = x.clone()
    for c, (mean_c, std_c) in enumerate(zip(mean, std)):
        x_r[c] *= std_c
        x_r[c] += mean_c
    return x_r


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_batch(loss_func, output, target, opt=None):
    loss = loss_func(output, target)

    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item(), None


def loss_epoch(model, loss_func, dataset_dl, device, opt=None):
    running_loss = 0.0
    len_data = len(dataset_dl.dataset)
    first_xb = None
    first_yb = None
    first_output = None
    for xb, yb in dataset_dl:
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)["out"]
        if first_xb is None:
            first_xb = xb
            first_yb = yb
            first_output = output
        loss_b, _ = loss_batch(loss_func, output, yb, opt)
        running_loss += loss_b
        # plot_batch(xb); plt.figure(); plot_batch(yb); plt.figure(); plot_batch(output, [0, 1])
    loss = running_loss / float(len_data)

    return loss, first_xb, first_yb, first_output


def train_val(model, params, device):
    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    path2weights = params["path2weights"]

    loss_history = {
        "train": [],
        "val": []}

    metric_history = {
        "train": [],
        "val": []}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        current_lr = get_lr(opt)
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, current_lr))

        model.train()
        output_check_dir = None
        if not epoch % params['checkpoint_epochs']:
            output_check_dir = params['output_dir']
        train_loss, train_xb_sample, train_yb_sample, train_output_sample = loss_epoch(model, loss_func, train_dl, device, opt)

        loss_history["train"].append(train_loss)
        # metric_history["train"].append(train_metric)

        model.eval()
        with torch.no_grad():
            val_loss, val_xb_sample, val_yb_sample, val_output_sample = loss_epoch(model, loss_func, val_dl, device)
        loss_history["val"].append(val_loss)
        # metric_history["val"].append(val_metric)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            print("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        print("train loss: %.6f" % (train_loss))
        print("valid loss: %.6f" % (val_loss))
        print("-" * 10)
        if output_check_dir is not None:
            plot_batch(train_xb_sample, [0], os.path.join(output_check_dir, "epoch{}.train_input.png".format(epoch)))
            plot_batch(train_yb_sample, [0], os.path.join(output_check_dir, "epoch{}.train_ground_truth.png".format(epoch)))
            plot_batch(train_output_sample, [0, 1], os.path.join(output_check_dir,
                                                             "epoch{}.train_prediction.png".format(epoch)))

            plot_loss(loss_hist=loss_history, num_epochs=epoch+1, output_dir=output_check_dir,
                      output_name="epoch{}.loss_history.png".format(epoch))
            plot_batch(val_xb_sample, [0], os.path.join(output_check_dir, "epoch{}.val_input.png".format(epoch)))
            plot_batch(val_yb_sample, [0], os.path.join(output_check_dir, "epoch{}.val_ground_truth.png".format(epoch)))
            plot_batch(val_output_sample, [0, 1], os.path.join(output_check_dir,
                                                             "epoch{}.val_prediction.png".format(epoch)))
    model.load_state_dict(best_model_wts)
    return model, loss_history, metric_history


def main():
    import argparse
    import json
    import os
    params = {}
    params['num_workers'] = 6
    params['learning_rate'] = 1e-6
    params['momentum'] = 0.9
    params['weight_decay'] = 0.0005
    params['gamma'] = 0.1
    params['step_size'] = 3
    params['num_classes'] = 2
    params['epochs'] = 50
    params['batch_size'] = 4
    params['patience'] = 20
    params['test_size'] = 0.2
    params['image_dir'] = '/home/ktdiedrich/data/lung-xray/CXR_pngs'
    params['mask_dir'] = '/home/ktdiedrich/data/lung-xray/masks'
    params['output_dir'] = '/home/ktdiedrich/output/lung-xray/models2'
    params['checkpoint_epochs'] = 5
    parser = argparse.ArgumentParser(description='Detect objects')
    parser.add_argument("--output_dir", "-o", type=str, help="output directory [{}]".format(params['output_dir']),
                        default=params['output_dir'])
    parser.add_argument('--image_dir', "-i", type=str,
                        help='images [{}]'.format(params['image_dir']), default=params['image_dir'])
    parser.add_argument('--mask_dir', "-M", type=str,
                        help='masks of training objects [{}]'.format(params['mask_dir']), default=params['mask_dir'])
    parser.add_argument("--num_workers", '-w', type=int, action='store', default=params['num_workers'],
                        help="worker threads, default {}".format(params['num_workers']))
    parser.add_argument("--num_classes", '-c', type=int, action='store', default=params['num_classes'],
                        help="classes [{}]".format(params['num_classes']))
    args = parser.parse_args()

    params['image_dir'] = args.image_dir
    params['mask_dir'] = args.mask_dir
    params['output_dir'] = args.output_dir
    params['num_workers'] = args.num_workers
    params['num_classes'] = args.num_classes

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    transform_train = Compose([Resize([h, w]),
                               ToTensor(),
                               Normalize(mean=mean, std=std)])

    transform_val = Compose([Resize([h, w]),
                             ToTensor(),
                             Normalize(mean=mean, std=std)])

    mask_transform = Compose([Resize([h, w]),
                              ToTensor()])

    train_ds_full = SegmentationDataset(path2img=params['image_dir'], path2masks=params['mask_dir'],
                                        transform=transform_train, mask_transform=mask_transform)

    val_ds_full = SegmentationDataset(path2img=params['image_dir'], path2masks=params['mask_dir'],
                                      transform=transform_val, mask_transform=mask_transform)

    sss = ShuffleSplit(n_splits=1, test_size=params['test_size'], random_state=0)

    indices = range(len(train_ds_full))

    for train_index, val_index in sss.split(indices):
        print("train = {}".format(len(train_index)))
        print("-" * 10)
        print("valid = {}".format(len(val_index)))

    train_ds = Subset(train_ds_full, train_index)
    print("Train data subset = {}".format(len(train_ds)))

    val_ds = Subset(val_ds_full, val_index)
    print("Valid data subset = {}".format(len(val_ds)))

    train_dl = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, collate_fn=collate_fn,
                          num_workers=params['num_workers'])
    val_dl = DataLoader(val_ds, batch_size=params['batch_size'], shuffle=False, collate_fn=collate_fn,
                        num_workers=params['num_workers'])
    model = deeplabv3_resnet50(pretrained=False, num_classes=params['num_classes'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # ## Define Loss Function
    criterion = nn.CrossEntropyLoss(reduction="sum")
    # ## Optimizer
    opt = optim.Adam(model.parameters(), lr=params['learning_rate'])
    lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=params['patience'], verbose=1)
    current_lr = get_lr(opt)
    print('current lr={}'.format(current_lr))

    path2models = os.path.join(params['output_dir'])
    if not os.path.exists(path2models):
        os.makedirs(path2models)

    params_train = {
        "num_epochs": params['epochs'],
        "optimizer": opt,
        "loss_func": criterion,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "output_dir": params['output_dir'],
        "checkpoint_epochs": params['checkpoint_epochs'],
        "lr_scheduler": lr_scheduler,
        "path2weights": os.path.join(path2models, "weights.pt")
    }

    model, loss_hist, _ = train_val(model, params_train, device)

    num_epochs = params_train["num_epochs"]

    plot_loss(loss_hist=loss_hist, num_epochs=num_epochs, output_dir=params['output_dir'])


if __name__ == '__main__':
    main()
