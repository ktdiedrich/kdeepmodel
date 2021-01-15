#!/usr/bin/env python3

"""
Augment input images and train classifier model to classify images
* Default hyper-parameters set in parameter dictionary
* Override default hyper-parameters with command line or web page arguments
    see: Python flask https://palletsprojects.com/p/flask/
    see: Javascript React https://reactjs.org/
* Dictionary of current training hyper-parameters saved to JSON in output directory with model
* Training output and or saves intermediate images and graphs for debugging and optimization,
    see: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    see: https://seaborn.pydata.org/
* Optimize hyper-parameters with genetic algorithms
    see: https://github.com/handcraftsman/GeneticAlgorithmsWithPython/
* Inference with another script with command line or web-page arguments
Karl Diedrich, PhD <ktdiedrich@gmail.com>

Based on:
* https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

Sample data https://www.kaggle.com/kmader/colorectal-histology-mnist
/home/ktdiedrich/data/MedNIST !wget -q https://www.dropbox.com/s/5wwskxctvcxiuea/MedNIST.tar.gz


"""

import torch
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

param = dict()

param = dict()
param['test_size'] = 0.2
param['show'] = False
param['print'] = False
param['epochs'] = 160
param['batch_size'] = 16
param['output_dir'] = '.'
param['model_output_name'] = "trained_model.pth"
param['figure_name'] = 'augmented_images.png'
param['validation_split'] = 0.2
param['figure_size'] = (9, 9)
param['learning_rate'] = 0.001
param['momentum'] = 0.5
param['dropout'] = 0.3
param['num_workers'] = 4
param['sample_size'] = 3
param['results_filename'] = 'results.json'
param['log_subdir'] = 'log'
param['log_filename'] = 'tensorboard.log'


def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def output_size(W, F, P, S):
    """
    https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-in-pytorch/
    :param W: width in
    :param F: filter diameter
    :param P: padding
    :param S: stride
    :return:width out
    """
    return (W-F+2*P)/S + 1


def net_output_dim(x):
    s1 = output_size(x.shape[1], 5, 0, 1)
    s2 = output_size(s1, 2, 0, 2)
    s3 = output_size(s2, 5, 0, 1)
    s4 = output_size(s3, 2, 0, 2)
    return int(np.floor(s4))


class Net(nn.Module):
    def __init__(self, output_n, net_dim):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(int(16 * net_dim * net_dim), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_n)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def show_batch(batch, number,  classes, title="Batch"):
    batch_size = len(batch[0])
    batch_side = int(np.ceil(np.sqrt(batch_size)))
    fig, axes = plt.subplots(batch_side, batch_side, sharex=True, sharey=True, figsize=param['figure_size'])
    fig.suptitle("{} {}".format(title, number))

    index = 0
    for x in range(batch_side):
        for y in range(batch_side):
            if index < batch_size:
                sample_item = batch[0][index]
                # Tensor Channel, Height, Width
                # Convert to Height, Width, Channel with permute
                sample_item = sample_item.permute(1, 2, 0)
                sample_item = sample_item / sample_item.max()  # normalize [0..1]
                axes[x][y].imshow(sample_item)
                label_index = batch[1][index]
                axes[x][y].set_title(classes[label_index])
                index += 1
    plt.show()


class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable, optional): A function/transform to be applied on the sample
        target_transform (callable, optional): A function/transform to be applied on the target

    """
    def __init__(self, dataset, transform=None, target_transform=None):
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        # yes, you don't need these 2 lines below :(
        if transform is None and target_transform is None:
            print("pass a transform")

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.dataset)


def train_model(net, train_loader, epochs, classes, device, sample_size, criterion, optimizer, show=False):
    train_results = dict()

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_number, batch_samples in enumerate(train_loader):
            if show and batch_number < sample_size:
                show_batch(batch=batch_samples, number=batch_number, classes=classes,
                           title="Training epoch {} batch".format(epoch))
            inputs, labels = batch_samples[0].to(device), batch_samples[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_results['loss'] = loss.item()
            if batch_number % 50 == 49:  # print every N mini-batches
                print('[{}, {}] loss: {:.3}'.format(epoch, batch_number, running_loss / 50))
                train_results['epoch'] = epoch
                train_results['batch'] = batch_number
                train_results['running_loss'] = running_loss
                running_loss = 0.0
    return net, train_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Augment images for model training.')
    parser.add_argument('input_dir', type=str, help='input_directory')
    parser.add_argument("--output_dir", "-o", type=str, required=False, default=param['output_dir'],
                        help="output directory, default {}".format(param['output_dir']))
    parser.add_argument("--test_size", "-t", type=float, action="store", default=param['test_size'], required=False,
                        help="test proportion size, default {}".format(param['test_size']))
    parser.add_argument("--epochs", "-e", type=int, action="store", help="epochs, default {}".format(param['epochs']),
                        default=param['epochs'], required=False)
    parser.add_argument("--num_workers", "-n", type=int, action="store",
                        help="number of worker threads, default {}, use 0 for debug".format(param['num_workers']),
                        default=param['num_workers'], required=False)
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
    param['num_workers'] = args.num_workers

    train_transform = transforms.Compose([
        # transforms.RandomRotation(360),
        # transforms.RandomResizedCrop(150),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # cifar = datasets.CIFAR10(root='./data', train=True,
    #                                    download=True, transform=test_transform)

    dataset = datasets.ImageFolder(root=param['input_dir'])

    dataset_len = len(dataset)
    test_data, train_data = torch.utils.data.random_split(dataset, (
        int(np.floor(dataset_len * param['test_size'])), int(np.ceil(dataset_len * (1 - param['test_size'])))))

    train_data = ApplyTransform(train_data, transform=train_transform)

    test_data = ApplyTransform(test_data, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                                 batch_size=param['batch_size'], shuffle=True,
                                                 num_workers=param['num_workers'])

    test_loader = torch.utils.data.DataLoader(test_data,
                                               batch_size=param['batch_size'], shuffle=True,
                                               num_workers=param['num_workers'])

    if not os.path.exists(param['output_dir']):
        os.makedirs(param['output_dir'])
    with open(os.path.join(param['output_dir'], 'param.json'), 'w') as fp:
        json.dump(param, fp)

    log_dir = os.path.join(param['output_dir'], param['log_subdir'])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # writer = SummaryWriter(os.path.join(log_dir, param['log_filename']))

    net_dim = net_output_dim(train_data[0][0])
    # net = Net(output_n=len(dataset.classes), net_dim=net_dim)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(dataset.classes))

    net = model_ft

    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=param['learning_rate'], momentum=param['momentum'])

    results = dict()

    net, train_results = train_model(net=net, train_loader=train_loader, epochs=param['epochs'], classes=dataset.classes,
                                     device=device, sample_size=param['sample_size'], criterion=criterion,
                                     optimizer=optimizer, show=param['show'])

    torch.save(net.state_dict(), os.path.join(param['output_dir'], param['model_output_name']))

    test_results = dict()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_number, batch_samples in enumerate(test_loader):
            if param['show'] and batch_number < param['sample_size']:
                show_batch(batch=batch_samples, number=batch_number, classes=dataset.classes, title="Testing batch")
            inputs, labels = batch_samples[0].to(device), batch_samples[1].to(device)
            outputs = net(inputs)
            out_ret, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_results['total'] = total
    test_results['correct'] = correct
    test_results['accuracy'] = float(correct)/total
    print("Test total={} correct={} accuracy={:.3}".format(total, correct, test_results['accuracy']))
    classes_n = len(dataset.classes)
    class_correct = list(0. for i in range(classes_n))
    class_total = list(0. for i in range(classes_n))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(classes_n):
        class_results = dict()
        class_results['total'] = class_total[i]
        class_results['correct'] = class_correct[i]
        class_accuracy = class_correct[i] / class_total[i]
        test_results[dataset.classes[i]] = class_results
        print('{}: total={} correct={} accuracy={} '.format(dataset.classes[i], class_total[i], class_correct[i],
                                                            class_accuracy))
    results['train'] = train_results
    results['test'] = test_results

    with open(os.path.join(param['output_dir'], param['results_filename']), 'w') as fp:
        json.dump(results, fp)

    print("fin")
