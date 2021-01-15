#!/usr/bin/env python

"""Variational autoencoder resources
https://debuggercafe.com/category/autoencoders/
https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
https://github.com/pytorch/ignite/blob/master/examples/notebooks/VAE.ipynb
"""


import torchvision
import torch.optim as optim
import argparse
import matplotlib
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from kdeepmodel.vae import model as vmodel
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
import torchvision.datasets as datasets
from pathlib import Path
import os


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def fit(model, dataloader, train_data, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        data, _ = data
        data = data.to(device)
        data = data.view(data.size(0), -1)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    train_loss = running_loss/len(dataloader.dataset)
    return train_loss


def validate(model, dataloader, val_data, device, criterion,batch_size, epoch, output_dir):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()

            # save the last batch input and output of every epoch
            if i == int(len(val_data) / dataloader.batch_size) - 1:
                num_rows = 8
                both = torch.cat((data.view(batch_size, 1, 28, 28)[:8],
                                  reconstruction.view(batch_size, 1, 28, 28)[:8]))
                save_image(both.cpu(), os.path.join(output_dir, "output{}.png".format(epoch)),
                                                    nrow=num_rows)
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss


def main():
    import argparse

    params = {}
    params['epochs'] = 10
    params['output_dir'] = os.path.join(str(Path.home()), "output/autoencode_mnist")
    parser = argparse.ArgumentParser(description='DICOM images')

    parser.add_argument('-e', '--epochs', default=params['epochs'], type=int,
                        help='number of epochs to train the VAE for [{}]'.format(params['epochs']))
    parser.add_argument('-o', '--output_dir', default=params['output_dir'], type=str,
                        help='output directory [{}]'.format(params['output_dir']))
    args = vars(parser.parse_args())
    epochs = args['epochs']
    output_dir = args['output_dir']
    batch_size = 64
    lr = 0.0001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args['output_dir'], exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    data_path = os.path.join(str(Path.home()), "data/mnist_down")
    # train and validation data
    train_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    val_data = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    # training and validation data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False
    )
    model = vmodel.LinearVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss(reduction='sum')

    train_loss = []
    val_loss = []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")
        train_epoch_loss = fit(model, train_loader, train_data, device, optimizer, criterion)
        val_epoch_loss = validate(model, val_loader, val_data, device, criterion,batch_size, epoch, output_dir)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")
        print(f"Val Loss: {val_epoch_loss:.4f}")


if __name__ == '__main__':
    main()

