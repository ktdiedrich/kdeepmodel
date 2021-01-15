#!/usr/bin/env python

"""Training utility functions
"""


def loop_train(data_loader, model, criterion, optimizer, epochs):
    """
    Training loop for Pytorch
    :param data_loader:
    :param model:
    :param criterion:
    :param optimizer:
    :param epochs:
    :return:
    """
    losses = []
    for epoch in range(epochs):
        for x, y in data_loader:
            yhat = model(x)
            loss = criterion(yhat, y)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses 

