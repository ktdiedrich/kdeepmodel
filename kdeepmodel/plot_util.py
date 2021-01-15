import matplotlib.pylab as plt
from torch import stack, Tensor
import os
from torchvision import utils
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import itertools


def get_file_paths(input_dir, num_examples, ext):
    file_paths = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(ext)]
    return file_paths


def make_file_image_grid(file_paths, load=np.load):
    """
    Make a grid of images read from file
    :param file_paths:
    :return: Tensor displaying images C, H W order
    """
    img_list = [torch.Tensor(load(file_path)) for file_path in file_paths]
    images = torch.stack(img_list).unsqueeze(1)
    nrow = int(math.ceil(math.sqrt(len(file_paths))))
    image_grid = utils.make_grid(images, nrow=nrow, padding=2, normalize=True)
    return image_grid


def plot_batch(batch, channels=[0], filepath=None):
    if isinstance(batch, list):
        batch = stack(batch)

    if isinstance(batch, Tensor):
        batch = batch.cpu().detach()
    batch_n = batch.shape[0]
    p = 1
    channel_n = len(channels)
    if len(batch.shape) == 4:
        for c in channels:
            for n in range(batch_n):
                plt.subplot(channel_n, batch_n, p)
                plt.imshow(batch[n, c, :, :])
                p += 1
    elif len(batch.shape) == 3:
        for n in range(batch_n):
            plt.subplot(1, batch_n, p, sharey=True)
            plt.imshow(batch[n, :, :])
            p += 1

    if filepath is not None:
        # plt.switch_backend("Agg")
        plt.savefig(filepath)
        plt.close()
    else:
        # plt.switch_backend("qt5agg")
        plt.show()


def plot_loss(loss_hist, num_epochs, output_dir=None, output_name="loss_hist.png"):
    fig = plt.figure()
    plt.title("Train-Val Loss")
    plt.plot(range(1, num_epochs + 1), loss_hist["train"], label="train")
    plt.plot(range(1, num_epochs + 1), loss_hist["val"], label="val")
    plt.ylabel("Loss")
    plt.xlabel("Training Epochs")
    plt.legend()
    if output_dir is not None and output_name is not None:
        # plt.switch_backend("Agg")
        plt.savefig(os.path.join(output_dir, output_name))
        plt.close()
    else:
        # plt.switch_backend("qt5agg")
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, output_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    # Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if output_path is not None:
        plt.savefig(output_path)