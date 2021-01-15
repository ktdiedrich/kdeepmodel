#!/usr/bin/env python
# coding: utf-8

""" Transfer style
Based on https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter08/Chapter8.ipynb
Karl T. Diedrich, PhD <ktdiedrich@gmail.com>

"""


import torch

from PIL import Image
import matplotlib.pylab as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
from torchvision.transforms.functional import to_pil_image
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import optim
import os
import numpy as np

mean_rgb = (0.485, 0.456, 0.406)
std_rgb = (0.229, 0.224, 0.225)


def imgtensor2pil(img_tensor):
    img_tensor_c = img_tensor.clone().detach()
    img_tensor_c*=torch.tensor(std_rgb).view(3, 1,1)
    img_tensor_c+=torch.tensor(mean_rgb).view(3, 1,1)
    img_tensor_c = img_tensor_c.clamp(0,1)
    img_pil=to_pil_image(img_tensor_c)
    return img_pil


def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features


def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n*c, h * w)
    gram = torch.mm(x, x.t())
    return gram


def get_content_loss(pred_features, target_features, layer):
    target= target_features[layer]
    pred = pred_features[layer]
    loss = F.mse_loss(pred, target)
    return loss


def get_style_loss(pred_features, target_features, style_layers_dict):
    loss = 0
    for layer in style_layers_dict:
        pred_fea = pred_features[layer]
        pred_gram = gram_matrix(pred_fea)
        n, c, h, w = pred_fea.shape
        target_gram = gram_matrix(target_features[layer])
        layer_loss = style_layers_dict[layer] * F.mse_loss(pred_gram, target_gram)
        loss += layer_loss/ (n* c * h * w)
    return loss


def total_variation(img):
    h, w = img.shape[2:]
    a = img[:, :, :h-1, :w-1] - img[:, :, 1:, :w-1]
    b = img[:, :, :h-1, :w-1] - img[:, :, :h-1, 1:]
    a = a*a
    b = b*b
    variation = torch.add(a, b)
    variation = torch.pow(variation, 1.5)
    variation = torch.sum(variation)
    return variation


def transer_style(path2content, path2style, width=1000, num_epochs=300,
    content_weight=1e1, style_weight=1e6, variation_weight=1e0):
    content_name = os.path.splitext(path2content)[0]
    style_name = os.path.splitext(path2style)[0]
    content_img = Image.open(path2content)
    w, h = content_img.size
    h_ratio = h / w
    height = int(width*h_ratio)
    style_img = Image.open(path2style)

    transformer = transforms.Compose([
                        transforms.Resize((height,width)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean_rgb, std_rgb)])
    content_tensor = transformer(content_img)

    style_tensor = transformer(style_img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_vgg = models.vgg19(pretrained=True).features.to(device).eval()
    for param in model_vgg.parameters():
        param.requires_grad_(False)
    # print(model_vgg)
    feature_layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',
                      '28': 'conv5_1'}

    con_tensor = content_tensor.unsqueeze(0).to(device)
    sty_tensor = style_tensor.unsqueeze(0).to(device)

    content_features = get_features(con_tensor, model_vgg, feature_layers)
    style_features = get_features(sty_tensor, model_vgg, feature_layers)

    #for key in content_features.keys():
    #    print(content_features[key].shape)
    input_tensor = con_tensor.clone().requires_grad_(True)
    optimizer = optim.Adam([input_tensor], lr=0.01)

    content_layer = "conv5_1"
    style_layers_dict = { 'conv1_1': 0.8,
                          'conv2_1': 0.6,
                          'conv3_1': 0.4,
                          'conv4_1': 0.2,
                          'conv5_1': 0.1
                          }

    for epoch in range(num_epochs+1):
        optimizer.zero_grad()
        input_features = get_features(input_tensor, model_vgg, feature_layers)
        content_loss = get_content_loss(input_features, content_features, content_layer)
        style_loss = get_style_loss(input_features, style_features, style_layers_dict)
        variation_loss = total_variation(input_tensor)
        neural_loss = content_weight * content_loss + style_weight * style_loss + variation_weight * variation_loss
        neural_loss.backward(retain_graph=True)
        optimizer.step()

        if epoch % 100 == 0:
            print('epoch {}, content loss: {:.2}, style loss {:.2}, variation loss {:.2}'.format(
              epoch,content_loss, style_loss, variation_loss))

    pil_img = imgtensor2pil(input_tensor[0].cpu())
    # plt.imshow(pil_img)
    style_dir = "styled"
    os.makedirs(style_dir, exist_ok=True)
    content_styled_name = os.path.join(style_dir, "{}_{}.png".format(content_name, style_name))
    pil_img.save(content_styled_name)
    print("Saved {}".format(content_styled_name))


if __name__ == '__main__':
    import re

    os.listdir(".")
    content_imgs = [f for f in os.listdir('.') if re.search(".*IMG.+", f)]
    STYLES = ["oil_color.jpg", "van_gough.jpg", "Korea_tiger.jpg", "Korea_buddhist.jpg", "picasso_2.jpg",
              "starry_night.jpg", "dali_2.jpg"]
    for cont_img in content_imgs:
        for style in STYLES:
            transer_style(cont_img, style, content_weight=1e2, style_weight=1e2, variation_weight=1e0)
