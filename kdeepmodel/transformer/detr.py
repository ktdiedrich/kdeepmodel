#!/usr/bin/env python

"""End-to-End Object Detection with Transformers
https://arxiv.org/abs/2005.12872
https://github.com/facebookresearch/detr
https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb#scrollTo=h91rsIPl7tVl
https://github.com/facebookresearch/detr/blob/master/main.py
"""

import torch
from torch import nn
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import torchvision.transforms as T
import numpy as np


class DETR(nn.Module):

    def __init__(self, num_classes, hidden_dim, nheads, num_encoder_layers, num_decoder_layers):
        super().__init__()
        self.backbone = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
        self.conv = nn.Conv2d(2048, hidden_dim, 1)
        self.transformer = nn.Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers)
        self.linear_class = nn.Linear(hidden_dim, num_classes+1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        x = self.backbone(inputs)
        h = self.conv(x)
        H, W = h.shape[-2:]
        pos = torch.cat([
            self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
            self.row_embed[:H].unsqueeze(1).repeat(1, W, 1)
        ], dim=-1).flatten(0, 1).unsqueeze(1)
        h = self.transformer(pos + h.flatten(2).permute(2, 0, 1), self.query_pos.unsqueeze(1))
        return self.linear_class(h), self.linear_bbox(h).sigmoid()


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    params = {}
    params['image'] = '/home/ktdiedrich/data/lung-xray/ChinaSet_AllFiles/CXR_png/CHNCXR_0497_1.png'
    img = imread(params['image'])
    rimg = resize(img, (1200, 800))
    img_rgb = np.array(([rimg] * 3))
    transform = T.Compose([T.ToTensor()])
    detr = DETR(num_classes=91, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
    detr.eval()
    tens_img = transform(img_rgb).permute((1, 0, 2)).unsqueeze(0).float()
    inputs = torch.randn(1, 3, 800, 1200)
    logits, boxes = detr(inputs)

    detr1 = DETR(num_classes=1, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6)
    img_logits, img_boxes = detr1(tens_img)

    print("fin")
