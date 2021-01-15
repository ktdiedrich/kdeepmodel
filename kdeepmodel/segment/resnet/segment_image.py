#!/usr/bin/env python

import torch
import importlib
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from kdeepmodel.plot_util import plot_batch
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt


def tensor_image(image_path, width, height, device):
    img = Image.open(image_path)
    img = img.resize((width, height))
    img_t = to_tensor(img).unsqueeze(0).to(device)
    if img_t.shape[1] == 1:
        img_t = torch.cat(3 * [img_t], dim=1)
    return img_t


def load_model(model, weights_path, device):
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    model = model.to(device)
    return model


def predict_segmentation(params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    img_t = tensor_image(params['image_path'], params['width'], params['height'], device)
    model = load_model(params['model'], params['weights_path'], device)
    with torch.no_grad():
        prediction = model(img_t)['out']
    return prediction, img_t, model


def predict_segmentation_by_script(params):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_t = tensor_image(params['image_path'], params['width'], params['height'], device)
    model = torch.jit.load(params['weights_path'])
    prediction = model(img_t)['out']
    return prediction, img_t, model


def trace_script(model, input_t):
    scripted_model = torch.jit.script(model, input_t)
    return scripted_model


def mask_segmentation(seg_prediction, input_tensor, threshold):
    """
    Threshold and apply mask to input image
    :param seg_prediction: segmentation prediction probabilities
    :param input_tensor: input image tensor
    :param threshold: Keep input values above threshold values, otherwise 0
    :return: Tensor of segmented image
    """
    seg_channel = seg_prediction[0, 1, :, :].cpu().detach()
    mask = (seg_channel > threshold).float() * 1
    img = input_tensor[0, 1, :, :].cpu().detach()
    img[mask == 0.0] = 0.0
    return img


if __name__ == '__main__':
    import argparse
    import json
    import os

    params = {}
    params['image_path'] = None
    params['weights_path'] = None
    params['output_dir'] = '.'
    # params['model'] = "torchvision.models.segmentation.deeplabv3_resnet50"
    params['height'] = 520
    params['width'] = 520
    params['num_classes'] = 2
    params['threshold'] = -1.0
    parser = argparse.ArgumentParser(description='Segment objects')
    parser.add_argument('image_path', type=str, help='image path')
    parser.add_argument('weights_path', type=str, help='weights path or Torchscript')
    parser.add_argument("--output_dir", "-o", type=str, help="output directory [{}]".format(params['output_dir']),
                        default=params['output_dir'])
    # parser.add_argument('--model', "-m", type=str,
    #                     help='model, {}'.format(params['model']), default=params['model'])
    parser.add_argument('--height', "-H", type=str,
                        help='height, {}'.format(params['height']), default=params['height'])
    parser.add_argument('--width', "-W", type=str,
                        help='width, {}'.format(params['width']), default=params['width'])
    parser.add_argument('--num_classes', "-n", type=str,
                        help='number of classes, {}'.format(params['num_classes']), default=params['num_classes'])
    TORCHSCRIPT_SAVE = None
    parser.add_argument("--torchscript_save", "-s", help="Save Torchscript to path, {}".format(TORCHSCRIPT_SAVE),
                        default=TORCHSCRIPT_SAVE)
    TORCH_SCRIPT_PREDICT = False
    parser.add_argument("--torchscript_predict", "-T", action="store_true",
                        help="Predict with Torchscript in weights_path, {}".format(TORCH_SCRIPT_PREDICT),
                        default=TORCH_SCRIPT_PREDICT)
    parser.add_argument('--threshold', "-t", type=str,
                        help='threshold, {}'.format(params['threshold']), default=params['threshold'])
    args = parser.parse_args()

    params['image_path'] = args.image_path
    params['weights_path'] = args.weights_path
    # params['model'] = args.model
    params['output_dir'] = args.output_dir
    params['height'] = args.height
    params['width'] = args.width

    if not os.path.exists(params['output_dir']):
        os.makedirs(params['output_dir'])

    with open(os.path.join(params['output_dir'], 'params.json'), 'w') as fp:
        json.dump(params, fp)

    if not args.torchscript_predict:
        params['model'] = deeplabv3_resnet50(pretrained=False, num_classes=params['num_classes'])
        seg_prediction, input_t, weighted_model = predict_segmentation(params)
        if args.torchscript_save is not None:
            traced_script = trace_script(weighted_model, input_t)
            traced_prediction = traced_script(input_t)['out']
            # plot_batch(traced_prediction, list(range(params['num_classes'])))
            traced_script.save(args.torchscript_save)
    else:
        seg_prediction, input_t, weighted_model = predict_segmentation_by_script(params)

    plot_batch(input_t)
    plt.figure()
    plot_batch(seg_prediction, [1])
    seg_image = mask_segmentation(seg_prediction, input_t, params['threshold'])
    plt.figure()
    plt.imshow(seg_image)
    plt.show()
    # plot_batch(segmentation, list(range(params['num_classes'])))


