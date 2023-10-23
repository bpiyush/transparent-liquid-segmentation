"""API access to segment liquid in a glass in a given image."""
import argparse
import os
import sys
import time
import copy
import argparse
import PIL

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import imageio

import utils
import unet
from test_unet_transparent_pouring import *


import warnings
warnings.filterwarnings("ignore")


def concat_images(images):
    im1 = images[0]
    dst = PIL.Image.new('RGB', (sum([im.width for im in images]), im1.height))
    for i, im in enumerate(images):
        dst.paste(im, (i * im.width, 0))
    return dst


def infer_mask(model, image, threshold=0.5):
    # Get device from the model parameters
    device = next(model.parameters()).device
    
    image = image.resize((150, 300))
    x = np.asarray(image)
    x = np.rollaxis(x, 2, 0)
    x = np.expand_dims(x, axis=0)
    x = torch.tensor(x).to(device)
    # x = torch.tensor(x)
    maskPred = model(x.float()).cpu().detach().numpy()
    maskPred = np.squeeze(maskPred)
    maskThresholded = np.zeros_like(maskPred)
    maskThresholded[maskPred > threshold] = 255
    maskThresholded = maskThresholded.astype(np.uint8)
    
    # Convert soft mask to PIL Image
    maskPred = (maskPred * 255).astype(np.uint8)
    soft_mask = PIL.Image.fromarray(maskPred).convert("RGB")
    
    # Convert hard mask to PIL Image
    hard_mask = PIL.Image.fromarray(maskThresholded).convert("RGB")
    
    return image, soft_mask, hard_mask


def load_model(device, ckpt_path=None):
    # Initialize model
    print("::: Initializing model :::")
    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()

    # Load checkpoint
    print("::: Loading checkpoint :::")
    if ckpt_path is None:
        curr_file = os.path.abspath(__file__)
        curr_dirc = os.path.dirname(curr_file)
        repo_dirc = os.path.dirname(curr_dirc)
        ckpt_path = os.path.join(
            repo_dirc, "data/saved_models/",
            "transparent_liquid_segmentation_unet_150x300_epoch_0_20231023_15_02_05"
        )
        assert os.path.exists(ckpt_path), "Checkpoint does not exist at {}".format(ckpt_path)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    print("::: Checkpoint loaded :::")
    return model


if __name__ == "__main__":

    # Initialize device
    device = torch.device("cuda")

    # Initialize model
    print("::: Initializing model :::")
    model = unet.UNet(n_channels=3, n_classes=1)
    model = model.float()

    # Load checkpoint
    print("::: Loading checkpoint :::")
    ckpt_path = "./data/saved_models/"\
        "transparent_liquid_segmentation_unet_150x300_epoch_0_20231023_15_02_05"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    print("::: Checkpoint loaded :::")

    # Forward pass on image
    curr_file = os.path.abspath(__file__)
    curr_dirc = os.path.dirname(curr_file)
    repo_dirc = os.path.dirname(curr_dirc)
    image_path = os.path.join(
        repo_dirc,
        "data/datasets/pouring_dataset/fakeB/rgb_2108.png"
    )
    mask_path = os.path.join(
        repo_dirc,
        "data/datasets/pouring_dataset/trainA/rgb_2108.jpg"
    )
    image = PIL.Image.open(image_path)
    image, maskPred, maskThresholded = infer_mask(model, image)

    mask = PIL.Image.open(mask_path)
    mask = mask.resize((150, 300))
 
    canvas = concat_images([mask, image, maskPred, maskThresholded])
    canvas.save("test.png")
