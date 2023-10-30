"""Liquid segmentation dataset based on generated images from Narsimhan et al, ICRA 2022."""
import os
from os.path import join, exists
import numpy as np
import PIL
import matplotlib.pyplot as plt
from glob import glob
import cv2
from natsort import natsorted
import tqdm

import shared_utils as su

from transformers import SamProcessor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class LiquidSegmentationV1(Dataset):
    def __init__(self, data_dir, split, processor, imsize=(256, 256), preload=True):

        self.data_dir = data_dir
        self.split = split
        self.preload = preload
        self.imsize = imsize
        self.processor = processor
        self.load_dataset(self.data_dir)

    def load_dataset(self, data_dir):
        self.image_dir = join(data_dir, "fakeB")
        self.amask_dir = join(data_dir, "trainA_liquid_masks")
        self.cup_mask_dir = join(data_dir, "trainA_cup_masks")
        
        # Load split indices
        assert self.split in ["train", "val"]
        split_path = join(data_dir, f"metadata/{self.split}V1.txt")
        split_indices = np.loadtxt(split_path, dtype=int)

        data = []
        n_images = len(split_indices)
        for i in tqdm.tqdm(split_indices, desc=f"Loading data for split {self.split}"):
            item = {
                "image_path": join(self.image_dir, f"rgb_{i}.png"),
                "liquid_mask_path": join(self.amask_dir, f"liquid_mask_{i}.npy"),
                "cup_mask_path": join(self.cup_mask_dir, f"cup_mask_{i}.npy"),
            }
            
            if self.preload:
                # Load images and masks right-away, since the dataset is small
                item["image"] = self.load_image(item["image_path"])
                item["liquid_mask"] = self.load_mask(item["liquid_mask_path"], bin_threshold=0.95)
                item["cup_mask"] = self.load_mask(item["cup_mask_path"], bin_threshold=None)
            
            data.append(item)
        self.data = data

    def __len__(self):
        return len(self.data)

    def load_image(self, image_path):
        assert exists(image_path), f"Image does not exist at {image_path}."
        image = PIL.Image.open(image_path)
        # Need to resize for SAM compatibility
        image = image.resize(self.imsize)
        return image
    
    def load_mask(self, mask_path, bin_threshold=None):
        assert exists(mask_path), f"Mask does not exist at {mask_path}."
        mask = np.load(mask_path)
        # Need to resize for SAM compatability
        mask = su.viz.resize_mask(mask, self.imsize)
        if bin_threshold is not None:
            # Binarize mask
            mask = su.viz.binarize_mask(mask, bin_threshold)
        return mask

    def __getitem__(self, idx, visualize_example=False):
        item = self.data[idx]

        image_path = item["image_path"]
        if self.preload:
            image = item["image"]
        else:
            image = self.load_image(image_path)

        liquid_mask_path = item["liquid_mask_path"]
        if self.preload:
            liquid_mask = item["liquid_mask"]
        else:
            liquid_mask = self.load_mask(liquid_mask_path, bin_threshold=0.95)
        
        cup_mask_path = item["cup_mask_path"]
        if self.preload:
            cup_mask = item["cup_mask"]
        else:
            cup_mask = self.load_mask(cup_mask_path, bin_threshold=None)
        cup_bbox = su.viz.mask_to_bounding_box(cup_mask, perturbation=10)
        prompt = cup_bbox

        if visualize_example:
            # Binarize cup mask
            cup_mask = su.viz.binarize_mask(cup_mask, 0.45)

            # Helpers for visualization
            from IPython.display import display
            liquid_mask_pil = su.viz.alpha_mask_to_pil_image(liquid_mask, 0.95)
            cup_mask_pil = su.viz.alpha_mask_to_pil_image(cup_mask, 0.45)
            image_with_liquid_mask = su.viz.add_mask_on_image(
                image, liquid_mask_pil, color="blue",
            )
            image_with_cup_mask = su.viz.add_mask_on_image(
                image, cup_mask_pil, color="yellow",
            )
            image_with_cup_bbox = su.viz.add_bbox_on_image(
                image, cup_bbox, color="yellow",
            )
            canvas = su.viz.concat_images(
                [image, image_with_liquid_mask, image_with_cup_mask, image_with_cup_bbox]
            )
            display(
                canvas
            )
            canvas.save("example.png")

        # prepare image and prompt for the model
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = liquid_mask

        return inputs


def load_dataset(split):
    data_dir = "./data/datasets/pouring_dataset"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    ds = LiquidSegmentationV1(data_dir=data_dir, split=split, processor=processor, preload=False)
    return ds


if __name__ == "__main__":
    import time
    
    # Load dataset
    data_dir = "./data/datasets/pouring_dataset"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    ds = LiquidSegmentationV1(data_dir=data_dir, split="val", processor=processor, preload=False)
    start = time.time()
    item = ds[0]
    end = time.time()
    print("Time taken to load one item (preload=False):", end - start)
    for k, v in item.items():
        print(k, v.shape)
    
    # Load dataset with preloading
    data_dir = "./data/datasets/pouring_dataset"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    ds = LiquidSegmentationV1(data_dir=data_dir, split="val", processor=processor, preload=True)
    start = time.time()
    item = ds[0]
    end = time.time()
    print("Time taken to load one item (preload=True):", end - start)


    # Batched dataloader
    bs = 4
    dl = DataLoader(ds, batch_size=bs, shuffle=True)
    start = time.time()
    batch = next(iter(dl))
    end = time.time()
    print("Time taken to load one sample (batch_time/batch_size):", (end - start) / bs)
