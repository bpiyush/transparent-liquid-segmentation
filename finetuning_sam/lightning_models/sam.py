"""Defines a PyTorch Lightning module for training a model with SAM."""
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.functional import threshold, normalize
from transformers import SamModel
import monai
from tqdm import tqdm
from statistics import mean

import warnings
warnings.filterwarnings("ignore")


class SAMLightningModule(pl.LightningModule):
    def __init__(self, sam_model, myhparams=dict(lr=1e-4, weight_decay=0)):
        super().__init__()

        self.sam_model = sam_model
        self.myhparams = myhparams
        self.seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        
        t_params = sum(p.numel() for p in self.parameters())
        print(f"Number of total parameters: {t_params / 1e6} M")
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {n_params / 1e6} M")
        
        self.save_hyperparameters(ignore=['sam_model'])

    def step(self, batch):
        # forward pass
        outputs = self.sam_model(
            pixel_values=batch["pixel_values"],
            input_boxes=batch["input_boxes"],
            multimask_output=False,
        )
        
        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float()
        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
        
        # HACK to avoid https://github.com/pytorch/pytorch/issues/43259
        loss += 0. * outputs["iou_scores"].sum()

        return loss
    
    def forward(self, pixel_values, input_boxes):
        outputs = self.sam_model(
            pixel_values=pixel_values,
            input_boxes=input_boxes,
            multimask_output=False,
        )
        return outputs
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        # Note: Hyperparameter tuning could improve performance here
        optimizer = Adam(
            self.parameters(),
            lr=self.myhparams.get("lr", 1e-4),
            weight_decay=self.myhparams.get("weight_decay", 0),
        )
        return optimizer


if __name__ == "__main__":

    # Load SAM model
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    # make sure we only compute gradients for mask decoder
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    # Load SAM Lightning module
    sam_lightning_module = SAMLightningModule(sam_model)
    
    
    # Test with actual data
    from finetuning_sam.datasets.liquid_segmentation import load_dataset
    ds = load_dataset(split="val")
    dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(dl))
    outputs = sam_lightning_module(
        pixel_values=batch["pixel_values"],
        input_boxes=batch["input_boxes"],
    )
    loss = sam_lightning_module.step(batch)
    print("loss:", loss.item())

