"""Trainer script."""
import os
import numpy as np
import pandas as pd
import torch
import torchvision
import torchaudio
import einops
import pytorch_lightning as pl
import json
import time

import warnings
warnings.filterwarnings("ignore")

from transformers import SamModel
from finetuning_sam.datasets.liquid_segmentation import load_dataset
from finetuning_sam.lightning_models.sam import SAMLightningModule
import shared_utils as su

pl_version = pl.__version__.split(".")


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--gpus", type=int, default=[0, 1, 2, 3], nargs="+")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--freeze_video_encoder", action="store_true")
    parser.add_argument("--freeze_audio_encoder", action="store_true")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--load_dataset_args", default="{}", type=json.loads)
    args = parser.parse_args()
    args.load_dataset_args = {k: eval(v) for k, v in args.load_dataset_args.items()}

    # Print args beautifully
    print("\n> Training with the following arguments:")
    for k, v in vars(args).items():
        print(f"> {k}: {v}")
    # sleep for 10s to allow user to kill the process
    time.sleep(10)


    # Load datasets
    su.log.print_update("[ Loading datasets ] ", pos="left", color="green")
    ds_train = load_dataset(split="train", **args.load_dataset_args)
    dl_train = torch.utils.data.DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ds_valid = load_dataset(split="val", **args.load_dataset_args)
    dl_valid = torch.utils.data.DataLoader(
        ds_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    
    # Load lightning module
    su.log.print_update("[ Loading SAM model ] ", pos="left", color="green")
    # Load SAM model
    sam_model = SamModel.from_pretrained("facebook/sam-vit-base")
    # make sure we only compute gradients for mask decoder
    for name, param in sam_model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)
    # Load SAM Lightning module
    sam_lightning_module = SAMLightningModule(sam_model, myhparams=dict(lr=args.lr))


    # Initialize a trainer
    su.log.print_update("[ Starting training ] ", pos="left", color="green")
    logger = None
    if not args.no_wandb:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        suffix = "-" + args.suffix if args.suffix else ""
        run_name = f"{timestamp}_sam_liquid_segmentation-lr={args.lr}" + suffix
        # Use a WandB logger
        logger = pl.loggers.WandbLogger(
            project="audio-visual-test-of-time",
            entity="bpiyush",
            name=run_name,
        )
    ckpt_saver = pl.callbacks.ModelCheckpoint(
        every_n_epochs=args.save_every,
        save_top_k=-1,
        save_last=True,
    )
    if pl_version[0] == "1":
        from pytorch_lightning.plugins import DDPPlugin
        trainer = pl.Trainer(
            gpus=args.gpus,
            max_epochs=args.max_epochs,
            logger=logger,
            log_every_n_steps=1,
            callbacks=[ckpt_saver],
            plugins=DDPPlugin(find_unused_parameters=False),
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            logger=logger,
            log_every_n_steps=1,
            callbacks=[ckpt_saver],
            devices=4,
            accelerator="gpu",
            strategy="ddp",
        )
    trainer.fit(
        model=sam_lightning_module,
        train_dataloaders=dl_train,
        val_dataloaders=dl_valid,
    )

