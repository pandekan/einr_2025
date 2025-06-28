import os
import glob
import torch
import numpy as np
import monai
from monai.transforms import (
    LoadImaged, AddChanneld, ScaleIntensityd, ResizeWithPadOrCropd,
    ToTensord, Compose
)
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import SSIMMetric
import pytorch_lightning as pl

class PET2PETModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = UNet(
            dimensions=2,  # use 3 for volumetric
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )
        self.loss_fn = monai.losses.MSELoss()
        self.metric = SSIMMetric(spatial_dims=2)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["tracer1"], batch["tracer2"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["tracer1"], batch["tracer2"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.metric(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_ssim", self.metric.aggregate().item(), prog_bar=True)
        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
