import os
import glob
import torch
import torch.nn as nn
import numpy as np
import unet
import pytorch_lightning as pl

class PET2PETModule(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = unet.UNet2D(
            in_channels=1,
            out_channels=1,
        )
        self.loss_fn = nn.MSELoss()
        #self.metric = SSIMMetric(spatial_dims=2)
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.metric(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        #self.log("val_ssim", self.metric.aggregate().item(), prog_bar=True)
        #self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
