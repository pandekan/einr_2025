import os
import glob
import torch
import torch.nn as nn
import numpy as np
#from unet import UNet2D
from unet_ayush import UNet2D
import pytorch_lightning as pl
import logging
logger = logging.getLogger(__name__)

class PET2PETModule(pl.LightningModule): 
    def __init__(self, lr=1e-4):
        super().__init__()
        self.model = UNet2D(in_channels=1, out_channels=1)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        
        return self.model(x)

    def training_step(self, batch, batch_idx):
        tracer1, tracer2 = batch

        assert torch.isfinite(tracer1).all(), "❌ Inf or NaN in tracer1 input!"
        assert torch.isfinite(tracer2).all(), "❌ Inf or NaN in tracer2 target!"

        preds = self(tracer1)
        assert not torch.isnan(preds).any(), "NaN in model output!"

        loss = self.loss_fn(preds, tracer2)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % 10 == 0:
            logger.info(f"✅ Train step {batch_idx} | Loss: {loss.item():.6f}")

        return loss

    def validation_step(self, batch, batch_idx):
        
        tracer1, tracer2 = batch

        assert torch.isfinite(tracer1).all(), "❌ Inf or NaN in tracer1 input!"
        assert torch.isfinite(tracer2).all(), "❌ Inf or NaN in tracer2 target!"

        preds = self(tracer1)
        assert not torch.isnan(preds).any(), "NaN in model output!"

        loss = self.loss_fn(preds, tracer2)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    
        if batch_idx % 5 == 0:
            logger.info(f"🔍 Val step {batch_idx} | Loss: {loss.item():.6f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
