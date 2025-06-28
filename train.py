import os
from torch.utils.data import DataLoader
from dataset import PairedPETDataset
from model import PET2PETModule
import pytorch_lightning as pl
from glob import glob

# Update these with your actual data paths
tracer1_train = glob("data/train/tracer1/*.npy")
tracer2_train = glob("data/train/tracer2/*.npy")
tracer1_val = glob("data/val/tracer1/*.npy")
tracer2_val = glob("data/val/tracer2/*.npy")

train_dataset = PairedPETDataset(tracer1_train, tracer2_train)
val_dataset = PairedPETDataset(tracer1_val, tracer2_val)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8)

model = PET2PETModule(lr=1e-4)

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",  # or "cpu"
    devices=1,
    log_every_n_steps=10,
    precision=16,  # for faster training
)

trainer.fit(model, train_loader, val_loader)
