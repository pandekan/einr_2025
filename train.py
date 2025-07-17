import os
import random
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import PETSliceDataset, extract_slices
from model import PET2PETModule
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from glob import glob
import logging

logger = logging.getLogger(__name__)

# prepare validation data
base_dir = "/home/jagust/head_ml/train"
patient_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
random.shuffle(patient_dirs)

mk_scans = []
ftp_scans = []

for patient_dir in patient_dirs:
    mk_path = None
    ftp_path = None
    for root, dirs, files in os.walk(patient_dir):
        if "wsuvr_cereg.nii" in files:
            if "MK6240" in root:
                mk_path = os.path.join(root, "wsuvr_cereg.nii")
            elif "FTP" in root:
                ftp_path = os.path.join(root, "wsuvr_cereg.nii")

    if mk_path and ftp_path:
        try:
            mk_img = nib.load(mk_path)
            ftp_img = nib.load(ftp_path)

            mk_scans.append(mk_img)
            ftp_scans.append(ftp_img)

            logger.info(f"✅ Loaded MK & FTP for patient: {os.path.basename(patient_dir)}")
        except Exception as e:
            logger.error(f"❌ Error loading patient {os.path.basename(patient_dir)}: {e}")

    if len(mk_scans) >= 25:
        break

logger.info(f"\n🎯 Loaded {len(mk_scans)} matched MK & FTP scans.")

# prepare training and validation data
val_base_dir = "/home/jagust/head_ml/validate"
val_patient_dirs = [os.path.join(val_base_dir, d) for d in os.listdir(val_base_dir) if os.path.isdir(os.path.join(val_base_dir, d))]
random.shuffle(val_patient_dirs)

mk_val_scans = []
ftp_val_scans = []

for patient_dir in val_patient_dirs:
    mk_path = None
    ftp_path = None
    for root, dirs, files in os.walk(patient_dir):
        if "wsuvr_cereg.nii" in files:
            if "MK6240" in root:
                mk_path = os.path.join(root, "wsuvr_cereg.nii")
            elif "FTP" in root:
                ftp_path = os.path.join(root, "wsuvr_cereg.nii")

    if mk_path and ftp_path:
        try:
            mk_img = nib.load(mk_path)
            ftp_img = nib.load(ftp_path)

            mk_val_scans.append(mk_img)
            ftp_val_scans.append(ftp_img)

            logger.info(f"✅ Loaded MK & FTP for val patient: {os.path.basename(patient_dir)}")
        except Exception as e:
            logger.error(f"❌ Failed to load val patient {os.path.basename(patient_dir)}: {e}")

    if len(mk_val_scans) >= 15:
        break

logger.info(f"\n🎯 Loaded {len(mk_val_scans)} matched validation MK & FTP scans.\n")

mask_img = nib.load("/home/jagust/head_ml/MNI_wholeCortex.nii")

mk_slices_list = extract_slices(mk_scans, mask_img, "mk_scans_masked_2D_slices.nii")
ftp_slices_list = extract_slices(ftp_scans, mask_img, "ftp_scans_masked_2D_slices.nii")
mk_val_slices_list = extract_slices(mk_val_scans, mask_img, "mk_val_scans_masked_2D_slices.nii")
ftp_val_slices_list = extract_slices(ftp_val_scans, mask_img, "ftp_val_scans_masked_2D_slices.nii")

# Dataloaders
train_dataset = PETSliceDataset(mk_slices_list, ftp_slices_list)
val_dataset = PETSliceDataset(mk_val_slices_list, ftp_val_slices_list)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

checkpoint_callback = ModelCheckpoint(
    dirpath='checkpoints/',  # Directory to save checkpoints
    filename='epoch={epoch:02d}-val_loss={val_loss:.2f}',  # Naming convention
    monitor='val_loss',  # Metric to monitor for saving best models
    save_top_k=1,  # Save only the best model based on monitor
    mode='min'  # 'min' for metrics like loss, 'max' for accuracy
)

model = PET2PETModule(lr=1e-3)

trainer = pl.Trainer(
    max_epochs=10,
    accelerator="cpu",  # or "gpu" if available
    devices=1,
    log_every_n_steps=1,
    precision=32,
    gradient_clip_val=1.0,
    callbacks = [checkpoint_callback] 
    
)

logger.info("🟢 Starting training...")
trainer.fit(model, train_loader, val_loader)
logger.info("✅ Training finished.")
