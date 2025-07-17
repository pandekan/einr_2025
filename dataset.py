import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
from torchvision.transforms import ToTensor


class PETSliceDataset(Dataset):
    def __init__(self, tracer1_slices, tracer2_slices):
        assert len(tracer1_slices) == len(tracer2_slices), "Mismatch in number of slices"
        self.tracer1_slices = tracer1_slices
        self.tracer2_slices = tracer2_slices

    def __len__(self):
        return len(self.tracer1_slices)

    def __getitem__(self, idx):
        tracer1 = self.tracer1_slices[idx].astype(np.float32)
        tracer2 = self.tracer2_slices[idx].astype(np.float32)

    # Replace NaNs/Infs with zeros before normalization
        tracer1 = np.nan_to_num(tracer1, nan=0.0, posinf=0.0, neginf=0.0)
        tracer2 = np.nan_to_num(tracer2, nan=0.0, posinf=0.0, neginf=0.0)

        t1_min = tracer1.min()
        t1_max = tracer1.max()
        t2_min = tracer2.min()
        t2_max = tracer2.max()

        print(f"Original tracer1 slice min: {t1_min}, max: {t1_max}")
        print(f"Original tracer2 slice min: {t2_min}, max: {t2_max}")

    # Only normalize if max > min; otherwise leave zeros
        if t1_max > t1_min:
            tracer1 = (tracer1 - t1_min) / (t1_max - t1_min)
        else:
            tracer1 = np.zeros_like(tracer1, dtype=np.float32)

        if t2_max > t2_min:
            tracer2 = (tracer2 - t2_min) / (t2_max - t2_min)
        else:
            tracer2 = np.zeros_like(tracer2, dtype=np.float32)

        print(f"🔬 tracer1 slice min: {tracer1.min()}, max: {tracer1.max()}")
        print(f"🔬 tracer2 slice min: {tracer2.min()}, max: {tracer2.max()}")

        tracer1 = torch.tensor(tracer1).unsqueeze(0)
        tracer2 = torch.tensor(tracer2).unsqueeze(0)

        return tracer1, tracer2


class PETDataset(Dataset):
    def __init__(self, tracer1_paths, tracer2_paths, transform=None):
        self.tracer1_paths = sorted(tracer1_paths)
        self.tracer2_paths = sorted(tracer2_paths)
        self.transform = transform or ToTensor()

    def __len__(self):
        return len(self.tracer1_paths)

    def __getitem__(self, idx):
        tracer1 = np.load(self.tracer1_paths[idx]).astype(np.float32)  # or use nibabel for NIfTI
        tracer2 = np.load(self.tracer2_paths[idx]).astype(np.float32)
        return self.transform(tracer1), self.transform(tracer2)
