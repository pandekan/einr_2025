import torch
from torch.utils.data import Dataset
import numpy as np
import os
from glob import glob
from torchvision.transforms import ToTensor

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
