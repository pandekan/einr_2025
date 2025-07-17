import torch
from torch.utils.data import Dataset
import numpy as np
import os
import nibabel as nib
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


def extract_slices(img_list, mask_img, output_path):
    if len(img_list) == 0:
        raise ValueError(f"No images found in the provided list for {output_path}.")

    mask_data = mask_img.get_fdata().astype(bool)

    example_img = img_list[0]
    if hasattr(example_img, 'get_fdata'):
        example_data = example_img.get_fdata()
    else:
        example_data = example_img

    height, width = example_data.shape[:2]
    num_subjects = len(img_list)

    newy = np.zeros((height, width, num_subjects), dtype=np.float32)
    slices_list = []

    for i, img in enumerate(img_list):
        if hasattr(img, 'get_fdata'):
            vol = img.get_fdata()
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)  # <-- Added here
        else:
            vol = img

        vol_masked = vol * mask_data

        print(f"vol min: {np.nanmin(vol)}, max: {np.nanmax(vol)}")
        print(f"mask min: {np.nanmin(mask_data)}, max: {np.nanmax(mask_data)}")
        print(f"vol_masked min: {np.nanmin(vol_masked)}, max: {np.nanmax(vol_masked)}")

        mid_slice = vol_masked.shape[2] // 2
        slice_2d = vol_masked[:, :, mid_slice]
        newy[:, :, i] = slice_2d
        slices_list.append(slice_2d)
        print(f"Processed masked image {i + 1} / {num_subjects}")

    affine = mask_img.affine
    output_nii = nib.Nifti1Image(newy, affine)
    nib.save(output_nii, output_path)
    print(f"\n✅ Done! Saved masked combined slices to: {output_path}\n")

    return slices_list


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
