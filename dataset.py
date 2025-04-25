import os
import torch  # Add this line
from torch.utils.data import Dataset
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.npy')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.npy')])
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch in the number of images and masks"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])
        mask = np.load(self.mask_paths[idx])

        # Ensure both image and mask have the same shape
        assert image.shape == mask.shape, f"Image and mask shapes do not match: {image.shape} vs {mask.shape}"
        
        # Convert to tensor
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)
        
        # Handle both 3D (single slice) and 4D (multiple slices) data
        if len(image.shape) == 3:  # Single slice (H, W, C)
            image = image.permute(2, 0, 1)  # (C, H, W)
            mask = mask.permute(2, 0, 1)    # (C, H, W)
        elif len(image.shape) == 4:  # Multiple slices (S, H, W, C)
            image = image.permute(0, 3, 1, 2)  # (S, C, H, W)
            mask = mask.permute(0, 3, 1, 2)    # (S, C, H, W)
        
        # Squeeze the slice dimension if there's only one slice
        if image.shape[0] == 1:
            image = image.squeeze(0)
            mask = mask.squeeze(0)
        
        return image, mask