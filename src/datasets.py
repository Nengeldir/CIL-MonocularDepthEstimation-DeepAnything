import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class DepthDataset(Dataset):
    def __init__(self, data_dir, list_file, transform=None, target_transform=None, has_gt=True):
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.has_gt = has_gt
        
        # Read file list
        with open(list_file, 'r') as f:
            if has_gt:
                self.file_pairs = [line.strip().split() for line in f]
            else:
                # For test set without ground truth
                self.file_list = [line.strip() for line in f]
    
    def __len__(self):
        return len(self.file_pairs if self.has_gt else self.file_list)
    
    def __getitem__(self, idx):
        if self.has_gt:
            rgb_path = os.path.join(self.data_dir, self.file_pairs[idx][0])
            depth_path = os.path.join(self.data_dir, self.file_pairs[idx][1])
            
            # Load RGB image
            rgb = Image.open(rgb_path).convert('RGB')
            
            # Load depth map
            depth = np.load(depth_path).astype(np.float32)
            depth = torch.from_numpy(depth)
            
            # Apply transformations
            if self.transform:
                rgb = self.transform(rgb)
            
            if self.target_transform:
                depth = self.target_transform(depth)
            else:
                # Add channel dimension if not done by transform
                depth = depth.unsqueeze(0)
            
            return rgb, depth, self.file_pairs[idx][0]  # Return filename for saving predictions
        else:
            # For test set without ground truth
            rgb_path = os.path.join(self.data_dir, self.file_list[idx].split(' ')[0])
            
            # Load RGB image
            rgb = Image.open(rgb_path).convert('RGB')
            
            # Apply transformations
            if self.transform:
                rgb = self.transform(rgb)
            
            return rgb, self.file_list[idx]  # No depth, just return the filename
        
