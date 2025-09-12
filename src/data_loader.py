"""
Patch extraction, masks, and dataloaders for WSI segmentation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# Class definitions
CLASSES = {0: 'Background', 1: 'Stroma', 2: 'Benign', 3: 'Tumor'}
CLASS_COLORS = {
    0: [0, 0, 0],        # Black - Background
    1: [0, 0, 255],      # Blue - Stroma  
    2: [0, 255, 0],      # Green - Benign
    3: [255, 255, 0]     # Yellow - Tumor
}

class WSIPatchDataset(Dataset):
    """Dataset for WSI patches with masks"""
    
    def __init__(self, wsi_paths, transform=None, patch_size=256, stride=128, min_tissue_ratio=0.1):
        self.wsi_paths = wsi_paths
        self.transform = transform
        self.patch_size = patch_size
        self.stride = stride
        self.min_tissue_ratio = min_tissue_ratio
        
        self.patches_img = []
        self.patches_mask = []
        self.wsi_info = []
        
        self._extract_all_patches()
    
    def _extract_all_patches(self):
        """Extract patches from all WSI files"""
        print("🔄 Extracting patches from all WSIs...")
        
        for idx, (img_path, mask_path) in enumerate(tqdm(self.wsi_paths)):
            wsi_img = load_image(img_path)
            wsi_mask = load_mask(mask_path)
            
            if wsi_img is not None and wsi_mask is not None:
                patches_img, patches_mask, coords = extract_patches(
                    wsi_img, wsi_mask, self.patch_size, self.stride, self.min_tissue_ratio
                )
                
                for i in range(len(patches_img)):
                    self.patches_img.append(patches_img[i])
                    self.patches_mask.append(patches_mask[i])
                    self.wsi_info.append({
                        'wsi_idx': idx,
                        'wsi_path': img_path,
                        'coord': coords[i],
                        'wsi_shape': wsi_img.shape[:2]
                    })
        
        print(f"✅ Total patches extracted: {len(self.patches_img)}")
    
    def __len__(self):
        return len(self.patches_img)
    
    def __getitem__(self, idx):
        image = self.patches_img[idx].copy()
        mask = self.patches_mask[idx].copy()
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long(), self.wsi_info[idx]

def load_image(image_path):
    """Load image as RGB numpy array"""
    image = cv2.imread(image_path)
    if image is not None:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return None

def load_mask(mask_path):
    """Load mask and convert RGB colors to class indices"""
    mask = cv2.imread(mask_path)
    if mask is not None:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        class_mask = np.zeros(mask.shape[:2], dtype=np.uint8)
        
        for class_id, color in CLASS_COLORS.items():
            class_pixels = np.all(mask == color, axis=2)
            class_mask[class_pixels] = class_id
            
        return class_mask
    return None

def is_tissue_patch(patch, tissue_threshold=0.1):
    """Check if patch contains sufficient tissue"""
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    tissue_pixels = np.sum(gray < 200)
    total_pixels = gray.shape[0] * gray.shape[1]
    return (tissue_pixels / total_pixels) > tissue_threshold

def extract_patches(image, mask, patch_size=256, stride=128, min_tissue_ratio=0.1):
    """Extract patches from WSI and corresponding mask"""
    patches_img, patches_mask, coordinates = [], [], []
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = image[y:y+patch_size, x:x+patch_size]
            patch_mask = mask[y:y+patch_size, x:x+patch_size]
            
            if is_tissue_patch(patch_img, min_tissue_ratio):
                patches_img.append(patch_img)
                patches_mask.append(patch_mask)
                coordinates.append((x, y))
    
    return np.array(patches_img), np.array(patches_mask), coordinates

def get_transforms(is_training=True):
    """Get augmentation transforms"""
    if is_training:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
