#!/usr/bin/env python3
"""
Standalone training script
Usage: python scripts/run_training.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import TrainingConfig
from train import train_model
import glob

def main():
    """Run training pipeline"""
    config = TrainingConfig()
    
    # Get training paths
    train_images = sorted(glob.glob(str(config.TRAIN_DIR / "*.png")))
    train_images = [img for img in train_images if not img.endswith('_mask.png')]
    train_masks = [img.replace('.png', '_mask.png') for img in train_images]
    train_paths = list(zip(train_images, train_masks))
    
    # Get validation paths
    val_images = sorted(glob.glob(str(config.VAL_DIR / "*.png")))
    val_images = [img for img in val_images if not img.endswith('_mask.png')]
    val_masks = [img.replace('.png', '_mask.png') for img in val_images]
    val_paths = list(zip(val_images, val_masks))
    
    print(f"📊 Training WSIs: {len(train_paths)}")
    print(f"📊 Validation WSIs: {len(val_paths)}")
    
    # Start training
    history, best_iou = train_model(train_paths, val_paths, config)
    print(f"🏆 Training completed! Best IoU: {best_iou:.4f}")

if __name__ == "__main__":
    main()