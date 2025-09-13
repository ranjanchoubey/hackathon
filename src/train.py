"""
Training and validation loops for WSI segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from hackathon.src.model import SegmentationModel, CombinedLoss
from hackathon.src.data_loader import WSIPatchDataset, get_transforms
from hackathon.src.utils import calculate_iou_per_class, calculate_mean_iou, reconstruct_wsi_mask
from hackathon.src.data_loader import load_image, load_mask, is_tissue_patch
import albumentations as A
from albumentations.pytorch import ToTensorV2

class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        self.epochs = 50
        self.batch_size = 8
        self.learning_rate = 1e-4
        self.num_classes = 4
        self.patch_size = 256
        self.stride = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_save_path = "models"
        self.patience = 15  # Early stopping patience

def calculate_patch_iou(outputs, targets, num_classes=4):
    """Calculate IoU for patch-level predictions"""
    predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
    
    batch_ious = []
    for pred, target in zip(predictions, targets):
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        ious = calculate_iou_per_class(pred_np, target_np, num_classes)
        batch_ious.append(np.mean(ious))
    
    return np.mean(batch_ious)

def evaluate_wsi_level(model, val_dataset, device, num_classes=4):
    """
    Evaluate model at WSI level by stitching patches
    This is the key evaluation method as per prompt requirements
    """
    model.eval()
    
    # Group patches by WSI
    wsi_groups = {}
    for i in range(len(val_dataset)):
        _, _, info = val_dataset[i]
        wsi_idx = info['wsi_idx']
        if wsi_idx not in wsi_groups:
            wsi_groups[wsi_idx] = []
        wsi_groups[wsi_idx].append(i)
    
    print(f" Evaluating {len(wsi_groups)} WSIs at WSI-level...")
    
    wsi_ious = []
    inference_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    with torch.no_grad():
        for wsi_idx, patch_indices in tqdm(wsi_groups.items(), desc="WSI Evaluation"):
            # Get WSI info
            _, _, first_info = val_dataset[patch_indices[0]]
            wsi_shape = first_info['wsi_shape']
            wsi_path = first_info['wsi_path']
            
            # Load ground truth mask
            mask_path = wsi_path.replace('.png', '_mask.png')
            true_wsi_mask = load_mask(mask_path)
            
            if true_wsi_mask is None:
                continue
            
            # Collect predictions for all patches of this WSI
            patch_predictions = []
            patch_coords = []
            
            for patch_idx in patch_indices:
                image, _, info = val_dataset[patch_idx]
                coord = info['coord']
                
                # Apply inference transform
                if isinstance(image, np.ndarray):
                    transformed = inference_transform(image=image)
                    image = transformed['image']
                
                # Predict
                image = image.unsqueeze(0).to(device)
                output = model(image)
                prediction = torch.argmax(torch.softmax(output, dim=1), dim=1)
                prediction = prediction.squeeze().cpu().numpy()
                
                patch_predictions.append(prediction)
                patch_coords.append(coord)
            
            # Reconstruct WSI prediction
            reconstructed_mask = reconstruct_wsi_mask(
                patch_predictions, patch_coords, wsi_shape, 
                val_dataset.patch_size, val_dataset.stride
            )
            
            # Calculate WSI-level IoU
            wsi_iou_scores = calculate_iou_per_class(reconstructed_mask, true_wsi_mask, num_classes)
            wsi_mean_iou = np.mean(wsi_iou_scores)
            wsi_ious.append(wsi_mean_iou)
    
    # Calculate overall metrics
    overall_mean_iou = np.mean(wsi_ious) if wsi_ious else 0.0
    
    # Calculate per-class IoU across all WSIs
    all_class_ious = {i: [] for i in range(num_classes)}
    
    return overall_mean_iou, wsi_ious

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (images, masks, _) in enumerate(pbar):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_iou = calculate_patch_iou(outputs, masks)
        
        running_loss += loss.item()
        running_iou += batch_iou
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
        })
    
    return running_loss / len(dataloader), running_iou / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (images, masks, _) in enumerate(pbar):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            batch_iou = calculate_patch_iou(outputs, masks)
            
            running_loss += loss.item()
            running_iou += batch_iou
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{running_loss/(batch_idx+1):.4f}'
            })
    
    return running_loss / len(dataloader), running_iou / len(dataloader)

def train_model(train_paths, val_paths, config):
    """
    Main training function following prompt requirements
    """
    # Create datasets
    print(" Creating datasets...")
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    train_dataset = WSIPatchDataset(train_paths, transform=train_transform, 
                                   patch_size=config.patch_size, stride=config.stride)
    val_dataset = WSIPatchDataset(val_paths, transform=val_transform,
                                 patch_size=config.patch_size, stride=config.stride)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=2)
    
    print(f" Training patches: {len(train_dataset)}")
    print(f" Validation patches: {len(val_dataset)}")
    
    # Initialize model
    model = SegmentationModel(num_classes=config.num_classes, 
                             encoder_name="resnet34", pretrained=True)
    model = model.to(config.device)
    
    # Loss and optimizer
    criterion = CombinedLoss(weight_ce=1.0, weight_dice=1.0)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                    factor=0.5, patience=5, verbose=True)
    
    # Create model directory
    os.makedirs(config.model_save_path, exist_ok=True)
    
    # Training loop
    best_wsi_iou = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'wsi_iou': []}
    
    print(f" Starting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        print(f"\n Epoch {epoch+1}/{config.epochs}")
        print("=" * 60)
        
        # Train
        train_loss, train_iou = train_epoch(model, train_loader, criterion, optimizer, config.device)
        
        # Validate
        val_loss, val_iou = validate_epoch(model, val_loader, criterion, config.device)
        
        # WSI-level evaluation every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == config.epochs - 1:
            print(" Performing WSI-level validation...")
            wsi_mean_iou, _ = evaluate_wsi_level(model, val_dataset, config.device)
            history['wsi_iou'].append(wsi_mean_iou)
            
            # Save best model based on WSI-level IoU
            if wsi_mean_iou > best_wsi_iou:
                best_wsi_iou = wsi_mean_iou
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_wsi_iou': best_wsi_iou,
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, os.path.join(config.model_save_path, 'best_model.pth'))
                
                print(f" New best WSI IoU: {best_wsi_iou:.4f} - Model saved!")
            else:
                patience_counter += 1
        else:
            wsi_mean_iou = history['wsi_iou'][-1] if history['wsi_iou'] else 0.0
            patience_counter += 1
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # Learning rate scheduling
        scheduler.step(wsi_mean_iou)
        
        # Print epoch summary
        print(f"\n Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f}")
        print(f"WSI IoU: {wsi_mean_iou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f" Early stopping triggered after {config.patience} epochs without improvement")
            break
    
    print(f"\n TRAINING COMPLETED!")
    print(f" Best WSI-level IoU: {best_wsi_iou:.4f}")
    
    return history, best_wsi_iou

if __name__ == "__main__":
    # Example usage
    import glob
    
    # Get training and validation paths
    train_dir = "datasets/Training"
    val_dir = "datasets/Validation"
    
    train_images = sorted(glob.glob(os.path.join(train_dir, "*.png")))
    train_images = [img for img in train_images if not img.endswith('_mask.png')]
    train_masks = [img.replace('.png', '_mask.png') for img in train_images]
    train_paths = list(zip(train_images, train_masks))
    
    val_images = sorted(glob.glob(os.path.join(val_dir, "*.png")))
    val_images = [img for img in val_images if not img.endswith('_mask.png')]
    val_masks = [img.replace('.png', '_mask.png') for img in val_images]
    val_paths = list(zip(val_images, val_masks))
    
    print(f" Training WSIs: {len(train_paths)}")
    print(f" Validation WSIs: {len(val_paths)}")
    
    # Training configuration
    config = TrainingConfig()
    
    # Start training
    history, best_iou = train_model(train_paths, val_paths, config)
    
    print(f" Training completed! Best IoU: {best_iou:.4f}")
