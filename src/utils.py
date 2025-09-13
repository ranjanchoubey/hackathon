"""
Utility functions for IoU calculation, visualization, mask saving
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

# Class definitions
CLASSES = {0: 'Background', 1: 'Stroma', 2: 'Benign', 3: 'Tumor'}
CLASS_COLORS = {
    0: [0, 0, 0],        # Black - Background
    1: [0, 0, 255],      # Blue - Stroma  
    2: [0, 255, 0],      # Green - Benign
    3: [255, 255, 0]     # Yellow - Tumor
}

def calculate_iou_per_class(pred_mask, true_mask, num_classes=4):
    """Calculate IoU for each class"""
    ious = []
    
    for class_id in range(num_classes):
        pred_class = (pred_mask == class_id)
        true_class = (true_mask == class_id)
        
        intersection = (pred_class & true_class).sum()
        union = (pred_class | true_class).sum()
        
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = float(intersection) / float(union)
            
        ious.append(iou)
    
    return ious

def calculate_mean_iou(pred_masks, true_masks, num_classes=4):
    """Calculate mean IoU across all patches/WSIs"""
    all_ious = {i: [] for i in range(num_classes)}
    
    for pred, true in zip(pred_masks, true_masks):
        ious = calculate_iou_per_class(pred, true, num_classes)
        for i, iou in enumerate(ious):
            all_ious[i].append(iou)
    
    mean_ious = []
    for i in range(num_classes):
        if all_ious[i]:
            mean_ious.append(np.mean(all_ious[i]))
        else:
            mean_ious.append(0.0)
    
    return mean_ious, np.mean(mean_ious)

def reconstruct_wsi_mask(patches_pred, patch_coords, wsi_shape, patch_size=256, stride=128):
    """Reconstruct full WSI mask from predicted patches"""
    wsi_h, wsi_w = wsi_shape
    reconstructed = np.zeros((wsi_h, wsi_w), dtype=np.float32)
    count_map = np.zeros((wsi_h, wsi_w), dtype=np.float32)
    
    for pred_patch, (x, y) in zip(patches_pred, patch_coords):
        y_end = min(y + patch_size, wsi_h)
        x_end = min(x + patch_size, wsi_w)
        
        actual_h = y_end - y
        actual_w = x_end - x
        
        reconstructed[y:y_end, x:x_end] += pred_patch[:actual_h, :actual_w].astype(np.float32)
        count_map[y:y_end, x:x_end] += 1.0
    
    count_map[count_map == 0] = 1
    reconstructed = reconstructed / count_map
    
    return reconstructed.astype(np.uint8)

def save_prediction_mask(prediction_mask, save_path):
    """
    Save prediction mask with correct label values (0,1,2,3) as per prompt requirements
    """
    prediction_mask = prediction_mask.astype(np.uint8)
    cv2.imwrite(save_path, prediction_mask)
    print(f" Prediction saved: {save_path}")
    print(f"Unique values: {np.unique(prediction_mask)}")

def visualize_predictions(image, true_mask, pred_mask, save_path=None):
    """Visualize original image, ground truth, and prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original WSI')
    axes[0].axis('off')
    
    # Ground truth
    colored_true = np.zeros((*true_mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored_true[true_mask == class_id] = color
    axes[1].imshow(colored_true)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    colored_pred = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        colored_pred[pred_mask == class_id] = color
    axes[2].imshow(colored_pred)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def print_class_distribution(mask, title="Mask"):
    """Print class distribution in mask"""
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    print(f"\n {title} Class Distribution:")
    for class_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        class_name = CLASSES.get(class_id, f"Unknown({class_id})")
        print(f"  {class_name}: {count:,} pixels ({percentage:.2f}%)")
