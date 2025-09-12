#!/usr/bin/env python3
"""
Standalone inference script with automatic visualization generation
Usage: python scripts/run_inference.py --input path/to/test/images --output path/to/predictions
"""

import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from src.config import InferenceConfig, TrainingConfig
from src.inference import generate_test_predictions
import torch

def create_color_map():
    """Create color map for segmentation classes"""
    colors = {
        0: [0, 0, 0],       # Background - Black
        1: [0, 255, 0],     # Stroma - Green  
        2: [255, 255, 0],   # Benign - Yellow
        3: [255, 0, 0]      # Tumor - Red
    }
    return colors

def mask_to_color(mask, colors):
    """Convert grayscale mask to colored visualization"""
    h, w = mask.shape
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    
    return colored_mask

def create_organized_visualizations(predictions_dir, test_dir, viz_base_dir):
    """Create organized visualizations directly in structured folders"""
    
    # Create organized folder structure
    folders = ['colored_masks', 'overlays', 'comparisons', 'statistics', 'summaries']
    for folder in folders:
        os.makedirs(os.path.join(viz_base_dir, folder), exist_ok=True)
    
    colors = create_color_map()
    pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('_prediction.png')]
    
    print(f"🎨 Creating organized visualizations for {len(pred_files)} predictions...")
    
    for pred_file in pred_files:
        base_name = pred_file.replace('_prediction.png', '')
        print(f"  Processing {base_name}...")
        
        # Load prediction
        pred_path = os.path.join(predictions_dir, pred_file)
        prediction_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        # Load original (if exists)
        original_file = f"{base_name}.png"
        orig_path = os.path.join(test_dir, original_file)
        
        if os.path.exists(orig_path):
            original_img = cv2.imread(orig_path)
        else:
            # Create placeholder if original not found
            original_img = np.ones_like(prediction_mask) * 128
            original_img = cv2.cvtColor(original_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # 1. Colored mask - save directly to colored_masks folder
        colored_mask = mask_to_color(prediction_mask, colors)
        colored_path = os.path.join(viz_base_dir, 'colored_masks', f"{base_name}_colored_mask.png")
        cv2.imwrite(colored_path, colored_mask)
        
        # 2. Overlay - save directly to overlays folder
        if len(original_img.shape) == 3:
            overlay = cv2.addWeighted(original_img, 0.4, colored_mask, 0.6, 0)
        else:
            overlay = colored_mask
        overlay_path = os.path.join(viz_base_dir, 'overlays', f"{base_name}_overlay.png")
        cv2.imwrite(overlay_path, overlay)
        
        # 3. Side-by-side comparison - save directly to comparisons folder
        if len(original_img.shape) == 3:
            comparison = np.hstack([original_img, colored_mask])
        else:
            original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
            comparison = np.hstack([original_rgb, colored_mask])
        comparison_path = os.path.join(viz_base_dir, 'comparisons', f"{base_name}_comparison.png")
        cv2.imwrite(comparison_path, comparison)
        
        # 4. Statistics - save directly to statistics folder
        create_class_statistics_plot(prediction_mask, 
            os.path.join(viz_base_dir, 'statistics', f"{base_name}_statistics.png"))
    
    # 5. Summary grid - save directly to summaries folder
    create_summary_visualization(pred_files, predictions_dir, viz_base_dir, colors)
    
    print(f"✅ All visualizations organized in: {viz_base_dir}")

def create_class_statistics_plot(prediction_mask, save_path):
    """Create bar plot of class distribution"""
    unique, counts = np.unique(prediction_mask, return_counts=True)
    total_pixels = prediction_mask.shape[0] * prediction_mask.shape[1]
    
    class_names = ['Background', 'Stroma', 'Benign', 'Tumor']
    class_colors = ['black', 'green', 'yellow', 'red']
    
    percentages = []
    for class_id in range(4):
        if class_id in unique:
            idx = np.where(unique == class_id)[0][0]
            percentage = (counts[idx] / total_pixels) * 100
        else:
            percentage = 0
        percentages.append(percentage)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, percentages, color=class_colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Percentage of Pixels')
    plt.title('Class Distribution in Prediction')
    plt.ylim(0, 100)
    
    for bar, percentage in zip(bars, percentages):
        if percentage > 0:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{percentage:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def create_summary_visualization(pred_files, predictions_dir, viz_base_dir, colors):
    """Create summary grid directly in summaries folder"""
    if not pred_files:
        return
    
    n_files = len(pred_files)
    cols = min(3, n_files)
    rows = (n_files + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    for i, pred_file in enumerate(pred_files):
        pred_path = os.path.join(predictions_dir, pred_file)
        prediction_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        colored_mask = mask_to_color(prediction_mask, colors)
        colored_mask_rgb = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
        
        axes[i].imshow(colored_mask_rgb)
        axes[i].set_title(pred_file.replace('_prediction.png', ''), fontsize=10)
        axes[i].axis('off')
        
        # Add statistics
        unique, counts = np.unique(prediction_mask, return_counts=True)
        total_pixels = prediction_mask.shape[0] * prediction_mask.shape[1]
        
        stats_text = ""
        class_names = ['Bg', 'St', 'Be', 'Tu']
        for class_id in range(4):
            if class_id in unique:
                idx = np.where(unique == class_id)[0][0]
                percentage = (counts[idx] / total_pixels) * 100
                stats_text += f"{class_names[class_id]}: {percentage:.1f}%\n"
        
        axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Hide unused subplots
    for i in range(n_files, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('All Predictions Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_base_dir, 'summaries', 'summary_grid.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Run inference on WSI images')
    parser.add_argument('--input', required=True, help='Path to input WSI directory')
    parser.add_argument('--output', required=True, help='Path to output directory')
    parser.add_argument('--model', default=None, help='Path to model weights (optional)')
    
    args = parser.parse_args()
    
    config = InferenceConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = args.model or str(config.BEST_MODEL_PATH)
    
    print(f"🔧 Device: {device}")
    print(f"📥 Input: {args.input}")
    print(f"📤 Output: {args.output}")
    print(f"🧠 Model: {model_path}")
    
    # Create output directories
    predictions_dir = os.path.join(args.output, 'predictions')
    visualizations_dir = os.path.join(args.output, 'visualizations')
    
    # Run inference
    print("\n🚀 Step 1: Generating predictions...")
    results = generate_test_predictions(
        model_path=model_path,
        test_dir=args.input,
        output_dir=predictions_dir,
        device=device
    )
    
    print(f"✅ Generated {len(results)} predictions!")
    
    # Create organized visualizations
    print("\n🎨 Step 2: Creating organized visualizations...")
    create_organized_visualizations(predictions_dir, args.input, visualizations_dir)
    
    print(f"\n🎯 Complete! Output structure:")
    print(f"📁 {args.output}/")
    print(f"  📁 predictions/ - Raw prediction masks")
    print(f"  📁 visualizations/")
    print(f"    📁 colored_masks/ - Colored segmentation masks")
    print(f"    📁 overlays/ - Overlays on original images")
    print(f"    📁 comparisons/ - Side-by-side comparisons")
    print(f"    📁 statistics/ - Class distribution charts")
    print(f"    📁 summaries/ - Summary grid visualization")

if __name__ == "__main__":
    main()