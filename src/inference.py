"""
WSI inference pipeline with stitching - generates prediction masks as per prompt requirements
"""

import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.model import SegmentationModel
from src.utils import reconstruct_wsi_mask, save_prediction_mask, visualize_predictions
from src.data_loader import load_image, load_mask, is_tissue_patch

def predict_wsi(model, wsi_path, patch_size=256, stride=128, device='cuda', min_tissue_ratio=0.1):
    """
    Predict full WSI by processing patches and stitching results
    Returns prediction mask with label values (0, 1, 2, 3) as per prompt requirements
    """
    # Load WSI
    wsi_image = load_image(wsi_path)
    if wsi_image is None:
        raise ValueError(f"Could not load image: {wsi_path}")
    
    print(f"🔍 Processing WSI: {os.path.basename(wsi_path)}")
    print(f"WSI dimensions: {wsi_image.shape}")
    
    model.eval()
    
    # Extract patches for inference
    patches_img, coords = extract_patches_for_inference(
        wsi_image, patch_size, stride, min_tissue_ratio
    )
    
    if len(patches_img) == 0:
        print("⚠️ No tissue patches found!")
        return np.zeros(wsi_image.shape[:2], dtype=np.uint8)
    
    print(f"📊 Extracted {len(patches_img)} tissue patches")
    
    # Process patches in batches
    batch_size = 16
    all_predictions = []
    
    # Define transforms for inference (same as training)
    inference_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    with torch.no_grad():
        for i in tqdm(range(0, len(patches_img), batch_size), desc="🔮 Predicting patches"):
            batch_patches = patches_img[i:i+batch_size]
            
            # Apply transforms
            batch_tensor = []
            for patch in batch_patches:
                transformed = inference_transform(image=patch)
                batch_tensor.append(transformed['image'])
            
            batch_tensor = torch.stack(batch_tensor).to(device)
            
            # Predict
            outputs = model(batch_tensor)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
    
    # Reconstruct full WSI - this creates the final prediction mask
    print("🧩 Reconstructing full WSI...")
    full_prediction = reconstruct_wsi_mask(
        all_predictions, coords, wsi_image.shape[:2], patch_size, stride
    )
    
    print(f"✅ Prediction complete! Shape: {full_prediction.shape}")
    print(f"Unique values: {np.unique(full_prediction)} (0=Background, 1=Stroma, 2=Benign, 3=Tumor)")
    
    return full_prediction

def extract_patches_for_inference(image, patch_size=256, stride=128, min_tissue_ratio=0.1):
    """Extract patches for inference (no masks needed)"""
    patches_img = []
    coordinates = []
    
    h, w = image.shape[:2]
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch_img = image[y:y+patch_size, x:x+patch_size]
            
            if is_tissue_patch(patch_img, min_tissue_ratio):
                patches_img.append(patch_img)
                coordinates.append((x, y))
    
    return np.array(patches_img), coordinates

def generate_validation_predictions(model_path, validation_dir, output_dir, device='cuda'):
    """
    Generate predictions for all validation WSIs as per prompt requirements
    Saves prediction masks with correct label values (0,1,2,3) in .png format
    """
    import glob
    
    # Load trained model - FIX: Add weights_only=False for compatibility
    print(f"📥 Loading model from: {model_path}")
    model = SegmentationModel(num_classes=4, encoder_name="resnet34", pretrained=False)
    
    # Fix for PyTorch version compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded! Best WSI IoU: {checkpoint['best_wsi_iou']:.4f}")
    
    # Get validation paths
    val_images = sorted(glob.glob(os.path.join(validation_dir, "*.png")))
    val_images = [img for img in val_images if not img.endswith('_mask.png')]
    val_masks = [img.replace('.png', '_mask.png') for img in val_images]
    
    print(f"📊 Found {len(val_images)} validation WSIs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    results = []
    for i, (img_path, mask_path) in enumerate(tqdm(zip(val_images, val_masks), desc="🎯 Generating predictions")):
        # Get image name
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # Predict WSI
        prediction = predict_wsi(model, img_path, device=device)
        
        # Save prediction with correct label values as per prompt
        output_path = os.path.join(output_dir, f"{img_name}_prediction.png")
        save_prediction_mask(prediction, output_path)
        
        # Load ground truth and calculate IoU for validation
        if os.path.exists(mask_path):
            from hackathon.src.utils import calculate_iou_per_class
            true_mask = load_mask(mask_path)
            if true_mask is not None:
                iou_scores = calculate_iou_per_class(prediction, true_mask)
                mean_iou = np.mean(iou_scores)
                
                print(f"📊 {img_name} IoU: {mean_iou:.4f}")
                print(f"   Per-class IoU: Background={iou_scores[0]:.3f}, Stroma={iou_scores[1]:.3f}, Benign={iou_scores[2]:.3f}, Tumor={iou_scores[3]:.3f}")
                
                results.append({
                    'image': img_name,
                    'prediction_path': output_path,
                    'shape': prediction.shape,
                    'unique_values': np.unique(prediction).tolist(),
                    'iou_per_class': iou_scores,
                    'mean_iou': mean_iou
                })
        else:
            results.append({
                'image': img_name,
                'prediction_path': output_path,
                'shape': prediction.shape,
                'unique_values': np.unique(prediction).tolist()
            })
    
    return results

def generate_test_predictions(model_path, test_dir, output_dir, device='cuda'):
    """
    Generate predictions for blind test WSIs (when provided)
    Saves prediction masks with correct label values (0,1,2,3) in .png format
    """
    import glob
    
    # Load trained model - FIX: Add weights_only=False for compatibility
    print(f"📥 Loading model from: {model_path}")
    model = SegmentationModel(num_classes=4, encoder_name="resnet34", pretrained=False)
    
    # Fix for PyTorch version compatibility
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Model loaded! Best WSI IoU: {checkpoint['best_wsi_iou']:.4f}")
    
    # Get test image paths
    test_images = sorted(glob.glob(os.path.join(test_dir, "*.png")))
    print(f"📊 Found {len(test_images)} test WSIs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate predictions
    results = []
    for img_path in tqdm(test_images, desc="🎯 Generating test predictions"):
        # Get image name
        img_name = os.path.basename(img_path).replace('.png', '')
        
        # Predict WSI
        prediction = predict_wsi(model, img_path, device=device)
        
        # Save prediction with correct label values as per prompt requirements
        output_path = os.path.join(output_dir, f"{img_name}_prediction.png")
        save_prediction_mask(prediction, output_path)
        
        results.append({
            'image': img_name,
            'prediction_path': output_path,
            'shape': prediction.shape,
            'unique_values': np.unique(prediction).tolist()
        })
    
    return results

if __name__ == "__main__":
    # Example usage for validation predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define paths
    model_path = "models/best_model.pth"
    validation_dir = "datasets/Validation"
    output_dir = "predictions/validation"
    
    # Generate validation predictions
    print("🚀 Generating validation predictions...")
    results = generate_validation_predictions(model_path, validation_dir, output_dir, device)
    
    print(f"\n🎉 Generated {len(results)} validation predictions!")
    print("📁 Saved in: predictions/validation/")
    print("🎯 Ready for blind test WSIs when provided!")
