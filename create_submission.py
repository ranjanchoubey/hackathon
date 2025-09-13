#!/usr/bin/env python3
"""
Create final submission package
"""

import os
import shutil
import cv2
import numpy as np
from datetime import datetime

def create_final_submission_package(submission_dir, your_name):
    """Create final submission package as required """

    # Create submission folder with your name (as per prompt requirements)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_submission_dir = f"./final_submission_{your_name}_{timestamp}"
    os.makedirs(final_submission_dir, exist_ok=True)

    # Copy all submission masks
    submission_files = [f for f in os.listdir(submission_dir) if f.endswith('_prediction.png')]

    print(f" Creating final submission package...")
    print(f" Submission directory: {final_submission_dir}")

    for file in submission_files:
        src = os.path.join(submission_dir, file)
        dst = os.path.join(final_submission_dir, file)
        shutil.copy(src, dst)

        # Verify the copied file has correct values
        pred_mask = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
        unique_vals = np.unique(pred_mask)
        file_size = os.path.getsize(dst) / (1024*1024)  # MB

        print(f"   {file} ({file_size:.1f}MB) - Values: {unique_vals}")

    # Create submission info file
    info_content = f"""# Prostate WSI Segmentation - Blind Test Predictions
Submitted by: {your_name}
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model Performance: 70.32% WSI-level IoU (training), 65.18% WSI-level IoU (validation)

## Files Included:
"""

    for file in sorted(submission_files):
        info_content += f"- {file}\n"

    info_content += f"""
## File Format:
- .png format with pixel values 0,1,2,3
- Background = 0, Stroma = 1, Benign = 2, Tumor = 3
- Same dimensions as original WSI
- Total files: {len(submission_files)}

## Model Details:
- Architecture: U-Net++ with ResNet34 encoder
- Training: Combined Dice + Cross-Entropy loss
- Patch size: 256x256, Stride: 128
- Best training IoU: 70.32%
"""

    with open(os.path.join(final_submission_dir, "submission_info.txt"), 'w') as f:
        f.write(info_content)

    # Create zip file for easy upload
    zip_filename = f"./submission_{your_name}_{timestamp}.zip"
    shutil.make_archive(zip_filename.replace('.zip', ''), 'zip', final_submission_dir)

    print(f"\n FINAL SUBMISSION READY!")
    print(f" Folder: {final_submission_dir}")
    print(f" Zip file: {zip_filename}")
    print(f" Total prediction files: {len(submission_files)}")
    print(f"\n Ready to upload!")

    return final_submission_dir, zip_filename

if __name__ == "__main__":
    # Set your parameters
    YOUR_NAME = "Mona_Kumari"
    submission_dir = "outputs/predictions"

    # Check if predictions exist
    if not os.path.exists(submission_dir):
        print(f" Error: Predictions directory not found: {submission_dir}")
        print("Please run inference first!")
        exit(1)
    
    pred_files = [f for f in os.listdir(submission_dir) if f.endswith('_prediction.png')]
    if not pred_files:
        print(f" Error: No prediction files found in {submission_dir}")
        print("Please run inference first!")
        exit(1)

    # Run the function
    final_dir, zip_file = create_final_submission_package(submission_dir, YOUR_NAME)
