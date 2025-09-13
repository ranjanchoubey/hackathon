#!/bin/bash

# ========================================
#  LINUX & macOS SCRIPT
# ========================================
# Prostate WSI Segmentation - Complete Inference Pipeline
# This script sets up the environment and runs inference with visualization
# 
# USAGE: ./run.sh

echo " Starting Prostate WSI Segmentation Pipeline..."
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo " Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo " Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo " Error: Failed to create virtual environment"
        exit 1
    fi
    echo " Virtual environment created"
else
    echo " Using existing virtual environment"
fi

# Activate virtual environment
echo " Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo " Installing required packages..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo " Error: Failed to install requirements"
    exit 1
fi
echo " Requirements installed"

# Check if test data exists
if [ ! -d "datasets/Test" ]; then
    echo " Error: Test dataset not found at datasets/Test"
    echo "Please ensure the test images are in the datasets/Test directory"
    exit 1
fi

# Check if model exists, download if not found
if [ ! -f "models/best_model.pth" ]; then
    echo " Model not found at models/best_model.pth"
    echo " Downloading pre-trained model from Google Drive..."
    
    # Create models directory if it doesn't exist
    mkdir -p models
    
    # Download using gdown (install if not present)
    if ! command -v gdown &> /dev/null; then
        echo " Installing gdown for Google Drive downloads..."
        pip install gdown
        if [ $? -ne 0 ]; then
            echo " Error: Failed to install gdown"
            echo "Please install gdown manually: pip install gdown"
            exit 1
        fi
    fi
    
    # Download the model file
    echo " Downloading model (this may take a few minutes)..."
    gdown "https://drive.google.com/uc?id=1MwaxSbJ4H508Pp2Cmpl8TSbAWycDMlUS" -O models/best_model.pth
    
    if [ $? -ne 0 ]; then
        echo " Error: Failed to download model from Google Drive"
        echo "Please download manually from: https://drive.google.com/file/d/1MwaxSbJ4H508Pp2Cmpl8TSbAWycDMlUS/view"
        echo "Save it as: models/best_model.pth"
        exit 1
    fi
    
    # Verify downloaded file
    if [ -f "models/best_model.pth" ]; then
        echo " Model downloaded successfully!"
        # Check file size (should be reasonable for a model file)
        file_size=$(ls -la models/best_model.pth | awk '{print $5}')
        if [ "$file_size" -lt 1000000 ]; then  # Less than 1MB might indicate download error
            echo " Warning: Downloaded file seems too small. Please verify the download."
        fi
    else
        echo " Error: Model download failed"
        exit 1
    fi
else
    echo " Model found at models/best_model.pth"
fi

# Create output directory
mkdir -p outputs

echo ""
echo " Running inference pipeline..."
echo " Input: datasets/Test"
echo " Output: outputs"
echo ""

# Run inference with visualization
python scripts/run_inference.py --input datasets/Test --output outputs

if [ $? -eq 0 ]; then
    echo ""
    echo " Pipeline completed successfully!"
    echo " Results available in:"
    echo "  - outputs/predictions/ (raw prediction masks)"
    echo "  - outputs/visualizations/ (organized visualizations)"
    echo ""
    echo " Visualization folders:"
    echo "  - colored_masks/ (color-coded segmentation)"
    echo "  - overlays/ (predictions on original images)"
    echo "  - comparisons/ (side-by-side views)"
    echo "  - statistics/ (class distribution charts)"
    echo "  - summaries/ (overview grid)"
    echo ""
    echo " Ready for submission package creation!"
else
    echo " Error: Inference pipeline failed"
    exit 1
fi