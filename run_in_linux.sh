#!/bin/bash

# ========================================
# 🐧🍎 LINUX & macOS SCRIPT
# ========================================
# Prostate WSI Segmentation - Complete Inference Pipeline
# This script sets up the environment and runs inference with visualization
# 
# USAGE: ./run.sh

echo "🚀 Starting Prostate WSI Segmentation Pipeline..."
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to create virtual environment"
        exit 1
    fi
    echo "✅ Virtual environment created"
else
    echo "📦 Using existing virtual environment"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📥 Installing required packages..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to install requirements"
    exit 1
fi
echo "✅ Requirements installed"

# Check if test data exists
if [ ! -d "datasets/Test" ]; then
    echo "❌ Error: Test dataset not found at datasets/Test"
    echo "Please ensure the test images are in the datasets/Test directory"
    exit 1
fi

# Check if model exists
if [ ! -f "models/best_model.pth" ]; then
    echo "❌ Error: Trained model not found at models/best_model.pth"
    echo "Please ensure the model file exists"
    exit 1
fi

# Create output directory
mkdir -p outputs

echo ""
echo "🎯 Running inference pipeline..."
echo "📥 Input: datasets/Test"
echo "📤 Output: outputs"
echo ""

# Run inference with visualization
python scripts/run_inference.py --input datasets/Test --output outputs

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Pipeline completed successfully!"
    echo "📊 Results available in:"
    echo "  - outputs/predictions/ (raw prediction masks)"
    echo "  - outputs/visualizations/ (organized visualizations)"
    echo ""
    echo "📁 Visualization folders:"
    echo "  - colored_masks/ (color-coded segmentation)"
    echo "  - overlays/ (predictions on original images)"
    echo "  - comparisons/ (side-by-side views)"
    echo "  - statistics/ (class distribution charts)"
    echo "  - summaries/ (overview grid)"
    echo ""
    echo "🚀 Ready for submission package creation!"
else
    echo "❌ Error: Inference pipeline failed"
    exit 1
fi