# ========================================
# 🪟 WINDOWS POWERSHELL SCRIPT (RECOMMENDED)
# ========================================
# Prostate WSI Segmentation - Complete Inference Pipeline
# This script sets up the environment and runs inference with visualization
# 
# USAGE: Right-click → "Run with PowerShell" OR type ".\run.ps1" in PowerShell

Write-Host " Starting Prostate WSI Segmentation Pipeline..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host " Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host " Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and add it to your PATH" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (!(Test-Path "venv")) {
    Write-Host " Creating virtual environment..." -ForegroundColor Cyan
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host " Error: Failed to create virtual environment" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host " Virtual environment created" -ForegroundColor Green
} else {
    Write-Host " Using existing virtual environment" -ForegroundColor Cyan
}

# Activate virtual environment
Write-Host " Activating virtual environment..." -ForegroundColor Cyan
& "venv\Scripts\Activate.ps1"

# Install/upgrade requirements
Write-Host " Installing required packages..." -ForegroundColor Cyan
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host " Error: Failed to install requirements" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host " Requirements installed" -ForegroundColor Green

# Check if test data exists
if (!(Test-Path "datasets\Test")) {
    Write-Host " Error: Test dataset not found at datasets\Test" -ForegroundColor Red
    Write-Host "Please ensure the test images are in the datasets\Test directory" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if model exists, download if not found
if (!(Test-Path "models\best_model.pth")) {
    Write-Host " Model not found at models\best_model.pth" -ForegroundColor Yellow
    Write-Host " Downloading pre-trained model from Google Drive..." -ForegroundColor Cyan
    
    # Create models directory if it doesn't exist
    if (!(Test-Path "models")) {
        New-Item -ItemType Directory -Path "models" | Out-Null
    }
    
    # Check if gdown is installed, install if not
    try {
        $gdownCheck = pip show gdown 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host " Installing gdown for Google Drive downloads..." -ForegroundColor Cyan
            pip install gdown
            if ($LASTEXITCODE -ne 0) {
                Write-Host " Error: Failed to install gdown" -ForegroundColor Red
                Write-Host "Please install gdown manually: pip install gdown" -ForegroundColor Yellow
                Read-Host "Press Enter to exit"
                exit 1
            }
        }
    } catch {
        Write-Host " Installing gdown for Google Drive downloads..." -ForegroundColor Cyan
        pip install gdown
    }
    
    # Download the model file
    Write-Host " Downloading model (this may take a few minutes)..." -ForegroundColor Cyan
    gdown "https://drive.google.com/uc?id=1MwaxSbJ4H508Pp2Cmpl8TSbAWycDMlUS" -O models\best_model.pth
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host " Error: Failed to download model from Google Drive" -ForegroundColor Red
        Write-Host "Please download manually from: https://drive.google.com/file/d/1MwaxSbJ4H508Pp2Cmpl8TSbAWycDMlUS/view" -ForegroundColor Yellow
        Write-Host "Save it as: models\best_model.pth" -ForegroundColor Yellow
        Read-Host "Press Enter to exit"
        exit 1
    }
    
    # Verify downloaded file
    if (Test-Path "models\best_model.pth") {
        Write-Host " Model downloaded successfully!" -ForegroundColor Green
        # Check file size (should be reasonable for a model file)
        $fileInfo = Get-Item "models\best_model.pth"
        if ($fileInfo.Length -lt 1000000) {  # Less than 1MB might indicate download error
            Write-Host " Warning: Downloaded file seems too small. Please verify the download." -ForegroundColor Yellow
        }
    } else {
        Write-Host " Error: Model download failed" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Host " Model found at models\best_model.pth" -ForegroundColor Green
}

# Create output directory
if (!(Test-Path "outputs")) {
    New-Item -ItemType Directory -Path "outputs" | Out-Null
}

Write-Host ""
Write-Host " Running inference pipeline..." -ForegroundColor Yellow
Write-Host " Input: datasets\Test" -ForegroundColor Cyan
Write-Host " Output: outputs" -ForegroundColor Cyan
Write-Host ""

# Run inference with visualization
python scripts\run_inference.py --input datasets\Test --output outputs

if ($LASTEXITCODE -ne 0) {
    Write-Host " Error: Inference pipeline failed" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host " Pipeline completed successfully!" -ForegroundColor Green
Write-Host " Results available in:" -ForegroundColor Cyan
Write-Host "  - outputs\predictions\ (raw prediction masks)" -ForegroundColor White
Write-Host "  - outputs\visualizations\ (organized visualizations)" -ForegroundColor White
Write-Host ""
Write-Host " Visualization folders:" -ForegroundColor Cyan
Write-Host "  - colored_masks\ (color-coded segmentation)" -ForegroundColor White
Write-Host "  - overlays\ (predictions on original images)" -ForegroundColor White
Write-Host "  - comparisons\ (side-by-side views)" -ForegroundColor White
Write-Host "  - statistics\ (class distribution charts)" -ForegroundColor White
Write-Host "  - summaries\ (overview grid)" -ForegroundColor White
Write-Host ""

# Ask if user wants to create submission package
$response = Read-Host " Do you want to create the final submission package? (y/n)"
if ($response -match "^[Yy]") {
    Write-Host ""
    Write-Host " Creating final submission package..." -ForegroundColor Cyan
    python create_submission.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host " Error creating submission package" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-Host " Complete! Your submission is ready for upload!" -ForegroundColor Green
} else {
    Write-Host " Inference complete! Run 'python create_submission.py' when ready to submit." -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"
