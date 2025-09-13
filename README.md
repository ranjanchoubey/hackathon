#  Prostate WSI Segmentation - Hackathon Solution

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A complete deep learning solution for semantic segmentation of prostate H&E stained Whole Slide Images (WSIs) using U-Net++ with ResNet34 encoder. This project provides automated inference pipeline with comprehensive visualizations and cross-platform compatibility.

##  **QUICK START**

###  **Linux & macOS Users**
```bash
./run_in_linux.sh
```

###  **Windows Users** 
```powershell
.\run_in_winodows.ps1
```

That's it! The scripts will automatically:
-  Set up Python environment
-  Install dependencies  
-  Generate predictions for all test images
-  Create organized visualizations
-  Optionally create submission package

---

##  **Model Performance**

| Metric | Training | Validation |
|--------|----------|------------|
| **WSI-level IoU** | **70.32%** | **65.18%** |
| **Architecture** | U-Net++ with ResNet34 encoder | |
| **Loss Function** | Combined Dice + Cross-Entropy | |
| **Patch Size** | 256×256 with stride 128 | |

### **Segmentation Classes**
- **Background (0)** - Non-tissue areas
- **Stroma (1)** - Stromal tissue 
- **Benign (2)** - Benign epithelium
- **Tumor (3)** - Tumor regions

---

##  **Project Structure**

```
├── hackathon/
│   ├──  run_in_linux.sh          # Linux/macOS automation script
│   ├──  run_in_winodows.ps1      # Windows PowerShell script  
│   ├──  create_submission.py     # Submission package creator
│   ├──  requirements.txt         # Python dependencies
│   ├──  LICENSE                  # MIT License
│   │
│   ├──  datasets/               # Training and test data
│   │   ├── Training/              # Training images and masks
│   │   ├── Validation/            # Validation images and masks  
│   │   ├── Test/                  # Test images for inference
│   │   └── Extra/                 # Additional data
│   │
│   ├──  models/                 # Trained model weights
│   │   ├── best_model.pth         # Best performing model
│   │   └── model_info.txt         # Model metadata
│   │
│   ├──  notebooks/              # Jupyter notebooks
│   │   ├── 01_training_pipeline.ipynb    # Training workflow
│   │   ├── 02_inference_demo.ipynb       # Inference demonstration
│   │   └── experiment.ipynb              # experimental analysis
│   │
│   ├──  outputs/                # Generated results
│   │   ├── logs/                  # Training logs
│   │   ├── predictions/           # Raw prediction masks (.png)
│   │   └── visualizations/        # Organized visual outputs
│   │       ├── colored_masks/     # Color-coded segmentation
│   │       ├── overlays/          # Predictions on originals
│   │       ├── comparisons/       # Side-by-side views
│   │       ├── statistics/        # Class distribution charts  
│   │       └── summaries/         # Overview grids
│   │
│   ├──  scripts/                # Execution scripts
│   │   ├── run_inference.py       # Main inference pipeline
│   │   └── run_training.py        # Training script
│   │
│   ├──  src/                    # Core source code
│   │   ├── config.py              # Configuration classes
│   │   ├── model.py               # U-Net++ model definition
│   │   ├── data_loader.py         # Data loading utilities
│   │   ├── inference.py           # Inference pipeline
│   │   ├── train.py               # Training pipeline
│   │   └── utils.py               # Helper utilities
│   │
│   └──  docs/                   # Documentation
       └── Hackathon_Report.docx   # Technical report
```

---

##  **Key Features**

### ** Automated Pipeline**
- One-click execution for any operating system
- Automatic environment setup and dependency management
- Comprehensive error checking and validation

### ** Rich Visualizations**  
- Color-coded segmentation masks
- Overlay predictions on original images
- Side-by-side comparisons
- Statistical analysis charts
- Summary overview grids

### ** Cross-Platform Support**
- Linux/macOS bash script
- Windows PowerShell script
- Automatic Python environment management

### ** Production Ready**
- Automated submission package creation
- Comprehensive logging and error handling
- Clean, modular codebase

---

##  **Manual Installation** (Optional)

If you prefer manual setup:

```bash
# Clone repository
git clone <repository-url>
cd hackathon

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run inference
python scripts/run_inference.py --input datasets/Test --output outputs

# Create submission (optional)
python create_submission.py
```

---

##  **Output Structure**

After running inference, you'll get:

```
outputs/
├── predictions/                    # Raw prediction masks
│   ├── image1_prediction.png       # Grayscale masks (0,1,2,3)
│   └── ...
└── visualizations/                 # Organized visualizations
    ├── colored_masks/              # Color-coded segmentations
    ├── overlays/                   # Predictions overlaid on originals
    ├── comparisons/                # Side-by-side comparisons  
    ├── statistics/                 # Class distribution charts
    └── summaries/                  # Overview grid
        └── summary_grid.png        # All predictions in one view
```

### **Color Legend**
-  **Black (0)**: Background
-  **Green (1)**: Stroma  
-  **Yellow (2)**: Benign epithelium
-  **Red (3)**: Tumor

---

##  **Technical Details**

### **Model Architecture**
- **Base**: U-Net++ (UNet Plus Plus)
- **Encoder**: ResNet34 (ImageNet pre-trained)
- **Input Size**: 256×256 patches
- **Stride**: 128 (50% overlap)
- **Classes**: 4 (Background, Stroma, Benign, Tumor)

### **Training Configuration**
- **Loss**: Combined Dice + Cross-Entropy
- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine annealing schedule
- **Augmentations**: Rotation, flipping, color jitter
- **Validation**: WSI-level IoU metric

### **Inference Pipeline**
- **Patch Extraction**: Tissue detection and filtering
- **Batch Processing**: Efficient GPU utilization  
- **Reconstruction**: Seamless WSI-level mask stitching
- **Post-processing**: Morphological operations

---

##  **Requirements**

- **Python**: 3.8 or higher
- **GPU**: Optional (CUDA-compatible), CPU inference supported
- **RAM**: Minimum 8GB recommended
- **Storage**: ~2GB for dependencies and model

### **Key Dependencies**
- PyTorch ≥ 2.0.0
- segmentation-models-pytorch ≥ 0.3.3
- OpenCV ≥ 4.8.0
- Albumentations ≥ 1.3.1
- Matplotlib, NumPy, Pandas

---

##  **Usage Examples**

### **Basic Inference**
```bash
# Automated (recommended)
./run_in_linux.sh

# Manual
python scripts/run_inference.py --input datasets/Test --output outputs
```

### **Custom Input Directory**
```bash
python scripts/run_inference.py --input /path/to/your/images --output /path/to/results
```

### **Create Submission Package**
```bash
python create_submission.py
```

---

##  **Contributing**

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)  
5. Open Pull Request

---

##  **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  **Authors**

- **Mona Kumari** - *Lead Developer* 
- **Team Members** - *Contributors*

---

##  **Acknowledgments**

- Hackathon organizers for providing the dataset
- PyTorch and segmentation-models-pytorch communities
- Research papers that inspired the methodology

---

##  **Support**

If you encounter any issues:

1. Check the console output for error messages
2. Ensure all requirements are installed
3. Verify input data format and paths
4. Create an issue with detailed error description

**Ready to segment some prostates? Just run the script for your OS!** 
