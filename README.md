# 🏥 Prostate WSI Segmentation - Hackathon Solution

A deep learning solution for semantic segmentation of prostate H&E stained Whole Slide Images (WSIs) using U-Net++ with ResNet34 encoder.

## 🏆 Results
- **Best WSI-level IoU**: 70.32% (training), 65.18% (validation)
- **Architecture**: U-Net++ with ResNet34 encoder
- **Classes**: Background, Stroma, Benign, Tumor

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Verify Installation
```bash
python -c "from src.config import Config; print('✅ Setup successful!')"
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)
```python
# Clone and run in Colab
!git clone https://github.com/YOUR_USERNAME/hackathon.git
%cd hackathon
!pip install -r requirements.txt

# Open and run: notebooks/01_training_pipeline.ipynb
```

### Option 2: Local Setup
```bash
git clone https://github.com/YOUR_USERNAME/hackathon.git
cd hackathon
pip install -r requirements.txt

# Run training
python scripts/run_training.py

# Or use notebooks
jupyter notebook notebooks/01_training_pipeline.ipynb
```

## 📂 Project Structure
```
hackathon/
├── 📓 notebooks/          # Jupyter notebooks for training & inference
├── 🧠 src/                # Source code (models, training, inference)
├── 📊 datasets/           # Training, validation, test data
├── 🎯 models/             # Trained model weights
├── 📈 outputs/            # Predictions, visualizations, logs
└── 📚 docs/               # Documentation
```

## 🛠️ Core Components
- **`src/model.py`**: U-Net++ segmentation model
- **`src/data_loader.py`**: WSI patch extraction & data loading
- **`src/train.py`**: Training pipeline with WSI-level evaluation
- **`src/inference.py`**: WSI inference with patch stitching
- **`src/utils.py`**: IoU calculation, visualization helpers

## 📊 Dataset
- **Training**: 24 WSIs with masks
- **Validation**: 6 WSIs with masks  
- **Test**: 6 blind test WSIs
- **Extra**: 30 additional WSIs (unlabeled)

## 🎯 Model Architecture
```python
UnetPlusPlus(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=4,  # Background, Stroma, Benign, Tumor
    activation=None
)
```

## 📈 Training Details
- **Loss**: Combined Dice + Cross-Entropy
- **Optimizer**: Adam (lr=1e-4)
- **Scheduler**: ReduceLROnPlateau
- **Patch Size**: 256×256, Stride: 128
- **Augmentations**: Flips, rotations, color jittering
- **Early Stopping**: Patience=15 epochs

## 🔍 Usage Examples

### Training
```python
from src.train import train_model
from src.config import TrainingConfig

config = TrainingConfig()
history, best_iou = train_model(train_paths, val_paths, config)
```

### Inference
```python
from src.inference import predict_wsi
from src.model import SegmentationModel

model = SegmentationModel(num_classes=4)
# Load weights...
prediction = predict_wsi(model, "path/to/wsi.png")
```

## 🎯 Quick Inference
```bash
# Run inference on new WSI images
python scripts/run_inference.py \
    --input path/to/test/images \
    --output path/to/predictions

# Or use the notebook
jupyter notebook notebooks/02_inference_demo.ipynb
```

## 📝 Citation
```bibtex
@misc{prostate_wsi_segmentation,
  title={Prostate WSI Segmentation using U-Net++},
  author={Your Name},
  year={2024},
  howpublished={Hackathon Solution}
}
```

## 🤝 Contributing
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.