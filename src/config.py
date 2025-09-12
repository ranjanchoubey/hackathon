"""
Configuration file for Prostate WSI Segmentation
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

class Config:
    """Base configuration"""
    
    # Data paths
    DATA_DIR = PROJECT_ROOT / "datasets"
    TRAIN_DIR = DATA_DIR / "Training"
    VAL_DIR = DATA_DIR / "Validation"
    TEST_DIR = DATA_DIR / "Test"
    EXTRA_DIR = DATA_DIR / "Extra"
    
    # Model paths
    MODEL_DIR = PROJECT_ROOT / "models"
    BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
    
    # Output paths
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    # Model hyperparameters
    NUM_CLASSES = 4
    ENCODER_NAME = "resnet34"
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    PATIENCE = 15
    
    # Data processing
    PATCH_SIZE = 256
    STRIDE = 128
    MIN_TISSUE_RATIO = 0.1
    
    # Training
    WEIGHT_CE = 1.0
    WEIGHT_DICE = 1.0
    
    # Device
    DEVICE = "cuda"
    
    # Class definitions
    CLASSES = {0: 'Background', 1: 'Stroma', 2: 'Benign', 3: 'Tumor'}
    CLASS_COLORS = {
        0: [0, 0, 0],        # Black - Background
        1: [0, 0, 255],      # Blue - Stroma  
        2: [0, 255, 0],      # Green - Benign
        3: [255, 255, 0]     # Yellow - Tumor
    }

class TrainingConfig(Config):
    """Training specific configuration"""
    BATCH_SIZE = 8
    EPOCHS = 50
    
class InferenceConfig(Config):
    """Inference specific configuration"""
    BATCH_SIZE = 16
    SAVE_VISUALIZATIONS = True