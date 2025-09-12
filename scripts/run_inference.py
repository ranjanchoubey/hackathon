#!/usr/bin/env python3
"""
Standalone inference script
Usage: python scripts/run_inference.py --input path/to/test/images --output path/to/predictions
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import InferenceConfig
from inference import generate_test_predictions
import torch

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
    
    # Run inference
    results = generate_test_predictions(
        model_path=model_path,
        test_dir=args.input,
        output_dir=args.output,
        device=device
    )
    
    print(f"✅ Generated {len(results)} predictions!")

if __name__ == "__main__":
    main()