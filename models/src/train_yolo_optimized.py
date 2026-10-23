"""
Optimized YOLOv8 Training Script for NBA Unified Dataset

Hardware: RTX 2070 Super (8GB VRAM)
Dataset: 7,239 images (17 classes)
Target: 200 epochs with FP16 mixed precision
Estimated Time: 12-14 hours

Key Optimizations:
- FP16 mixed precision (2x speedup)
- Image caching (faster data loading)
- Multi-worker data loading
- Optimal batch size for 8GB VRAM
- Weights & Biases tracking
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch
from datetime import datetime


def check_gpu():
    """Check GPU availability and print info."""
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, training will be slow on CPU!")
        return False
    
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print("\n" + "="*60)
    print("🎮 GPU INFORMATION")
    print("="*60)
    print(f"Device: {device_name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print("="*60 + "\n")
    
    return True


def train_model(args):
    """Train YOLOv8 model with optimized settings."""
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print(f"📦 Loading YOLOv8{args.model} model...")
    model = YOLO(f'yolov8{args.model}.pt')
    
    # Training configuration
    print("\n" + "="*60)
    print("🏋️  TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model: YOLOv8{args.model}")
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Workers: {args.workers}")
    print(f"Device: {'GPU (CUDA)' if has_gpu else 'CPU'}")
    print(f"Mixed Precision (FP16): {args.amp}")
    print(f"Cache Images: {args.cache}")
    print(f"Project: {args.output}")
    print("="*60 + "\n")
    
    # Confirm before starting
    if not args.yes:
        response = input("⏳ Start training? This will take 12-14 hours. (y/n): ")
        if response.lower() != 'y':
            print("❌ Training cancelled")
            return
    
    print(f"🚀 Starting training at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")
    
    # Train the model
    results = model.train(
        # Data
        data=args.data,
        
        # Training duration
        epochs=args.epochs,
        
        # Batch and image size
        batch=args.batch,
        imgsz=args.imgsz,
        
        # Optimization
        amp=args.amp,              # FP16 mixed precision (2x speedup)
        cache=args.cache,          # Cache images for faster loading
        workers=args.workers,      # Multi-threaded data loading
        
        # Data augmentation (aggressive for better generalization)
        hsv_h=0.015,              # HSV-Hue augmentation
        hsv_s=0.7,                # HSV-Saturation augmentation
        hsv_v=0.4,                # HSV-Value augmentation
        degrees=0.0,              # Image rotation (+/- deg)
        translate=0.1,            # Image translation (+/- fraction)
        scale=0.5,                # Image scale (+/- gain)
        shear=0.0,                # Image shear (+/- deg)
        perspective=0.0,          # Image perspective (+/- fraction)
        flipud=0.0,               # Flip up-down (probability)
        fliplr=0.5,               # Flip left-right (probability)
        mosaic=1.0,               # Mosaic augmentation (probability)
        mixup=0.1,                # MixUp augmentation (probability)
        copy_paste=0.0,           # Copy-paste augmentation
        
        # Hyperparameters
        lr0=0.01,                 # Initial learning rate
        lrf=0.01,                 # Final learning rate (lr0 * lrf)
        momentum=0.937,           # SGD momentum
        weight_decay=0.0005,      # Optimizer weight decay
        warmup_epochs=3.0,        # Warmup epochs
        warmup_momentum=0.8,      # Warmup initial momentum
        warmup_bias_lr=0.1,       # Warmup initial bias lr
        
        # Loss weights
        box=7.5,                  # Box loss gain
        cls=0.5,                  # Class loss gain
        dfl=1.5,                  # DFL loss gain
        
        # Validation
        val=True,                 # Validate during training
        patience=50,              # Early stopping patience (epochs)
        save=True,                # Save checkpoints
        save_period=10,           # Save checkpoint every N epochs
        
        # Output
        project=str(output_dir),
        name=args.name,
        exist_ok=True,
        
        # Tracking
        plots=True,               # Save plots (loss, metrics, etc.)
        verbose=True,             # Verbose output
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nBest model: {output_dir / args.name / 'weights' / 'best.pt'}")
    print(f"Last model: {output_dir / args.name / 'weights' / 'last.pt'}")
    print(f"Results: {output_dir / args.name}")
    
    # Print final metrics
    if results.results_dict:
        print("\n📊 FINAL METRICS:")
        metrics = results.results_dict
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 on NBA Unified Dataset')
    
    # Model configuration
    parser.add_argument('--model', type=str, default='m', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--data', type=str, default='data/nba-unified.yaml',
                        help='Path to dataset YAML config')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (32 optimal for RTX 2070 Super 8GB)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Input image size')
    
    # Performance optimization
    parser.add_argument('--amp', type=bool, default=True,
                        help='Use automatic mixed precision (FP16)')
    parser.add_argument('--cache', type=bool, default=True,
                        help='Cache images in RAM for faster training')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of worker threads for data loading')
    
    # Output configuration
    parser.add_argument('--output', type=str, default='models/runs/train',
                        help='Output directory for training results')
    parser.add_argument('--name', type=str, default='nba-unified-v1',
                        help='Training run name')
    
    # Misc
    parser.add_argument('--yes', '-y', action='store_true',
                        help='Skip confirmation prompt')
    
    args = parser.parse_args()
    
    # Validate data file exists
    if not Path(args.data).exists():
        print(f"❌ Error: Data config not found: {args.data}")
        print("\nRun dataset merge first:")
        print("  python data/ingestion/merge_datasets.py")
        return
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()
