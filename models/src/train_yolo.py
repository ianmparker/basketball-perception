"""
Train YOLOv8 on Basketball Players dataset.

Usage:
    python models/src/train_yolo.py --model yolov8n --epochs 50 --batch 16
    python models/src/train_yolo.py --model yolov8s --epochs 100 --batch 8 --resume
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch


def train_model(
    model_size: str = "n",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    resume: bool = False,
    device: str = None,
):
    """
    Train YOLOv8 model on basketball players dataset.
    
    Args:
        model_size: Model size - n (nano), s (small), m (medium), l (large), x (xlarge)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        resume: Resume training from last checkpoint
        device: Device to train on (cuda/cpu), auto-detect if None
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_yaml = project_root / "data" / "basketball-players.yaml"
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    # Auto-detect device
    if device is None:
        device = '0' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'=' * 60}")
    print(f"YOLOV8 TRAINING CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model: yolov8{model_size}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Image Size: {img_size}")
    print(f"Device: {device}")
    print(f"Dataset: {data_yaml}")
    print(f"Resume: {resume}")
    print(f"{'=' * 60}\n")
    
    # Load model
    if resume:
        # Resume from last checkpoint
        model = YOLO('runs/detect/train/weights/last.pt')
        print("📁 Resuming from last checkpoint...")
    else:
        # Load pretrained model
        model = YOLO(f'yolov8{model_size}.pt')
        print(f"📦 Loaded pretrained yolov8{model_size}.pt")
    
    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project='runs/detect',
        name='train',
        exist_ok=True,
        pretrained=not resume,
        patience=10,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
    )
    
    print(f"\n{'=' * 60}")
    print("✓ TRAINING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Results saved to: runs/detect/train")
    print(f"Best model: runs/detect/train/weights/best.pt")
    print(f"Last model: runs/detect/train/weights/last.pt")
    print(f"{'=' * 60}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Basketball Players dataset")
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size: n (nano), s (small), m (medium), l (large), x (xlarge)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size for training'
    )
    parser.add_argument(
        '--img-size', '-i',
        type=int,
        default=640,
        help='Input image size'
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume training from last checkpoint'
    )
    parser.add_argument(
        '--device', '-d',
        type=str,
        default=None,
        help='Device to train on (cuda/cpu). Auto-detect if not specified.'
    )
    
    args = parser.parse_args()
    
    try:
        train_model(
            model_size=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            resume=args.resume,
            device=args.device,
        )
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        raise


if __name__ == "__main__":
    main()
