"""
Train YOLOv11m on the NBA Ball Detection dataset (basketball-only).
Source: https://universe.roboflow.com/cps-etiji/nba-ball-detection
Dataset: 1125 train / 109 val / 54 test — 1 class (basketball)
"""

import os
# Disable wandb's separate service process (uses Windows named pipes that
# OneDrive can disconnect, killing training). Runs wandb in-process instead.
os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

# Patch torchvision NMS to fall back to CPU (CUDA kernel missing in this build)
import torchvision.ops as _tvops
_orig_nms = _tvops.nms
def _nms_cpu_fallback(boxes, scores, iou_threshold):
    return _orig_nms(boxes.cpu(), scores.cpu(), float(iou_threshold)).to(boxes.device)
_tvops.nms = _nms_cpu_fallback


def main():
    parser = argparse.ArgumentParser(description="Train ball detection model")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    # GPU info
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")
    else:
        print("WARNING: CUDA not available")

    data_yaml = Path("data/nba_ball_detection/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset not found: {data_yaml}")

    model = YOLO("yolo11m.pt")

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        amp=True,
        cache='disk',
        workers=2,
        project="runs/detect",
        name="ball_yolo11m",
        exist_ok=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        # Single-class, no need for heavy augmentation
        patience=30,
        save_period=10,
        verbose=True,
    )

    print("\nTraining complete!")
    print(f"Best weights: runs/detect/ball_yolo11m/weights/best.pt")


if __name__ == "__main__":
    main()
