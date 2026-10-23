"""
Phase 6: Detection Upgrade — Train YOLOv11m and RT-DETR v2-l
on the basketball-players-fy4c2 dataset (ball + player detection).

Dataset: 1140 train / 32 val / 24 test — 9 classes
  Ball, Hoop, Period, Player, Ref, Shot Clock, Team Name, Team Points, Time Remaining

Hardware: RTX 2070 Super (8GB VRAM)
"""

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


def check_gpu():
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — training will be slow")
        return
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")


def train(model_key: str, epochs: int, batch: int, imgsz: int):
    check_gpu()

    data_yaml = Path("data/ball_player_detection/data.yaml")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    # Model presets — adjust batch size for VRAM
    presets = {
        "yolo11m": {
            "weights": "yolo11m.pt",
            "name": "phase6_yolo11m",
            "batch": batch or 16,
        },
        "yolo11l": {
            "weights": "yolo11l.pt",
            "name": "phase6_yolo11l",
            "batch": batch or 8,
        },
        "rtdetr-l": {
            "weights": "rtdetr-l.pt",
            "name": "phase6_rtdetr_l",
            "batch": batch or 4,  # RT-DETR is VRAM-hungry
        },
        "yolov8m": {
            "weights": "yolov8m.pt",
            "name": "phase6_yolov8m",
            "batch": batch or 16,
        },
    }

    if model_key not in presets:
        raise ValueError(f"Unknown model: {model_key}. Choose from {list(presets.keys())}")

    cfg = presets[model_key]
    print(f"\n{'='*60}")
    print(f"Training {model_key}")
    print(f"  Weights: {cfg['weights']}")
    print(f"  Epochs:  {epochs}")
    print(f"  Batch:   {cfg['batch']}")
    print(f"  ImgSz:   {imgsz}")
    print(f"  Data:    {data_yaml}")
    print(f"{'='*60}\n")

    model = YOLO(cfg["weights"])

    model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=cfg["batch"],
        imgsz=imgsz,
        device=0,
        amp=True,
        cache=True,
        workers=4,
        patience=30,
        save_period=10,
        project="runs/detect",
        name=cfg["name"],
        exist_ok=True,
        pretrained=True,
        # Augmentation
        mosaic=1.0,
        mixup=0.1,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
    )

    print(f"\nDone. Best weights: runs/detect/{cfg['name']}/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 detection training")
    parser.add_argument(
        "model",
        choices=["yolo11m", "yolo11l", "rtdetr-l", "yolov8m"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default 100)")
    parser.add_argument("--batch", type=int, default=0, help="Batch size (0 = use preset)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default 640)")
    args = parser.parse_args()
    train(args.model, args.epochs, args.batch, args.imgsz)
