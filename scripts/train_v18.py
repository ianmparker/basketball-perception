"""
YOLOv8x-pose v18 training script.

Patches torchvision.ops.nms with a CPU fallback so AMP (fp16) training works
on cu128 where torchvision has no CUDA NMS kernel.
"""
import os
import torch
import torchvision.ops as tvops

# ------------------------------------------------------------------
# CPU fallback for torchvision NMS.
# Training forward/backward does NOT use NMS — only val inference does.
# Moving boxes/scores to CPU for NMS and back to GPU costs <1ms per batch.
# ------------------------------------------------------------------
_orig_nms = tvops.nms


def _nms_cpu_fallback(boxes, scores, iou_threshold):
    dev = boxes.device
    return _orig_nms(boxes.cpu(), scores.cpu(), float(iou_threshold)).to(dev)


# Patch only tvops.nms — do NOT also patch torch.ops.torchvision.nms
# (tvops.nms calls torch.ops.torchvision.nms internally; double-patch causes recursion)
tvops.nms = _nms_cpu_fallback

# ------------------------------------------------------------------
# W&B config
# ------------------------------------------------------------------
os.environ.setdefault("WANDB_PROJECT", "ai-basketball-coach")
os.environ.setdefault("WANDB_NAME", "court_kp_v18")

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
from ultralytics import YOLO  # noqa: E402 (import after patch)

if __name__ == "__main__":
    # Use best.pt from previous training run (327 epochs, pose_loss=0.072)
    # best.pt contains the best model checkpoint — inference-ready
    best_pt = "runs/pose/court_kp_v189/weights/best.pt"
    
    if os.path.exists(best_pt):
        print(f"[v18 train] Loading best model from: {best_pt}")
        print("[v18 train] (327 epochs already completed with excellent convergence)")
        print("[v18 train] Starting new training with this checkpoint...")
        model = YOLO(best_pt)
    else:
        print("[v18 train] best.pt not found; starting from pretrained yolov8x-pose")
        model = YOLO("yolov8x-pose.pt")
    
    model.train(
        data="data/court_keypoints_yolo/data.yaml",
        epochs=500,
        imgsz=640,
        batch=16,
        fliplr=0.0,
        amp=True,        # fp16 — safe now that NMS falls back to CPU
        cache=True,      # pre-load images to RAM; eliminates I/O variance
        workers=4,
        project="runs/pose",
        name="court_kp_v18_resume",  # new run directory to avoid conflicts
        device=0,
        patience=80,     # stop early if no improvement for 80 epochs
    )
