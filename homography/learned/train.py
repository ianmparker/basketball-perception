"""Training Script — Direct Homography Regression (v14+)

Trains CourtKeypointHomography to predict a 3x3 homography matrix directly
from frame features, supervised by keypoint reprojection error (no DLT needed).

Loss:
  PRIMARY:   Keypoint reprojection -- apply predicted H to GT visible keypoint
             image coords, penalise L2 distance to GT court coords (feet).
  AUXILIARY: Heatmap MSE -- keeps backbone features landmark-aware.

Usage:
    python homography/learned/train.py \\
        --train-dir data/court_keypoints/train \\
        --val-dir   data/court_keypoints/valid
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional

# Load .env so WANDB_API_KEY and ROBOFLOW_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import wandb

# resolve imports when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from homography.learned.model import CourtKeypointHomography, COURT_KEYPOINTS_FT, apply_homography
from homography.learned.prepare_data import CourtKeypointDataset

# Court diagonal in feet — used to normalise reprojection loss to [0, 1] scale
_COURT_DIAG_FT: float = (94.0 ** 2 + 50.0 ** 2) ** 0.5  # ≈ 107 ft


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class HomographyLoss(nn.Module):
    """
    Keypoint reprojection loss via the predicted H (no DLT required).

    PRIMARY: Apply H_pred to GT visible keypoint image coords → compare to
             known court coords (COURT_KEYPOINTS_FT). Loss in feet.
             Gradient flows through H_pred → HomographyRegressionHead → backbone.

    AUXILIARY: Heatmap MSE to keep backbone features court-landmark-aware.
    """

    def __init__(self, heatmap_weight: float = 0.1):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        heatmaps_pred: torch.Tensor,  # (B, N_kp, Hm, Wm)
        heatmaps_gt:   torch.Tensor,  # (B, N_kp, Hm, Wm)
        mask:          torch.Tensor,  # (B, N_kp) bool
        kps_xy:        torch.Tensor,  # (B, N_kp, 2) pixel coords in INPUT space
        H_pred:        torch.Tensor,  # (B, 3, 3)
        image_hw:      tuple,         # (H_img, W_img)
    ) -> tuple:
        B, N, Hm, Wm = heatmaps_gt.shape
        device = H_pred.device

        if heatmaps_pred.shape[-2:] != heatmaps_gt.shape[-2:]:
            heatmaps_pred = F.interpolate(
                heatmaps_pred, size=(Hm, Wm), mode="bilinear", align_corners=False
            )

        mask_f    = mask.float()                            # (B, N)
        n_visible = mask_f.sum(dim=1).clamp(min=1)         # (B,)

        # --- Auxiliary: heatmap MSE (masked to visible keypoints) ---
        pixel_mse    = self.mse(heatmaps_pred, heatmaps_gt).mean(dim=(-2, -1))  # (B, N)
        heatmap_loss = (pixel_mse * mask_f).sum(dim=1) / n_visible
        heatmap_loss = heatmap_loss.mean()

        # --- Primary: keypoint reprojection via predicted H ---
        H_img, W_img = image_hw
        # Normalise GT keypoint pixel coords to [0, 1]
        kps_norm = kps_xy.float().clone()
        kps_norm[..., 0] /= W_img
        kps_norm[..., 1] /= H_img   # (B, N, 2) in [0, 1]

        court_gt   = COURT_KEYPOINTS_FT.to(device).unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)
        court_pred = apply_homography(H_pred, kps_norm)     # (B, N, 2) in feet

        err = (court_pred - court_gt).norm(dim=-1)          # (B, N) in feet
        err = err.clamp(max=_COURT_DIAG_FT)                 # cap outlier frames

        reproj_per_img = (err * mask_f).sum(dim=1) / n_visible  # (B,) in feet
        reproj_loss    = reproj_per_img.mean()

        total = reproj_loss + self.heatmap_weight * heatmap_loss

        metrics = {
            "reproj_ft":    reproj_loss.item(),
            "heatmap_loss": heatmap_loss.item(),
        }
        return total, metrics


# ---------------------------------------------------------------------------
# Train / validate one epoch
# ---------------------------------------------------------------------------

def run_epoch(
    model, loader, criterion, optimizer, scaler, device, is_train: bool, epoch: int
) -> dict:
    model.train() if is_train else model.eval()

    totals = {"loss": 0, "reproj_ft": 0, "heatmap_loss": 0}
    n = 0
    tag = "Train" if is_train else "Val"

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"Epoch {epoch} [{tag}]", leave=False)
        for batch in pbar:
            images   = batch["image"].to(device, non_blocking=True)
            heatmaps = batch["heatmaps"].to(device, non_blocking=True)
            mask     = batch["mask"].to(device, non_blocking=True)
            kps_xy   = batch["kps_xy"].to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=device.type == "cuda"):
                H_pred, _kp_coords, _kp_conf, heatmaps_pred = model(images)
                loss, metrics = criterion(
                    heatmaps_pred, heatmaps, mask, kps_xy, H_pred,
                    (images.shape[2], images.shape[3])
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            loss_val = loss.item()
            totals["loss"]         += loss_val if math.isfinite(loss_val) else 0
            totals["reproj_ft"]    += metrics.get("reproj_ft", 0)
            totals["heatmap_loss"] += metrics.get("heatmap_loss", 0)
            n += 1
            pbar.set_postfix(reproj=f"{metrics['reproj_ft']:.2f}ft")

    return {k: v / max(n, 1) for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.wandb:
        wandb.init(project="ai-basketball-coach", name=f"homography-heatmap-{args.name}", config=vars(args))

    # Datasets
    train_ds = CourtKeypointDataset(args.train_dir, augment=True)
    if args.extra_train_dir:
        extra_ds = CourtKeypointDataset(args.extra_train_dir, augment=True)
        train_ds = ConcatDataset([train_ds, extra_ds])
        print(f"Extra train: +{len(extra_ds)} samples (total {len(train_ds)})")
    val_ds   = CourtKeypointDataset(args.val_dir,   augment=False)
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    _persist = args.workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, persistent_workers=_persist)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True, persistent_workers=_persist)

    # Model
    backbone_type = getattr(args, "backbone", "hrnet")
    model = CourtKeypointHomography(
        pretrained=True,
        conf_threshold=args.conf_threshold,
        coco_pose_ckpt=getattr(args, "backbone_ckpt", None),
        backbone_type=backbone_type,
    ).to(device)

    # DINOv2: backbone already frozen in _dinov2_backbone(); disable heatmap aux
    if backbone_type == "dinov2":
        args.freeze_backbone = True
        args.heatmap_weight  = 0.0
        print("[INFO] DINOv2 mode: backbone frozen, heatmap_weight=0")

    # Optionally freeze backbone so only the regression head trains.
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
        frozen_mb = sum(p.numel() for p in model.backbone.parameters()) * 4 / 1e6
        print(f"[INFO] Backbone frozen ({frozen_mb:.0f} MB frozen, head-only training)")
    else:
        # Try to enable gradient checkpointing to reduce activation memory
        # (~6x reduction). Not all backbone variants support it.
        try:
            if hasattr(model.backbone, "set_grad_checkpointing"):
                model.backbone.set_grad_checkpointing(True)
                print("[INFO] Gradient checkpointing enabled on backbone")
        except (AssertionError, NotImplementedError):
            print("[INFO] Gradient checkpointing not supported by this backbone; using batch-size to control memory")

    if args.compile and device.type == "cuda":
        print("Compiling model with torch.compile (max-autotune)...")
        model = torch.compile(model, mode="max-autotune")

    # Three-tier LR: backbone (pretrained) < heatmap head (auxiliary) < homo head (primary)
    if args.freeze_backbone:
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        bb_lr   = args.backbone_lr if args.backbone_lr is not None else args.lr * 0.05
        head_lr = args.lr
        param_groups = [
            {"params": model.backbone.parameters(),  "lr": bb_lr},
            {"params": model.homo_head.parameters(), "lr": head_lr},
        ]
        if model.heatmap_head is not None:
            param_groups.append({"params": model.heatmap_head.parameters(), "lr": head_lr})
        optimizer = optim.AdamW(param_groups, weight_decay=args.weight_decay)
        print(f"[INFO] LR: backbone={bb_lr:.1e}, heads={head_lr:.1e}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    criterion = HomographyLoss(heatmap_weight=args.heatmap_weight)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_reproj   = float("inf")
    patience_count = 0

    for epoch in range(1, args.epochs + 1):
        train_m = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True,  epoch)
        val_m   = run_epoch(model, val_loader,   criterion, None,      scaler, device, False, epoch)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train_reproj={train_m['reproj_ft']:.2f}ft  "
            f"val_reproj={val_m['reproj_ft']:.2f}ft  "
            f"lr={lr:.2e}"
        )

        if args.wandb:
            wandb.log({"epoch": epoch, "lr": lr,
                       "train/loss":      train_m["loss"],
                       "train/reproj_ft": train_m["reproj_ft"],
                       "train/heatmap":   train_m["heatmap_loss"],
                       "val/loss":        val_m["loss"],
                       "val/reproj_ft":   val_m["reproj_ft"],
                       "val/heatmap":     val_m["heatmap_loss"]})

        # Track best reproj for reporting
        val_reproj = val_m["reproj_ft"]
        if math.isfinite(val_reproj) and val_reproj < best_reproj:
            best_reproj = val_reproj

        # Early stopping on total val loss
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            patience_count = 0
            ckpt = output_dir / "best.pt"
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": best_val_loss, "reproj_ft": best_reproj,
                        "args": vars(args)}, ckpt)
            print(f"  [best] checkpoint saved (val_loss={best_val_loss:.5f})")
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        if math.isfinite(best_reproj) and best_reproj < args.target_reproj:
            print(f"Target reached: {best_reproj:.3f} ft < {args.target_reproj} ft")
            break

    print(f"\nDone. Best val loss: {best_val_loss:.5f}  Best val reproj: {best_reproj:.3f} ft")
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train court homography regression network")

    p.add_argument("--train-dir", required=True)
    p.add_argument("--extra-train-dir", default=None,
                   help="Optional additional training split (e.g. test/) merged with --train-dir")
    p.add_argument("--val-dir",   required=True)
    p.add_argument("--output",    default="homography/learned/checkpoints")

    p.add_argument("--epochs",         type=int,   default=150)
    p.add_argument("--batch-size",     type=int,   default=16)
    p.add_argument("--lr",             type=float, default=1e-3,
                   help="LR for head params (homo_head, heatmap_head)")
    p.add_argument("--backbone-lr",    type=float, default=None,
                   help="LR for backbone (default: lr * 0.05)")
    p.add_argument("--weight-decay",   type=float, default=1e-4)
    p.add_argument("--workers",        type=int,   default=4)
    p.add_argument("--patience",       type=int,   default=30)
    p.add_argument("--target-reproj",  type=float, default=0.7)
    p.add_argument("--heatmap-weight", type=float, default=0.1,
                   help="Weight on auxiliary heatmap MSE loss (default 0.1)")
    p.add_argument("--freeze-backbone", action="store_true", default=False)
    p.add_argument("--backbone-ckpt",  default=None)
    p.add_argument("--backbone",       default="hrnet", choices=["hrnet", "dinov2"],
                   help="Backbone architecture: hrnet (default) or dinov2")
    p.add_argument("--conf-threshold", type=float, default=0.3)  # kept for compat

    p.add_argument("--compile",    action="store_true", default=True)
    p.add_argument("--no-compile", dest="compile", action="store_false")
    p.add_argument("--wandb",  action="store_true")
    p.add_argument("--name",   default="run1")

    args = p.parse_args()
    train(args)

