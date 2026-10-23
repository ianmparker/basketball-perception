#!/usr/bin/env python3
"""
Reprojection Error Evaluation for v18 YOLOv8x-pose model.

For each image in the validation set:
  1. Run model inference → predicted keypoints (pixel coords + confidence)
  2. Load GT YOLO label → GT visible keypoints (vis=2, pixel coords)
  3. Build homography H from predicted keypoints → court coords (RANSAC)
  4. Evaluate H: project GT pixel coords → court coords, measure error in feet
  5. Report mean / median / p90 reprojection error across all valid frames

Beats v14-v17 baseline of 9.5 ft if we see < 6 ft median.

Usage:
    conda activate basketball-perception
    python scripts/reproj_eval_v18.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# Patch torchvision NMS to use CPU fallback
import torchvision.ops as tvops
_orig_nms = tvops.nms
def _nms_cpu_fallback(boxes, scores, iou_threshold):
    dev = boxes.device
    return _orig_nms(boxes.cpu(), scores.cpu(), float(iou_threshold)).to(dev)
tvops.nms = _nms_cpu_fallback

# Authoritative court keypoints (feet) — indexed by COCO export position.
# From homography/learned/model.py (COURT_KEYPOINTS_FT).
COURT_KPS_FT = np.array([
    [0.0,  50.0],  # COCO pos 0 → near_right_corner
    [0.0,  47.0],  # COCO pos 1 → near_3pt_right_at_baseline
    [0.0,  33.0],  # COCO pos 2 → near_right_paint_corner
    [0.0,  17.0],  # COCO pos 3 → near_left_paint_corner
    [0.0,   3.0],  # COCO pos 4 → near_3pt_left_at_baseline
    [0.0,   0.0],  # COCO pos 5 → near_left_corner
    [47.0, 50.0],  # COCO pos 6 → halfcourt_right_sideline
    [47.0, 19.0],  # COCO pos 7 → center_circle_left_arc
    [75.0, 17.0],  # COCO pos 8 → far_left_FT_T
    [75.0, 33.0],  # COCO pos 9 → far_right_FT_T
    [94.0, 50.0],  # COCO pos 10 → far_right_corner
    [94.0, 47.0],  # COCO pos 11 → far_3pt_right_at_baseline
    [94.0, 33.0],  # COCO pos 12 → far_right_paint_corner
    [94.0, 17.0],  # COCO pos 13 → far_left_paint_corner
    [94.0,  3.0],  # COCO pos 14 → far_3pt_left_at_baseline
    [47.0,  0.0],  # COCO pos 15 → halfcourt_left_sideline
    [47.0, 31.0],  # COCO pos 16 → center_circle_right_arc
    [47.0, 25.0],  # COCO pos 17 → center_circle_center
], dtype=np.float32)

NUM_KPS = 18
CONF_THRESHOLD = 0.3     # minimum confidence to use a predicted keypoint in H
MIN_PTS_FOR_H = 4        # minimum points needed to solve homography
RANSAC_THRESH_FT = 5.0   # RANSAC inlier threshold (feet, in court space)
VAL_DIR = Path("data/court_keypoints/valid")
MODEL_PATH = Path("runs/pose/court_kp_v18_resume/weights/best.pt")


def parse_yolo_label(label_path: Path, img_w: int, img_h: int):
    """
    Parse a YOLO-pose label file.
    Returns (kp_px, vis) where:
      kp_px: (18, 2) float32 pixel coords
      vis:   (18,)   int (0=absent, 1=occluded, 2=visible)
    """
    with open(label_path) as f:
        line = f.readline().strip().split()
    # Format: class cx cy w h  [kp_x kp_y kp_vis] × 18
    vals = list(map(float, line))
    kp_data = vals[5:]  # skip class + bbox
    kp_px = np.zeros((NUM_KPS, 2), dtype=np.float32)
    vis = np.zeros(NUM_KPS, dtype=int)
    for i in range(NUM_KPS):
        nx, ny, v = kp_data[3*i], kp_data[3*i+1], int(kp_data[3*i+2])
        kp_px[i] = [nx * img_w, ny * img_h]
        vis[i] = v
    return kp_px, vis


def apply_homography(H, pts):
    """
    Apply homography H (3x3) to pts (N,2) → (N,2).
    """
    pts_h = np.concatenate([pts, np.ones((len(pts), 1), dtype=np.float32)], axis=1)
    out = (H @ pts_h.T).T  # (N, 3)
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-6, 1e-6 * np.sign(w + 1e-12), w)
    return out[:, :2] / w


def eval_image(img_path: Path, model):
    """
    Run evaluation on a single validation image.
    Returns dict with reprojection errors (ft) or None if skipped.
    """
    label_path = img_path.with_suffix(".txt")
    if not label_path.exists():
        return None

    # Load image to get dimensions
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    # ── GT keypoints ──────────────────────────────────────────────────────
    gt_kp_px, gt_vis = parse_yolo_label(label_path, w, h)

    # Visible GT: vis==2, non-zero coords, within image bounds
    gt_mask = (
        (gt_vis == 2) &
        (gt_kp_px[:, 0] > 1) & (gt_kp_px[:, 0] < w - 1) &
        (gt_kp_px[:, 1] > 1) & (gt_kp_px[:, 1] < h - 1)
    )
    if gt_mask.sum() < MIN_PTS_FOR_H:
        return {"skipped": "too_few_gt_kps", "n_gt": int(gt_mask.sum())}

    # ── Model predictions ─────────────────────────────────────────────────
    results = model(img, conf=0.3, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return {"skipped": "no_detection"}

    pred_kp_px = results[0].keypoints.xy[0].cpu().numpy()    # (18, 2) pixel coords
    pred_conf  = results[0].keypoints.conf[0].cpu().numpy()  # (18,)

    # Filter to high-confidence, non-zero predicted keypoints
    pred_mask = (
        (pred_conf >= CONF_THRESHOLD) &
        (pred_kp_px[:, 0] > 1) & (pred_kp_px[:, 0] < w - 1) &
        (pred_kp_px[:, 1] > 1) & (pred_kp_px[:, 1] < h - 1)
    )
    if pred_mask.sum() < MIN_PTS_FOR_H:
        return {"skipped": "too_few_pred_kps", "n_pred": int(pred_mask.sum())}

    src_px    = pred_kp_px[pred_mask].astype(np.float32)       # predicted pixel coords
    dst_court = COURT_KPS_FT[pred_mask].astype(np.float32)     # corresponding court coords

    # ── Collinearity check on court coords ────────────────────────────────
    # If all predicted court coords are collinear (rank < 2), H is undefined.
    centered = dst_court - dst_court.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered)
    # Ratio of 2nd to 1st singular value: < 0.01 → nearly collinear
    if sv[0] < 1e-6 or (sv[1] / sv[0]) < 0.01:
        return {"skipped": "collinear_court_pts",
                "n_pred": int(pred_mask.sum()),
                "sv_ratio": float(sv[1] / (sv[0] + 1e-9))}

    # ── Build H from predictions (plain DLT, no RANSAC) ──────────────────
    # Confidence-filtered points are already reliable.
    # Image-space baseline collinearity causes RANSAC minimal-set degeneracy.
    H, _ = cv2.findHomography(src_px, dst_court, method=0)
    if H is None:
        return {"skipped": "dlt_failed"}

    # Sanity: reject wildly degenerate H
    h22 = H[:2, :2]
    det_h = abs(np.linalg.det(h22))
    if det_h < 1e-6 or det_h > 1e10:
        return {"skipped": "degenerate_H", "det": float(det_h)}

    n_inliers = int(pred_mask.sum())  # all conf-filtered points used (DLT, no RANSAC)

    # ── Evaluate H on GT visible keypoints ───────────────────────────────
    gt_indices = np.where(gt_mask)[0]
    gt_px_eval    = gt_kp_px[gt_indices].astype(np.float32)
    gt_court_eval = COURT_KPS_FT[gt_indices].astype(np.float32)

    court_pred = apply_homography(H, gt_px_eval)
    errors_ft  = np.linalg.norm(court_pred - gt_court_eval, axis=1)  # (N,)

    return {
        "image": img_path.name,
        "n_pred_kps": int(pred_mask.sum()),
        "n_gt_kps": int(gt_mask.sum()),
        "n_ransac_inliers": n_inliers,
        "mean_error_ft": float(errors_ft.mean()),
        "median_error_ft": float(np.median(errors_ft)),
        "max_error_ft": float(errors_ft.max()),
        "per_kp_errors": errors_ft.tolist(),
    }


def main():
    print(f"Loading model: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found — {MODEL_PATH}")
        sys.exit(1)

    model = YOLO(str(MODEL_PATH))

    # Collect all validation images
    img_paths = sorted(VAL_DIR.glob("*.jpg"))
    print(f"Evaluating {len(img_paths)} validation images...\n")

    results = []
    skipped = []
    all_errors = []

    for i, img_path in enumerate(img_paths):
        r = eval_image(img_path, model)
        if r is None:
            skipped.append({"image": img_path.name, "reason": "io_error"})
            continue
        if "skipped" in r:
            skipped.append({"image": img_path.name, **r})
            continue

        results.append(r)
        all_errors.extend(r["per_kp_errors"])

        if (i + 1) % 10 == 0 or (i + 1) == len(img_paths):
            print(f"  [{i+1:3d}/{len(img_paths)}] "
                  f"running mean={np.mean(all_errors):.2f} ft  "
                  f"median={np.median(all_errors):.2f} ft  "
                  f"(processed {len(results)}, skipped {len(skipped)})")

    # ── Final summary ──────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("v18 REPROJECTION ERROR SUMMARY (vs v14-v17 baseline: 9.5 ft)")
    print("="*60)

    if all_errors:
        all_errors = np.array(all_errors)
        mean_e   = float(np.mean(all_errors))
        median_e = float(np.median(all_errors))
        p90_e    = float(np.percentile(all_errors, 90))
        p95_e    = float(np.percentile(all_errors, 95))

        print(f"  Valid frames:    {len(results)} / {len(img_paths)} ({len(skipped)} skipped)")
        print(f"  Total keypoints: {len(all_errors)}")
        print(f"  Mean error:      {mean_e:.2f} ft")
        print(f"  Median error:    {median_e:.2f} ft")
        print(f"  90th pctile:     {p90_e:.2f} ft")
        print(f"  95th pctile:     {p95_e:.2f} ft")
        print(f"  Max error:       {float(all_errors.max()):.2f} ft")
        print()

        baseline = 9.5
        if median_e < baseline:
            print(f"  ✓ BEAT BASELINE: {median_e:.2f} ft < {baseline} ft (improvement: {baseline - median_e:.2f} ft)")
        else:
            print(f"  ✗ Did not beat baseline: {median_e:.2f} ft >= {baseline} ft")
    else:
        mean_e = median_e = p90_e = p95_e = None
        print("  No valid frames processed.")

    print()
    skip_reasons = {}
    for s in skipped:
        r = s.get("skipped", s.get("reason", "unknown"))
        skip_reasons[r] = skip_reasons.get(r, 0) + 1
    if skip_reasons:
        print(f"  Skip reasons: {skip_reasons}")

    # ── Save results ───────────────────────────────────────────────────────
    output = {
        "model": str(MODEL_PATH),
        "n_images": len(img_paths),
        "n_valid": len(results),
        "n_skipped": len(skipped),
        "mean_error_ft": mean_e,
        "median_error_ft": median_e,
        "p90_error_ft": p90_e,
        "p95_error_ft": p95_e,
        "baseline_ft": 9.5,
        "beat_baseline": bool(median_e is not None and median_e < 9.5),
        "skip_breakdown": skip_reasons,
        "per_image_results": results,
        "skipped_images": skipped,
    }
    out_path = Path("homography/learned/v18_reproj_eval.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
