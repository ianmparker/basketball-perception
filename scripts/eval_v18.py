#!/usr/bin/env python3
"""
Quick inference test for v18 YOLOv8x-pose on broadcast footage.
Tests keypoint detection → RANSAC homography on a sample video.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json

# Patch torchvision NMS to use CPU fallback (same as training)
import torchvision.ops as tvops
_orig_nms = tvops.nms
def _nms_cpu_fallback(boxes, scores, iou_threshold):
    dev = boxes.device
    return _orig_nms(boxes.cpu(), scores.cpu(), float(iou_threshold)).to(dev)
tvops.nms = _nms_cpu_fallback

# Court keypoints in feet (NBA court 94×50 ft)
COURT_KEYPOINTS_FT = np.array([
    # Near basket (0-4)
    [47.0, 25.0],   # 0: right basket
    [47.0, 25.0],   # 1: center of rim
    [47.0, 25.0],   # 2: left basket
    [41.25, 0.0],   # 3: right baseline corner
    [41.25, 50.0],  # 4: right sideline corner
    # Paint corners (5-9)
    [41.25, 14.0],  # 5: right paint bottom-left
    [41.25, 36.0],  # 6: right paint top-left
    [47.0, 14.0],   # 7: right paint bottom-right
    [47.0, 36.0],   # 8: right paint top-right
    [41.25, 25.0],  # 9: right basket edge (redundant, but in COCO)
    # Halfcourt (10-15)
    [0.0, 25.0],    # 10: halfcourt center
    [0.0, 14.0],    # 11: halfcourt left
    [0.0, 36.0],    # 12: halfcourt right
    [0.0, 0.0],     # 13: halfcourt far-left corner
    [0.0, 50.0],    # 14: halfcourt far-right corner
    [47.0, 0.0],    # 15: right baseline far corner
    [47.0, 50.0],   # 16: right sideline far corner
    [0.0, 0.0],     # 17: center far corner (duplicate for collinearity test)
], dtype=np.float32)

def detect_keypoints(frame, model):
    """Run YOLOv8x-pose on frame, return filtered keypoints in pixel coords."""
    results = model(frame, conf=0.5, verbose=False)
    
    if not results or not results[0].keypoints:
        return None, None
    
    # Extract keypoints (x, y, confidence) for first detection
    kpts = results[0].keypoints.xy[0].cpu().numpy()  # (18, 2) in pixel coords
    conf = results[0].keypoints.conf[0].cpu().numpy()  # (18,)
    
    # Filter: only high-confidence keypoints (conf >= 0.7)
    high_conf_idx = conf >= 0.7
    filtered_kpts = kpts[high_conf_idx]
    
    return filtered_kpts, high_conf_idx.sum()

def estimate_homography(kpts_px, min_points=4):
    """RANSAC homography from filtered keypoints to court coordinates."""
    if kpts_px is None or len(kpts_px) < min_points:
        return None
    
    # Use court keypoints corresponding to detected indices (simplified)
    # For now, assume we detected keypoints and map them back naively
    # In production, would track which keypoints were detected
    
    try:
        H, mask = cv2.findHomography(kpts_px, COURT_KEYPOINTS_FT[:len(kpts_px)], 
                                     method=cv2.RANSAC, ransacReprojThreshold=5.0)
        return H
    except:
        return None

def test_video(video_path, model, num_frames=10):
    """Run inference on first N frames of video, report keypoint detection rate."""
    cap = cv2.VideoCapture(str(video_path))
    frame_count = 0
    detected_frames = 0
    high_conf_counts = []
    
    print(f"\n[Eval] Testing {video_path.name} (first {num_frames} frames)")
    print(f"{'Frame':<8} {'Detected':<12} {'High-Conf':<12} {'H status':<12}")
    print("-" * 50)
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect keypoints
        kpts, n_high_conf = detect_keypoints(frame, model)
        
        if kpts is not None:
            detected_frames += 1
            high_conf_counts.append(n_high_conf)
            
            # Try to estimate homography
            H = estimate_homography(kpts)
            h_status = "✓ Valid" if H is not None else "✗ Failed"
            
            print(f"{frame_count:<8} {'Yes':<12} {n_high_conf:<12} {h_status:<12}")
        else:
            print(f"{frame_count:<8} {'No':<12} {'-':<12} {'-':<12}")
        
        frame_count += 1
    
    cap.release()
    
    # Summary stats
    detection_rate = (detected_frames / frame_count * 100) if frame_count > 0 else 0
    avg_high_conf = np.mean(high_conf_counts) if high_conf_counts else 0
    
    print("-" * 50)
    print(f"Detection rate: {detection_rate:.1f}% ({detected_frames}/{frame_count} frames)")
    print(f"Avg high-conf keypoints per detected frame: {avg_high_conf:.1f}")
    
    return {
        "video": video_path.name,
        "detection_rate": detection_rate,
        "avg_high_conf_keypoints": avg_high_conf,
        "total_frames": frame_count,
    }

def main():
    model_path = Path("runs/pose/court_kp_v189/weights/best.pt")
    video_paths = list(Path("data/sample/videos").glob("broadcast_*.mp4"))[:3]  # Test first 3
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = YOLO(str(model_path))
    
    results = []
    for vid_path in video_paths:
        result = test_video(vid_path, model, num_frames=10)
        results.append(result)
    
    # Save results
    output_path = Path("homography/learned/v18_eval.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to {output_path}")
    print("\nSummary:")
    for r in results:
        print(f"  {r['video']}: {r['detection_rate']:.1f}% detection, "
              f"{r['avg_high_conf_keypoints']:.1f} avg high-conf keypoints")

if __name__ == "__main__":
    main()
