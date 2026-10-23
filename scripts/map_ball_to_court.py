"""
Map ball pixel positions to court coordinates using homography.

Two modes:
  --calibration  Legacy: manual homography calibration JSON (single static H)
  --model        Learned: v18 YOLOv8x-pose model (per-frame H from court keypoints)

Usage (learned):
  python scripts/map_ball_to_court.py \
      --ball-positions data/results/broadcast_01_ball_positions.json \
      --model runs/pose/court_kp_v18_resume/weights/best.pt

Usage (legacy):
  python scripts/map_ball_to_court.py \
      --ball-positions data/results/broadcast_01_ball_positions.json \
      --calibration homography/manual_calibration.json
"""

import json
import numpy as np
import argparse
import cv2
from pathlib import Path

# ── v18 learned homography constants ─────────────────────────────────────────
# Authoritative court keypoints (ft), COCO export position order.
# Must match homography/learned/model.py COURT_KEYPOINTS_FT.
_COURT_KPS_FT = np.array([
    [0.0,  50.0],  # kp 0  near_right_corner
    [0.0,  47.0],  # kp 1  near_3pt_right_at_baseline
    [0.0,  33.0],  # kp 2  near_right_paint_corner
    [0.0,  17.0],  # kp 3  near_left_paint_corner
    [0.0,   3.0],  # kp 4  near_3pt_left_at_baseline
    [0.0,   0.0],  # kp 5  near_left_corner
    [47.0, 50.0],  # kp 6  halfcourt_right_sideline
    [47.0, 19.0],  # kp 7  center_circle_left_arc
    [75.0, 17.0],  # kp 8  far_left_FT_T
    [75.0, 33.0],  # kp 9  far_right_FT_T
    [94.0, 50.0],  # kp10  far_right_corner
    [94.0, 47.0],  # kp11  far_3pt_right_at_baseline
    [94.0, 33.0],  # kp12  far_right_paint_corner
    [94.0, 17.0],  # kp13  far_left_paint_corner
    [94.0,  3.0],  # kp14  far_3pt_left_at_baseline
    [47.0,  0.0],  # kp15  halfcourt_left_sideline
    [47.0, 31.0],  # kp16  center_circle_right_arc
    [47.0, 25.0],  # kp17  center_circle_center
], dtype=np.float32)

_CONF_THRESHOLD = 0.3
_MIN_PTS_FOR_H  = 4


def load_homography_matrix(calibration_file):
    """Load homography matrix from calibration JSON."""
    with open(calibration_file, 'r') as f:
        calibration = json.load(f)
    
    # Extract homography matrix
    H = np.array(calibration['homography_matrix'])
    
    print(f"Loaded homography from: {calibration_file}")
    print(f"Video: {calibration.get('video_path', 'N/A')}")
    print(f"Calibration points: {len(calibration.get('image_points', []))}")
    
    return H, calibration


def pixel_to_court(pixel_point, H):
    """
    Transform pixel coordinates to court coordinates using homography.
    
    Args:
        pixel_point: [x, y] in pixel coordinates
        H: 3x3 homography matrix
    
    Returns:
        [x, y] in court coordinates (feet)
    """
    # Convert to homogeneous coordinates
    px, py = pixel_point
    pixel_homogeneous = np.array([px, py, 1.0])
    
    # Apply homography transformation
    court_homogeneous = H @ pixel_homogeneous
    
    # Convert back from homogeneous coordinates
    court_x = court_homogeneous[0] / court_homogeneous[2]
    court_y = court_homogeneous[1] / court_homogeneous[2]
    
    return [float(court_x), float(court_y)]


def map_ball_positions_to_court(ball_positions_file, calibration_file, output_file=None):
    """
    Map all detected ball positions from pixel to court coordinates.
    
    Args:
        ball_positions_file: JSON file with ball detections
        calibration_file: JSON file with homography calibration
        output_file: Output JSON file (default: input_name_court_coords.json)
    
    Returns:
        Dictionary with court coordinates for each frame
    """
    # Load ball positions
    with open(ball_positions_file, 'r') as f:
        ball_data = json.load(f)
    
    # Load homography
    H, calibration = load_homography_matrix(calibration_file)
    
    # Process each frame
    ball_court_positions = {}
    frames_mapped = 0
    
    for frame_idx_str, detections in ball_data['ball_positions'].items():
        frame_idx = int(frame_idx_str)
        
        if len(detections) == 0:
            ball_court_positions[frame_idx] = []
            continue
        
        # Map each detection to court coordinates
        court_detections = []
        for detection in detections:
            pixel_center = detection['center']
            
            # Transform to court coordinates
            court_coords = pixel_to_court(pixel_center, H)
            
            court_detections.append({
                'pixel_coords': pixel_center,
                'court_coords': court_coords,
                'confidence': detection['confidence'],
                'bbox': detection['bbox']
            })
        
        ball_court_positions[frame_idx] = court_detections
        frames_mapped += 1
    
    print(f"\nMapping Summary:")
    print(f"  Total frames: {ball_data['total_frames']}")
    print(f"  Frames with ball detected: {ball_data['frames_with_detection']}")
    print(f"  Frames mapped to court: {frames_mapped}")
    
    # Create output
    if output_file is None:
        input_path = Path(ball_positions_file)
        output_file = input_path.stem + '_court_coords.json'
    
    output_data = {
        'ball_positions_file': str(ball_positions_file),
        'calibration_file': str(calibration_file),
        'video_path': ball_data['video_path'],
        'total_frames': ball_data['total_frames'],
        'fps': ball_data['fps'],
        'frames_with_detection': ball_data['frames_with_detection'],
        'detection_rate': ball_data['detection_rate'],
        'court_dimensions': {
            'length': 94.0,  # feet
            'width': 50.0    # feet
        },
        'ball_court_positions': ball_court_positions
    }
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCourt coordinates saved to: {output_file}")
    
    return output_data


# ── v18 learned homography ────────────────────────────────────────────────────

def _load_v18_model(model_path: str):
    """Load v18 YOLOv8x-pose model with torchvision NMS CPU fallback patch."""
    # Patch torchvision NMS — required when torchvision is the cpu-only build
    # but PyTorch is CUDA: NMS kernel missing → apply CPU fallback.
    try:
        import torchvision.ops as tvops
        _orig_nms = tvops.nms
        def _nms_cpu_fallback(boxes, scores, iou_threshold):
            dev = boxes.device
            return _orig_nms(boxes.cpu(), scores.cpu(), float(iou_threshold)).to(dev)
        tvops.nms = _nms_cpu_fallback
    except Exception:
        pass  # torchvision not installed or already patched

    from ultralytics import YOLO
    model = YOLO(model_path)
    print(f"Loaded v18 model: {model_path}")
    return model


def _build_homography_v18(frame: np.ndarray, model) -> "np.ndarray | None":
    """
    Run v18 YOLOv8x-pose on a BGR frame → plain-DLT homography (pixel → court ft).

    Returns 3×3 H, or None if:
      - no keypoints detected above confidence threshold
      - fewer than 4 usable points
      - all court-space points are collinear (SVD rank check)
      - findHomography returns None or a degenerate matrix
    """
    h, w = frame.shape[:2]

    results = model(frame, conf=_CONF_THRESHOLD, verbose=False)
    if not results or results[0].keypoints is None or len(results[0].keypoints.xy) == 0:
        return None

    kp_px   = results[0].keypoints.xy[0].cpu().numpy()    # (18, 2)
    kp_conf = results[0].keypoints.conf[0].cpu().numpy()  # (18,)

    mask = (
        (kp_conf >= _CONF_THRESHOLD) &
        (kp_px[:, 0] > 1) & (kp_px[:, 0] < w - 1) &
        (kp_px[:, 1] > 1) & (kp_px[:, 1] < h - 1)
    )
    if mask.sum() < _MIN_PTS_FOR_H:
        return None

    src_px    = kp_px[mask].astype(np.float32)
    dst_court = _COURT_KPS_FT[mask].astype(np.float32)

    # Collinearity check: if all court coords lie on a line, H is undefined
    centered = dst_court - dst_court.mean(axis=0)
    _, sv, _ = np.linalg.svd(centered)
    if sv[0] < 1e-6 or (sv[1] / sv[0]) < 0.01:
        return None

    # Plain DLT (method=0): RANSAC fails on broadcast views because
    # near-baseline keypoints are collinear in image space, causing
    # every RANSAC minimal 4-point sample to be degenerate.
    H, _ = cv2.findHomography(src_px, dst_court, method=0)
    if H is None:
        return None

    det = abs(np.linalg.det(H[:2, :2]))
    if det < 1e-6 or det > 1e10:
        return None

    return H


def map_ball_positions_with_model(
    ball_positions_file: str,
    model_path: str,
    output_file: str = None,
) -> dict:
    """
    Map ball pixel positions to court coordinates using the v18 learned homography.

    For each video frame:
      1. Run v18 YOLOv8x-pose → detect 18 court keypoints
      2. Build homography H via plain DLT (pixel → court ft)
      3. If H is valid, project detected ball center pixel → court coords
      4. If H is invalid (pose model failed this frame), use the last valid H
         (temporal fallback — camera angle rarely changes between frames)

    Args:
        ball_positions_file: JSON from detect_ball_positions.py
        model_path:          Path to v18 best.pt model
        output_file:         Output JSON path (default: <stem>_court_coords.json)

    Returns:
        Output data dict (also written to output_file)
    """
    with open(ball_positions_file, 'r') as f:
        ball_data = json.load(f)

    video_path = Path(ball_data['video_path'])
    if not video_path.exists():
        raise FileNotFoundError(
            f"Source video not found: {video_path}\n"
            "The v18 learned mode opens the original video to run pose inference.\n"
            "Ensure the video is present at the path stored in the ball positions JSON."
        )

    model = _load_v18_model(model_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ball_positions = ball_data['ball_positions']

    # Find the last frame index we actually need to process
    frame_indices_with_detections = {
        int(k) for k, v in ball_positions.items() if v
    }
    last_needed = max(frame_indices_with_detections) if frame_indices_with_detections else 0
    frames_to_process = min(last_needed + 1, total_video_frames)

    print(f"\nv18 learned homography mapping:")
    print(f"  Video:         {video_path}")
    print(f"  Ball frames:   {len(frame_indices_with_detections)} frames with detections")
    print(f"  Processing:    frames 0–{frames_to_process - 1} (to cover last detection)")

    ball_court_positions = {}
    last_valid_H = None
    H_valid_count = 0
    H_fallback_count = 0
    H_failed_count = 0
    frames_mapped = 0

    for frame_idx in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break

        # Run v18 pose model on every frame to keep H up to date
        H = _build_homography_v18(frame, model)
        if H is not None:
            last_valid_H = H
            H_valid_count += 1

        # Only map frames that have ball detections
        frame_key = str(frame_idx)
        detections = ball_positions.get(frame_key, [])

        if not detections:
            ball_court_positions[frame_idx] = []
            continue

        if last_valid_H is not None:
            used_H = last_valid_H
            h_source = "current" if H is not None else "temporal_fallback"
            if H is None:
                H_fallback_count += 1
        else:
            # No valid H yet for this frame
            H_failed_count += 1
            ball_court_positions[frame_idx] = [
                {
                    'pixel_coords': det['center'],
                    'court_coords': None,
                    'confidence': det['confidence'],
                    'bbox': det['bbox'],
                    'homography_valid': False,
                    'homography_source': 'none',
                }
                for det in detections
            ]
            continue

        court_detections = []
        for det in detections:
            court_coords = pixel_to_court(det['center'], used_H)
            court_detections.append({
                'pixel_coords': det['center'],
                'court_coords': court_coords,
                'confidence': det['confidence'],
                'bbox': det['bbox'],
                'homography_valid': True,
                'homography_source': h_source,
            })
        ball_court_positions[frame_idx] = court_detections
        frames_mapped += 1

    cap.release()

    print(f"\nMapping summary:")
    print(f"  Total ball-detection frames: {len(frame_indices_with_detections)}")
    print(f"  Mapped (H valid this frame): {H_valid_count} frames had fresh H")
    print(f"  Mapped (temporal fallback):  {H_fallback_count} frames used last-valid H")
    print(f"  Unmapped (no H at all):      {H_failed_count} frames")
    print(f"  Total court-coord frames:    {frames_mapped}")

    if output_file is None:
        output_file = Path(ball_positions_file).stem + '_court_coords.json'

    output_data = {
        'ball_positions_file': str(ball_positions_file),
        'model_path': str(model_path),
        'homography_mode': 'v18_learned_dlt',
        'video_path': ball_data['video_path'],
        'total_frames': ball_data['total_frames'],
        'fps': ball_data['fps'],
        'frames_with_detection': ball_data['frames_with_detection'],
        'detection_rate': ball_data['detection_rate'],
        'homography_stats': {
            'H_valid_frames': H_valid_count,
            'H_temporal_fallback_frames': H_fallback_count,
            'H_failed_frames': H_failed_count,
        },
        'court_dimensions': {'length': 94.0, 'width': 50.0},
        'ball_court_positions': ball_court_positions,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nCourt coordinates saved to: {output_file}")
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Map ball positions to court coordinates')
    parser.add_argument('--ball-positions', type=str, required=True,
                       help='JSON file with ball detections (from detect_ball_positions.py)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to v18 YOLOv8x-pose .pt model (learned homography)')
    parser.add_argument('--calibration', type=str, default=None,
                       help='JSON file with manual homography calibration (legacy)')
    parser.add_argument('--output', type=str, help='Output JSON file path')

    args = parser.parse_args()

    if args.model is None and args.calibration is None:
        parser.error("Provide either --model (v18 learned) or --calibration (manual).")
    if args.model is not None and args.calibration is not None:
        parser.error("Provide only one of --model or --calibration, not both.")

    if args.model is not None:
        map_ball_positions_with_model(
            ball_positions_file=args.ball_positions,
            model_path=args.model,
            output_file=args.output,
        )
    else:
        map_ball_positions_to_court(
            ball_positions_file=args.ball_positions,
            calibration_file=args.calibration,
            output_file=args.output,
        )


if __name__ == '__main__':
    main()
