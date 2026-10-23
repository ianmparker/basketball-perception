"""Feature extraction for shot-candidate filtering.

A candidate is a frame emitted by the high-recall shot detector. Features are computed
from the already-collected pipeline frame JSON, so this is cheap to iterate on.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

HOOPS_FT = ((5.25, 25.0), (88.75, 25.0))


def _nearest_frame(frames_by_idx: dict[int, dict], frame: int) -> dict | None:
    if frame in frames_by_idx:
        return frames_by_idx[frame]
    keys = sorted(frames_by_idx)
    if not keys:
        return None
    return frames_by_idx[min(keys, key=lambda k: abs(k - frame))]


def _hoop_pixels(H) -> list[tuple[float, float]]:
    if H is None:
        return []
    try:
        H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    except np.linalg.LinAlgError:
        return []
    out = []
    for hx, hy in HOOPS_FT:
        pt = np.array([[[hx, hy]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, H_inv)[0][0]
        out.append((float(px[0]), float(px[1])))
    return out


def _wrist_up(player: dict) -> bool:
    kps = player.get("pose_kps") or []
    kpc = player.get("pose_kpc") or []
    if len(kps) < 11 or len(kpc) < 11:
        return False
    for wrist, elbow in ((9, 7), (10, 8)):
        if kpc[wrist] > 0.25 and kpc[elbow] > 0.25 and kps[wrist][1] < kps[elbow][1] - 3:
            return True
    return False


def candidate_features(data: dict, event_frame: int, event_confidence: float = 0.0,
                       window: int = 5) -> dict:
    """Features for one candidate frame from a ClipResult JSON payload."""
    frames = data.get("frames", [])
    by_idx = {int(fr["frame_idx"]): fr for fr in frames}
    center = _nearest_frame(by_idx, event_frame)
    if center is None:
        return {}
    h = float(data.get("height") or 360)
    w = float(data.get("width") or 640)
    ball = center.get("ball") or {}
    bx = float(ball.get("cx", 0.0))
    by = float(ball.get("cy", 0.0))
    tracked = bool(ball.get("tracked"))

    # Ball window
    window_frames = []
    for idx in range(event_frame - window, event_frame + 1):
        fr = _nearest_frame(by_idx, idx)
        if fr is not None:
            window_frames.append(fr)
    balls = [fr.get("ball") for fr in window_frames if fr.get("ball")]
    tracked_balls = [b for b in balls if b.get("tracked")]
    tracked_frac = len(tracked_balls) / max(1, len(window_frames))
    ys = [float(b["cy"]) for b in tracked_balls]
    xs = [float(b["cx"]) for b in tracked_balls]
    rise_px = (max(ys) - min(ys)) if len(ys) >= 2 else 0.0
    if len(tracked_balls) >= 3:
        b3, b2, b1 = tracked_balls[-3:]
        vx = (float(b1["cx"]) - float(b2["cx"]))
        vy = (float(b1["cy"]) - float(b2["cy"]))
    else:
        vx = vy = 0.0
    speed = math.hypot(vx, vy)
    vert_frac = abs(vy) / (abs(vx) + abs(vy) + 1e-6)

    # Hoop distance
    hoop_px = _hoop_pixels(center.get("H"))
    min_hoop = min(
        (math.hypot(bx - hx, by - hy) for hx, hy in hoop_px), default=float("nan")
    )

    # Nearest player and pose
    players = center.get("players") or []
    nearest_dist = float("nan")
    nearest_area = 0.0
    wrist_up = False
    for p in players:
        box = p.get("box_xyxy") or [0, 0, 0, 0]
        cx = (float(box[0]) + float(box[2])) / 2
        cy = (float(box[1]) + float(box[3])) / 2
        dist = math.hypot(cx - bx, cy - by)
        if math.isnan(nearest_dist) or dist < nearest_dist:
            nearest_dist = dist
            nearest_area = ((box[2] - box[0]) * (box[3] - box[1])) / (w * h)
            wrist_up = _wrist_up(p)

    quality = center.get("homography_quality") or {}
    inlier = quality.get("inlier_ratio")
    return {
        "ball_height_frac": 1.0 - by / h,
        "ball_speed_px": speed,
        "ball_vert_frac": vert_frac,
        "rise_px": rise_px,
        "tracked_frac": tracked_frac,
        "ball_near_hoop_px": min_hoop,
        "nearest_player_px": nearest_dist,
        "nearest_player_area": nearest_area,
        "pose_wrist_up": int(wrist_up),
        "court_view": int(bool(center.get("court_view"))),
        "homog_inlier": float(inlier) if inlier is not None else 0.0,
        "event_confidence": float(event_confidence),
    }


FEATURE_COLUMNS = [
    "ball_height_frac",
    "ball_speed_px",
    "ball_vert_frac",
    "rise_px",
    "tracked_frac",
    "ball_near_hoop_px",
    "nearest_player_px",
    "nearest_player_area",
    "pose_wrist_up",
    "court_view",
    "homog_inlier",
    "event_confidence",
]
