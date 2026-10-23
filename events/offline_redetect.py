"""Re-run event detection on a *smoothed* two-pass track.

Why this exists
---------------
The live shot detector runs during pass 1, where the ball is only detected on
a fraction of frames: a 43% detection rate means upward streaks rarely reach
the trigger length.  Pass 2 smooths and interpolates the ball track, which
produces a clean arc — but the detectors never see it.

This module replays shot detection over the completed `FrameResult` list so the
detectors consume the same track the renderer does.  It is deliberately a
replay (same `OnlineShotDetector`), so the two paths cannot drift apart.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from events.shot_detector_v2 import OnlineShotDetector, ShotEvent


def _tracks_array(frame) -> np.ndarray:
    players = getattr(frame, "players", None) or []
    if not players:
        return np.empty((0, 8), dtype=np.float32)
    return np.array(
        [
            [p.box_xyxy[0], p.box_xyxy[1], p.box_xyxy[2], p.box_xyxy[3],
             p.track_id, p.conf, getattr(p, "cls_id", 0), 0]
            for p in players
        ],
        dtype=np.float32,
    )


def redetect_shots(
    frames: Sequence,
    fps: float = 30.0,
    *,
    frame_h: int = 1080,
    require_court_projection: bool = False,
) -> tuple[List[ShotEvent], dict]:
    """Replay shot detection over smoothed frames.

    Returns (events, diagnostics).  Events are not written back here; the
    caller decides whether to trust them.
    """
    detector = OnlineShotDetector(
        fps=fps, require_court_projection=require_court_projection
    )
    events: List[ShotEvent] = []

    for frame in frames:
        ball = getattr(frame, "ball", None)
        H = getattr(frame, "H", None)
        event = detector.update(
            frame_idx=int(frame.frame_idx),
            ball_cx=float(ball.cx) if ball is not None else 0.0,
            ball_cy=float(ball.cy) if ball is not None else 0.0,
            ball_tracked=bool(ball is not None and ball.tracked),
            ball_coasted=bool(ball.coasted) if ball is not None else True,
            frame_h=int(frame_h),
            tracks=_tracks_array(frame),
            H=np.asarray(H) if H is not None else None,
        )
        if event is not None:
            events.append(event)

    return events, detector.diagnostics()
