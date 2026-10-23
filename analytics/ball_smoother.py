"""
Phase 7b — Offline ball-track smoothing.

Strategy (inspired by abdullahtarek/basketball_analysis, improved with our motion gate):
  1. Walk the FrameResult list, take only NON-coasted, NON-interpolated ball detections
     as raw measurements (cx, cy in pixel space).
  2. Motion-gate raw detections: drop any sample whose displacement from the previous
     accepted sample exceeds `max_jump_px * frame_gap`.
  3. Fill gaps with hindsight-aware values.  A ball in flight follows a parabola, so
     short gaps (≤ `linear_gap_max` frames) use a straight chord and long gaps use a
     local constant-acceleration fit (B-04).  Leading/trailing gaps use the nearest
     known value.
  4. Optional rolling-median smoothing over `smooth_window` frames for extra crispness.
  5. Write smoothed (cx, cy) back into each FrameResult.ball and set `coasted=False`.
     `interpolated` is set ONLY on frames that have no real detection behind them, so
     callers can tell a measured position from an invented one.

Player court positions can be smoothed the same way via `smooth_player_court_positions`.

This is offline-only. The streaming online pipeline is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence
import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid importing torch/ultralytics for analytics-only callers
    from inference.pipeline import FrameResult, PlayerTrack

DEFAULT_MAX_JUMP_PX = 200.0
DEFAULT_LINEAR_GAP_MAX = 8


def _fill_gaps(
    values: np.ndarray,
    *,
    linear_gap_max: int = DEFAULT_LINEAR_GAP_MAX,
    curve_context: int = 3,
) -> np.ndarray:
    """Fill NaNs in a 1-D series, using a curve for long internal gaps.

    Short gaps are linear; a gap longer than `linear_gap_max` is filled from a
    degree-2 polynomial fitted to up to `curve_context` known anchors on each
    side (a locally constant-acceleration model).  If there are too few anchors
    or the fit is degenerate, the gap falls back to linear.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    known = np.flatnonzero(~np.isnan(out))
    if known.size == 0:
        return out
    if known.size == 1:
        out[:] = out[known[0]]
        return out

    # Leading / trailing NaNs: nearest known value.
    out[: known[0]] = out[known[0]]
    out[known[-1] + 1:] = out[known[-1]]

    for a, b in zip(known[:-1], known[1:]):
        gap = int(b - a - 1)
        if gap <= 0:
            continue
        gap_idx = np.arange(a + 1, b, dtype=np.float64)

        if gap <= linear_gap_max:
            out[a + 1:b] = np.linspace(out[a], out[b], gap + 2)[1:-1]
            continue

        lo = max(0, int(np.searchsorted(known, a)) - curve_context)
        hi = min(int(known.size), int(np.searchsorted(known, b)) + curve_context + 1)
        anchors = known[lo:hi]
        filled = False
        if anchors.size >= 3:
            t = anchors.astype(np.float64)
            coeffs = np.polyfit(t, out[anchors], 2)
            if np.all(np.isfinite(coeffs)):
                out[a + 1:b] = np.polyval(coeffs, gap_idx)
                filled = True
        if not filled:
            out[a + 1:b] = np.linspace(out[a], out[b], gap + 2)[1:-1]

    return out


def smooth_ball_track(
    frames: Sequence[FrameResult],
    *,
    max_jump_px: float = DEFAULT_MAX_JUMP_PX,
    smooth_window: int = 5,
    linear_gap_max: int = DEFAULT_LINEAR_GAP_MAX,
    project_to_court=None,
) -> int:
    """
    Mutate `frames` in place: replace each FrameResult.ball with a smoothed version.

    Parameters
    ----------
    frames : sequence of FrameResult (length N).
    max_jump_px : reject raw detections jumping > max_jump_px * frame_gap from the
        last accepted one. Default 200 px (shared with the CLI default).
    smooth_window : rolling-median window applied AFTER interpolation; set to 0 or 1
        to disable. Default 5.
    linear_gap_max : gaps of at most this many frames are filled with a straight
        line; longer gaps use a local quadratic fit. Default 8 frames.
    project_to_court : optional callable (H, cx, cy) -> (x_ft, y_ft) used to refresh
        ball.court_pos from the smoothed pixel position. If None, leaves court_pos as-is.

    Returns
    -------
    Number of frames that gained a ball position from interpolation
    (i.e. previously had no ball, now do).
    """
    from inference.pipeline import BallState

    n = len(frames)
    cx = np.full(n, np.nan, dtype=np.float64)
    cy = np.full(n, np.nan, dtype=np.float64)

    # --- Stage 1: collect raw, non-coasted detections ---
    for i, f in enumerate(frames):
        b = f.ball
        if b is None or b.coasted:
            continue
        cx[i] = b.cx
        cy[i] = b.cy

    # --- Stage 2: motion gate (drop big jumps) ---
    last_i = -1
    for i in range(n):
        if np.isnan(cx[i]):
            continue
        if last_i < 0:
            last_i = i
            continue
        gap = i - last_i
        dx = cx[i] - cx[last_i]
        dy = cy[i] - cy[last_i]
        if (dx * dx + dy * dy) ** 0.5 > max_jump_px * gap:
            cx[i] = np.nan
            cy[i] = np.nan
        else:
            last_i = i

    # --- Stage 3: hindsight interpolation (gap-aware, B-04) ---
    # `backed` records which frames still hold a real, motion-gate-passed detection.
    # Everything else gets a position from interpolation, and only that deserves to be
    # reported as interpolated. Stamping every frame was a bug: the overlay label and
    # the JSON field both became constant, so a reader could not tell a detection from
    # an invented position.
    backed = ~np.isnan(cx)
    cx_s = _fill_gaps(cx, linear_gap_max=linear_gap_max)
    cy_s = _fill_gaps(cy, linear_gap_max=linear_gap_max)

    # --- Stage 4: optional rolling-median smoothing ---
    if smooth_window and smooth_window > 1:
        df = pd.DataFrame({"cx": cx_s, "cy": cy_s})
        df = df.rolling(window=smooth_window, center=True, min_periods=1).median()
        cx_s = df["cx"].to_numpy()
        cy_s = df["cy"].to_numpy()

    # --- Stage 5: write back ---
    filled = 0
    for i, f in enumerate(frames):
        if np.isnan(cx_s[i]) or np.isnan(cy_s[i]):
            continue
        had_ball = f.ball is not None
        prev_box = f.ball.det_box if had_ball else None
        prev_conf = f.ball.det_conf if had_ball else 0.0

        court_pos = None
        if project_to_court is not None and f.H is not None:
            court_pos = project_to_court(f.H, float(cx_s[i]), float(cy_s[i]))

        f.ball = BallState(
            tracked=True,
            coasted=False,
            cx=float(cx_s[i]),
            cy=float(cy_s[i]),
            coast_frames=0,
            court_pos=court_pos if court_pos is not None
                      else (f.ball.court_pos if had_ball else None),
            det_box=prev_box,
            det_conf=prev_conf,
            interpolated=not bool(backed[i]),
        )
        if not had_ball:
            filled += 1

    return filled


def smooth_player_court_positions(
    frames: Sequence[FrameResult],
    *,
    smooth_window: int = 5,
) -> int:
    """
    Mutate `frames` in place: replace each PlayerTrack.court_pos with a rolling-median
    smoothed version per track_id. Pixel bbox is left untouched (still renders on the
    actual player); only the court projection is smoothed (kills minimap jitter).

    Returns total number of (frame, track) pairs updated.
    """
    if smooth_window is None or smooth_window <= 1:
        return 0

    # Gather per-track time series of (frame_idx, x_ft, y_ft)
    series: dict[int, list[tuple[int, float, float]]] = {}
    for f in frames:
        for p in f.players:
            if p.court_pos is None:
                continue
            series.setdefault(p.track_id, []).append((f.frame_idx, p.court_pos[0], p.court_pos[1]))

    smoothed: dict[tuple[int, int], tuple[float, float]] = {}  # (frame_idx, track_id) -> (x, y)
    for tid, pts in series.items():
        if len(pts) < 2:
            continue
        arr = np.array(pts, dtype=np.float64)  # (k, 3)
        df = pd.DataFrame({"x": arr[:, 1], "y": arr[:, 2]})
        df = df.rolling(window=smooth_window, center=True, min_periods=1).median()
        for k, fi in enumerate(arr[:, 0].astype(int)):
            smoothed[(int(fi), tid)] = (float(df["x"].iloc[k]), float(df["y"].iloc[k]))

    n_updated = 0
    for f in frames:
        for p in f.players:
            key = (f.frame_idx, p.track_id)
            if key in smoothed:
                p.court_pos = smoothed[key]
                n_updated += 1
    return n_updated
