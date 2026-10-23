"""Phase 13 — live analytics overlays for the broadcast panel.

Port/adapt of three drawers from abdullahtarek/basketball_analysis:
  - team_ball_control_drawer   → draw_team_ball_control()
  - pass_and_interceptions_drawer → draw_pass_int_counter()
  - speed_and_distance_drawer  → draw_speed_distance_labels() + draw_top_distance_panel()

Design
──────
All three overlays need *cumulative* state up to the current frame:
  • Team ball-control %: running count of possession_team_id per frame.
  • Pass/int counters:   running count of pass/intercept events up to this frame.
  • Speed/distance:      per-player fields already live on PlayerTrack (Phase 12d).

We precompute the two running-count series once per clip and feed them to the
renderer. Per-player speed labels just read PlayerTrack.speed_ft_per_s.

No external dependencies beyond OpenCV (already a project dep).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ── pre-computation helpers ──────────────────────────────────────────────────

def compute_team_ball_control_series(results: list) -> List[Tuple[int, int, int]]:
    """Cumulative (team1_frames, team2_frames, indet_frames) at each frame index.

    `indet` = no team is credited (loose ball, no possession resolved).
    Returns list of length len(results); entry i holds running totals
    inclusive of frame i.
    """
    series: List[Tuple[int, int, int]] = []
    t1 = t2 = indet = 0
    for r in results:
        team = getattr(r, "possession_team_id", None)
        if team == 1:
            t1 += 1
        elif team == 2:
            t2 += 1
        else:
            indet += 1
        series.append((t1, t2, indet))
    return series


def compute_pass_int_series(
    n_frames: int,
    pass_int_result,
) -> List[Dict[int, Dict[str, int]]]:
    """Cumulative pass/int counts per team up to each frame.

    Returns list length n_frames; entry i is {team_id: {"passes": k, "intercepts": k}}
    with keys for teams 1 and 2 pre-initialised.
    """
    series: List[Dict[int, Dict[str, int]]] = []
    running = {1: {"passes": 0, "intercepts": 0},
               2: {"passes": 0, "intercepts": 0}}
    passes = sorted(getattr(pass_int_result, "passes", []),
                    key=lambda e: e.frame_idx) if pass_int_result else []
    ints   = sorted(getattr(pass_int_result, "interceptions", []),
                    key=lambda e: e.frame_idx) if pass_int_result else []
    pi = ii = 0
    for f in range(n_frames):
        while pi < len(passes) and passes[pi].frame_idx <= f:
            tid = passes[pi].team_id
            if tid in running:
                running[tid]["passes"] += 1
            pi += 1
        while ii < len(ints) and ints[ii].frame_idx <= f:
            tid = ints[ii].team_id
            if tid in running:
                running[tid]["intercepts"] += 1
            ii += 1
        # Snapshot a copy
        series.append({1: dict(running[1]), 2: dict(running[2])})
    return series


# ── drawers ──────────────────────────────────────────────────────────────────

_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_SHADOW = (20, 20, 20)


def _translucent_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     color=_BLACK, alpha: float = 0.55) -> None:
    ov = frame.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)


def draw_team_ball_control(
    frame: np.ndarray,
    cum_counts: Tuple[int, int, int],
    team_colors: Dict[int, Tuple[int, int, int]],
    anchor: Tuple[int, int] = None,  # (x, y) top-right of panel; default bottom-right
) -> None:
    """Render the rolling Team Ball Control % panel in-place."""
    H, W = frame.shape[:2]
    t1, t2, indet = cum_counts
    total = t1 + t2  # exclude indet from the percentage denominator (matches Tarek)
    if total == 0:
        pct1 = pct2 = 0.0
    else:
        pct1 = 100.0 * t1 / total
        pct2 = 100.0 * t2 / total

    # Panel geometry
    pw, ph = 320, 80
    if anchor is None:
        x2 = W - 12; y2 = H - 12
        x1 = x2 - pw; y1 = y2 - ph
    else:
        x1, y1 = anchor; x2 = x1 + pw; y2 = y1 + ph

    _translucent_box(frame, x1, y1, x2, y2, alpha=0.55)
    cv2.putText(frame, "TEAM BALL CONTROL", (x1 + 10, y1 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)

    # Bar
    bar_x1 = x1 + 10; bar_x2 = x2 - 10
    bar_y1 = y1 + 32; bar_y2 = bar_y1 + 16
    bar_w  = bar_x2 - bar_x1
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (60, 60, 60), -1)
    split = bar_x1 + int(bar_w * (pct1 / 100.0))
    c1 = team_colors.get(1, (220, 220, 220))
    c2 = team_colors.get(2, (30,  30,  180))
    cv2.rectangle(frame, (bar_x1, bar_y1), (split, bar_y2), c1, -1)
    cv2.rectangle(frame, (split,  bar_y1), (bar_x2, bar_y2), c2, -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), _WHITE, 1)

    # Labels
    cv2.putText(frame, f"T1 {pct1:4.1f}%", (bar_x1, bar_y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c1, 2, cv2.LINE_AA)
    label2 = f"{pct2:4.1f}% T2"
    (tw, _), _ = cv2.getTextSize(label2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.putText(frame, label2, (bar_x2 - tw, bar_y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c2, 2, cv2.LINE_AA)


def draw_pass_int_counter(
    frame: np.ndarray,
    counts: Dict[int, Dict[str, int]],
    team_colors: Dict[int, Tuple[int, int, int]],
    anchor: Tuple[int, int] = (12, None),  # default top-left under HUD
) -> None:
    """Render pass / interception cumulative counters in-place."""
    H, W = frame.shape[:2]
    pw, ph = 220, 78
    x1 = anchor[0] if anchor and anchor[0] is not None else 12
    y1 = anchor[1] if anchor and anchor[1] is not None else H - ph - 12
    x2 = x1 + pw; y2 = y1 + ph

    _translucent_box(frame, x1, y1, x2, y2, alpha=0.55)
    cv2.putText(frame, "PASSES / INTERCEPTS", (x1 + 8, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)

    for i, team in enumerate((1, 2)):
        c = team_colors.get(team, (200, 200, 200))
        yy = y1 + 38 + i * 22
        line = f"T{team}  P {counts[team]['passes']:2d}   I {counts[team]['intercepts']:2d}"
        cv2.putText(frame, line, (x1 + 10, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)


def draw_speed_label(frame: np.ndarray, player) -> None:
    """Per-player speed label under their box.

    Operates on an inference.pipeline.PlayerTrack; only draws if the player
    has a non-zero speed (populated only in two-pass after compute_speed_and_distance).
    """
    speed = getattr(player, "speed_ft_per_s", 0.0) or 0.0
    if speed <= 0.1:
        return
    x1, y1, x2, y2 = player.box_xyxy
    label = f"{speed:4.1f} ft/s"
    # Shadow for legibility
    cv2.putText(frame, label, (x1 + 1, y2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _SHADOW, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, y2 + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 240, 255), 1, cv2.LINE_AA)


def draw_top_distance_panel(
    frame: np.ndarray,
    totals: List[Tuple[int, float, float, Optional[int]]],
    team_colors: Dict[int, Tuple[int, int, int]],
    top_n: int = 3,
    anchor: Tuple[int, int] = (12, 180),
) -> None:
    """Top-N players by cumulative distance traveled.

    `totals`: list of (track_id, distance_ft, peak_speed_ft_s, team_id).
    """
    if not totals:
        return
    top = sorted(totals, key=lambda t: -t[1])[:top_n]
    pw = 230
    ph = 28 + 22 * len(top)
    x1, y1 = anchor
    x2 = x1 + pw; y2 = y1 + ph

    _translucent_box(frame, x1, y1, x2, y2, alpha=0.55)
    cv2.putText(frame, "DISTANCE LEADERS", (x1 + 8, y1 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _WHITE, 1, cv2.LINE_AA)
    for i, (tid, dist, peak, team) in enumerate(top):
        c = team_colors.get(team, (220, 220, 220)) if team else (220, 220, 220)
        line = f"#{tid:<3d} {dist:6.1f} ft   pk {peak:4.1f}"
        cv2.putText(frame, line, (x1 + 10, y1 + 40 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
