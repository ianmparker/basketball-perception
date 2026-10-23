"""
events/dunk_detector.py — Player-proximity dunk / at-rim layup detector.

Trigger condition (frame-by-frame):
  1. Ball tracker transitions to coasted (coast_frames == 1).
  2. In the last LOOKBACK frames the ball was within BALL_HOOP_PX pixels
     of either NBA hoop (in image space, computed via inverse homography).
     This lookback handles the case where a false positive takes over after
     the real ball disappears at the rim, making coast fire later.
  3. At least one tracked player is within PLAYER_HOOP_PX pixels of that
     same hoop — used to identify the finisher.
  4. Cooldown: no re-trigger within COOLDOWN_FRAMES.

Why pixel space?
  Court-projection fails for players under the basket (out-of-homography
  region) and for the ball at rim height (perspective error ~22ft off).
  Pixel distance to the hoop (computed via H_inv) is far more reliable.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class DunkEvent:
    frame: int
    player_track_id: int           # -1 if no identified finisher
    hoop_side: str                 # "left" | "right"
    ball_hoop_dist_px: float       # pixel distance of closest recent ball pos to hoop
    player_hoop_dist_px: float     # pixel distance of finisher to hoop center (-1 = unknown)
    player_court_ft: Optional[Tuple[float, float]] = None   # (x, y) if available


class OnlineDunkDetector:
    """Frame-by-frame dunk / at-rim finish detector using pixel-space lookback."""

    BALL_HOOP_PX    = 200   # ball must be within this many pixels of hoop
    CONSECUTIVE_MIN = 3     # consecutive frames ball must be near hoop
    PLAYER_HOOP_PX  = 400   # player centre within this many pixels of hoop
    LOOKBACK        = 20    # frames of ball history to scan
    COOLDOWN        = 60    # frames before re-trigger
    DISPLAY_FRAMES  = 90    # frames caller should flash the banner

    def __init__(self):
        self._cooldown_rem = 0
        # Rolling deque: (cx_px, cy_px) for every frame (including coasted)
        self._ball_hist: deque = deque(maxlen=self.LOOKBACK)
        self.events: List[DunkEvent] = []

    def reset(self):
        """Call on scene cut."""
        self._ball_hist.clear()

    def update(
        self,
        frame_idx: int,
        coast_frames: int,
        ball_cx: float,
        ball_cy: float,
        # list of ((hoop_px, hoop_py), side) — precomputed from H_inv this frame
        hoop_pixels: List[Tuple[Tuple[float, float], str]],
        # list of (cx_px, cy_px, track_id, court_pos_or_None) for ALL players
        players_px: List[Tuple[float, float, int, Optional[Tuple[float, float]]]],
    ) -> Optional[DunkEvent]:
        """
        Feed one frame.  Returns DunkEvent on dunk detection, else None.

        Algorithm
        ---------
        On coast_frames==1, scan ball position history against the CURRENT
        frame's hoop pixel locations (from H_inv).  Require at least
        CONSECUTIVE_MIN consecutive history entries within BALL_HOOP_PX pixels
        of a hoop.  This filters out single-frame false matches from noisy
        homography estimates in early video frames.
        """
        # Always record ball position so history stays current
        self._ball_hist.append((ball_cx, ball_cy))

        if self._cooldown_rem > 0:
            self._cooldown_rem -= 1

        if coast_frames != 1:
            return None
        if self._cooldown_rem > 0:
            return None
        if not hoop_pixels:
            return None

        # ── For each hoop, check for a run of CONSECUTIVE_MIN nearby ball frames ─
        for (hx, hy), side in hoop_pixels:
            dists = [hypot(bx - hx, by - hy) for bx, by in self._ball_hist]

            max_run = cur_run = 0
            for d in dists:
                if d <= self.BALL_HOOP_PX:
                    cur_run += 1
                    if cur_run > max_run:
                        max_run = cur_run
                else:
                    cur_run = 0

            if max_run < self.CONSECUTIVE_MIN:
                continue  # not enough consecutive frames near this hoop

            min_d = min(dists)

            # ── Find the player closest to this hoop ─────────────────────────
            best_player_id   = -1
            best_player_dist = float("inf")
            best_court       = None

            for pcx, pcy, tid, court_pos in players_px:
                d = hypot(pcx - hx, pcy - hy)
                if d < self.PLAYER_HOOP_PX and d < best_player_dist:
                    best_player_dist = d
                    best_player_id   = tid
                    best_court       = court_pos

            if best_player_id < 0:
                best_player_dist = -1.0

            ev = DunkEvent(
                frame=frame_idx,
                player_track_id=best_player_id,
                hoop_side=side,
                ball_hoop_dist_px=min_d,
                player_hoop_dist_px=best_player_dist,
                player_court_ft=best_court,
            )
            self.events.append(ev)
            self._cooldown_rem = self.COOLDOWN
            return ev

        return None

    def summary(self) -> str:
        return f"Dunk/at-rim events: {len(self.events)}"
