"""
Phase 9b — Online screen (pick) detector.

Runs frame-by-frame using BoT-SORT player court positions (feet).
No new models required — uses the homography + track data already computed.

Screen trigger logic:
─────────────────────
A screen is confirmed when ALL of the following hold for MIN_DURATION frames:

  1. SCREENER:  a tracked player whose court-space speed is < MAX_SCREENER_SPEED
                (they are stationary / nearly stationary for the window).

  2. CUTTER:    a DIFFERENT tracked player who is moving TOWARD the screener
                at >= MIN_CUTTER_SPEED ft/frame AND whose distance from the
                screener is in [MIN_PROX, MAX_PROX] ft.

  3. PROXIMITY: screener–cutter distance <= MAX_PROX ft (they are near
                each other).

  4. BALL:      (optional) ball court position within BALL_PROX ft of the
                cutter — flags as on_ball vs off_ball screen.

Cooldown: once a screen is emitted, the same *unordered* screener–cutter pair
cannot re-trigger for COOLDOWN frames. Keying the cooldown on an ordered pair
let a noisy speed estimate emit both (A screens B) and (B screens A) for the
same collision, which is physically impossible.

Output: ScreenEvent emitted on the confirmation frame.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


# ─────────────────────────── data classes ────────────────────────────────────

@dataclass
class ScreenEvent:
    frame: int
    screener_id: int
    cutter_id: int
    screen_type: str                             # 'on_ball' | 'off_ball'
    screener_pos_ft: Tuple[float, float]         # (x, y) court feet
    cutter_pos_ft: Tuple[float, float]
    separation_ft: float                         # screener–cutter distance
    confidence: float                            # 0–1


# ─────────────────────────── speed estimation ────────────────────────────────

def track_speeds(
    history: Dict[int, deque],
    window: int = 3,
) -> Dict[int, float]:
    """Per-track speed in ft/frame, normalised by the actual frame gap.

    Consecutive history entries are not necessarily consecutive frames: tracks go
    missing for a few frames and reappear. Dividing distance by the frame gap is
    what makes the number a speed; using raw distance makes a 3-frame gap look
    like a 3x faster player. This is the same defect previously fixed in the shot
    detector.
    """
    speeds: Dict[int, float] = {}
    for tid, hist in history.items():
        if len(hist) < 2:
            continue
        rates = []
        for i in range(1, min(len(hist), window + 1)):
            f0, x0, y0 = hist[-i - 1]
            f1, x1, y1 = hist[-i]
            gap = max(int(f1) - int(f0), 1)
            rates.append(float(np.hypot(x1 - x0, y1 - y0)) / gap)
        speeds[tid] = float(np.mean(rates)) if rates else 0.0
    return speeds


# ─────────────────────────── main detector ───────────────────────────────────

class OnlineScreenDetector:
    """
    Stateful, frame-by-frame screen detector.

    Usage (inside demo loop):
        detector = OnlineScreenDetector()
        ...
        events = detector.update(
            frame_idx      = fi,
            players_court  = players_court,   # list of (x_ft, y_ft, track_id)
            ball_court     = ball_court,       # (x_ft, y_ft) or None
        )
        for ev in events:
            print(ev)
    """

    # ── tuneable constants ─────────────────────────────────────────────────
    # Court-space speeds are expressed per frame at REFERENCE_FPS. At 60 fps the
    # same physical speed is half as many feet per frame, so a raw 0.40 ft/frame
    # would accept a 24 ft/s player as "nearly still" (12 ft/s at 30 fps).
    MAX_SCREENER_SPEED = 0.40   # ft/frame at REFERENCE_FPS — screener nearly still
    MIN_CUTTER_SPEED   = 0.20   # ft/frame at REFERENCE_FPS — cutter moving
    MIN_PROX           = 1.5    # ft  — min screener–cutter separation
    MAX_PROX           = 7.0    # ft  — max screener–cutter separation (body contact ~2ft, allow up to 7)
    BALL_PROX          = 12.0   # ft  — ball within this → on_ball screen
    MIN_DURATION       = 5      # frames both conditions must hold before trigger
    COOLDOWN           = 90     # frames per screener–cutter pair
    HISTORY            = 12     # frames of player position history kept
    DISPLAY_FRAMES     = 90     # frames caller should flash the HUD label
    REFERENCE_FPS      = 30.0   # rate the numeric thresholds above assume

    # Minimum frames a player must be tracked before acting as screener
    MIN_TRACK_AGE      = 5

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        # Speeds are ft/frame, so they fall as fps rises; windows are durations, so
        # they rise with fps. Keeping both in one physical unit is what makes the
        # same thresholds meaningful at 25, 30 or 60 fps.
        scale = fps / self.REFERENCE_FPS
        self.max_screener_speed = self.MAX_SCREENER_SPEED / scale
        self.min_cutter_speed = self.MIN_CUTTER_SPEED / scale
        self.min_duration = max(1, int(round(self.MIN_DURATION * scale)))
        self.cooldown = max(1, int(round(self.COOLDOWN * scale)))
        self.display_frames = max(1, int(round(self.DISPLAY_FRAMES * scale)))
        # Per-track position history: deque of (frame_idx, x_ft, y_ft)
        self._history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.HISTORY)
        )
        # Candidate accumulator: (screener_id, cutter_id) → consecutive frames
        self._candidates: Dict[Tuple[int, int], int] = {}
        # Cooldown per pair
        self._cooldown: Dict[Tuple[int, int], int] = {}
        self.events: List[ScreenEvent] = []

    def reset(self):
        """Call on scene cut."""
        self._history.clear()
        self._candidates.clear()
        self._cooldown.clear()

    @staticmethod
    def _pair_key(a: int, b: int) -> Tuple[int, int]:
        """Order-independent key for a screener/cutter pair."""
        return (a, b) if a <= b else (b, a)

    # ── main update ────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        players_court: List[Tuple],          # (x_ft, y_ft, track_id)
        ball_court: Optional[Tuple[float, float]] = None,
    ) -> List[ScreenEvent]:
        """Feed one frame. Returns list of newly triggered ScreenEvents (usually empty)."""

        # tick all cooldowns
        for pair in list(self._cooldown):
            self._cooldown[pair] -= 1
            if self._cooldown[pair] <= 0:
                del self._cooldown[pair]

        # Update position history for all visible players
        for entry in players_court:
            x_ft, y_ft, tid = entry[0], entry[1], entry[2]
            self._history[tid].append((frame_idx, x_ft, y_ft))

        # Compute per-track speed (ft/frame) and current position
        speeds: Dict[int, float] = track_speeds(self._history)
        positions: Dict[int, Tuple[float, float]] = {}
        for tid, hist in self._history.items():
            if len(hist) < 2:
                continue
            _, cx, cy = hist[-1]
            positions[tid] = (cx, cy)

        # Only consider players visible THIS frame
        visible = {entry[2]: (entry[0], entry[1]) for entry in players_court
                   if len(self._history[entry[2]]) >= self.MIN_TRACK_AGE}

        fired: List[ScreenEvent] = []
        active_pairs: set = set()

        # Check all ordered pairs (screener, cutter) — screener must be slow
        for s_id, (sx, sy) in visible.items():
            if speeds.get(s_id, 999) > self.max_screener_speed:
                continue  # not stationary enough

            for c_id, (cx, cy) in visible.items():
                if c_id == s_id:
                    continue
                pair = (s_id, c_id)

                # proximity check
                sep = np.hypot(sx - cx, sy - cy)
                if sep < self.MIN_PROX or sep > self.MAX_PROX:
                    self._candidates.pop(pair, None)
                    continue

                # cutter speed check
                if speeds.get(c_id, 0) < self.min_cutter_speed:
                    self._candidates.pop(pair, None)
                    continue

                # cutter must be MOVING TOWARD screener
                hist_c = self._history[c_id]
                if len(hist_c) < 2:
                    continue
                _, px0, py0 = hist_c[-2]
                prev_sep = np.hypot(sx - px0, sy - py0)
                if sep >= prev_sep:  # not getting closer
                    self._candidates.pop(pair, None)
                    continue

                # all conditions met — increment candidate counter
                self._candidates[pair] = self._candidates.get(pair, 0) + 1
                active_pairs.add(pair)

                # has it persisted long enough?
                if self._candidates[pair] < self.min_duration:
                    continue

                # cooldown check. Keyed on the unordered pair: two players cannot
                # be each other's screener and cutter, so whichever role the noisy
                # speed estimate assigns, the same collision must not re-fire.
                if self._pair_key(s_id, c_id) in self._cooldown:
                    continue

                # ── emit event ──
                # On-ball: ball within BALL_PROX ft of cutter
                on_ball = False
                if ball_court is not None:
                    bx, by = ball_court
                    if np.hypot(bx - cx, by - cy) <= self.BALL_PROX:
                        on_ball = True

                conf = min(1.0, self._candidates[pair] / (self.min_duration * 2))
                ev = ScreenEvent(
                    frame=frame_idx,
                    screener_id=s_id,
                    cutter_id=c_id,
                    screen_type="on_ball" if on_ball else "off_ball",
                    screener_pos_ft=(sx, sy),
                    cutter_pos_ft=(cx, cy),
                    separation_ft=float(sep),
                    confidence=conf,
                )
                self.events.append(ev)
                fired.append(ev)
                self._cooldown[self._pair_key(s_id, c_id)] = self.cooldown
                self._candidates[pair] = 0  # reset so it must re-accumulate

        # Prune candidate pairs that were not active this frame
        for pair in list(self._candidates):
            if pair not in active_pairs:
                del self._candidates[pair]

        return fired

    # ── summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n = len(self.events)
        on  = sum(1 for e in self.events if e.screen_type == "on_ball")
        off = sum(1 for e in self.events if e.screen_type == "off_ball")
        if n == 0:
            return "0 screens detected"
        return f"{n} screen(s) — {on}x on_ball  {off}x off_ball"
