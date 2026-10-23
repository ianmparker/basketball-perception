"""
Phase 9c — Online pick-and-roll (PnR) detector.

Combines screen events (Phase 9b) and shot events (Phase 9a) to identify
pick-and-roll possessions.

Logic
─────
A PnR is flagged when the following conditions are met within PNR_WINDOW frames:

  1. An on_ball ScreenEvent fires (screener is stationary, cutter/ball-handler
     is moving toward screener with ball nearby).

  2. A ShotEvent fires where the shooter is:
       a. The cutter (ball-handler drives for layup/dunk)   → 'drive'
       b. The cutter (ball-handler pulls up for a jumper)   → 'pull_up'
       c. The screener (screener rolls to basket and shots) → 'roll_shot'

Only on_ball screens can seed a PnR because off_ball screens don't involve the
current ball-handler.

Usage
─────
    pnr = OnlinePnRDetector()

    # after each screen event:
    for sev in screen_evs:
        pnr.record_screen(sev)

    # after each shot event:
    pnr_ev = pnr.check_shot(shot_ev)
    if pnr_ev:
        print(pnr_ev)

    # on scene cut:
    pnr.reset()
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from events.screen_detector_v2 import ScreenEvent
from events.shot_detector_v2 import ShotEvent


# ─────────────────────────── data class ──────────────────────────────────────

@dataclass
class PnREvent:
    frame: int                              # frame the shot / PnR resolves
    screen_frame: int                       # frame the seed screen fired
    shot_frame: int                         # frame the shot fired
    screener_id: int
    ball_handler_id: int                    # cutter_id from the screen event
    shooter_id: Optional[int]              # shooter_track_id from shot event
    outcome: str                            # 'pull_up' | 'roll_shot' | 'unknown'
    shot_type: Optional[str]               # from the shot event
    shot_distance_ft: Optional[float]      # from the shot event
    separation_ft: float                    # screener–cutter sep at screen trigger
    confidence: float                       # combined confidence


# ─────────────────────────── main detector ───────────────────────────────────

class OnlinePnRDetector:
    """
    Stateful combiner that watches for PnR sequences in an event stream.

    Keeps a short sliding window of recent on_ball ScreenEvents and checks
    each ShotEvent against them.
    """

    # How long after a screen can the shot still count as PnR?
    PNR_WINDOW: int = 150      # frames (~5 s at 30 fps)

    # Flash duration for HUD overlay
    DISPLAY_FRAMES: int = 120

    def __init__(self):
        self._screen_buffer: List[ScreenEvent] = []
        self.events: List[PnREvent] = []

    def reset(self):
        """Call on scene cut — clears the screen buffer so stale screens
        cannot match shots from a new scene."""
        self._screen_buffer.clear()

    def record_screen(self, screen_ev: ScreenEvent) -> None:
        """
        Register a screen event as a potential PnR seed.
        Only on_ball screens are eligible.
        """
        if screen_ev.screen_type != "on_ball":
            return
        self._screen_buffer.append(screen_ev)

    def check_shot(self, shot_ev: ShotEvent) -> Optional[PnREvent]:
        """
        Check whether shot_ev resolves a pending PnR action.

        Searches the screen buffer from most-recent to oldest and returns a
        PnREvent for the best match (closest-in-time on_ball screen whose
        ball-handler or screener matches the shooter), or None if no match.

        Also evicts screen events that have aged out of the PNR_WINDOW.
        """
        shot_frame   = shot_ev.frame_start
        shooter_id   = shot_ev.shooter_track_id

        # Expire old screens
        self._screen_buffer = [
            s for s in self._screen_buffer
            if shot_frame - s.frame <= self.PNR_WINDOW
        ]

        if shooter_id is None:
            return None  # can't match without a known shooter

        # Find best matching screen (most recent first)
        best: Optional[ScreenEvent] = None
        for sev in reversed(self._screen_buffer):
            # Gap in frames from screen trigger to shot
            gap = shot_frame - sev.frame
            if gap < 0:
                continue  # shot before screen — shouldn't happen but guard it

            if shooter_id == sev.cutter_id:
                best = sev
                # Distinguish drive-to-basket from pull-up jumper.
                # shot_type="layup" covers both layups and dunks (both are
                # close-range shots the ball handler attacks off the screen).
                if shot_ev.shot_type == "layup":
                    outcome = "drive"
                else:
                    outcome = "pull_up"
                break
            if shooter_id == sev.screener_id:
                best = sev
                outcome = "roll_shot"
                break

        if best is None:
            return None

        # Combine confidence: screen conf × shot conf, penalise large gaps
        gap = shot_frame - best.frame
        gap_penalty  = max(0.0, 1.0 - gap / self.PNR_WINDOW)
        combined_conf = best.confidence * shot_ev.confidence * gap_penalty

        pnr_ev = PnREvent(
            frame            = shot_frame,
            screen_frame     = best.frame,
            shot_frame       = shot_frame,
            screener_id      = best.screener_id,
            ball_handler_id  = best.cutter_id,
            shooter_id       = shooter_id,
            outcome          = outcome,
            shot_type        = shot_ev.shot_type,
            shot_distance_ft = shot_ev.shot_distance_ft,
            separation_ft    = best.separation_ft,
            confidence       = combined_conf,
        )
        self.events.append(pnr_ev)

        # Remove the matched screen from the buffer so it can't double-match
        self._screen_buffer.remove(best)

        return pnr_ev

    # ── summary ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        n   = len(self.events)
        dr  = sum(1 for e in self.events if e.outcome == "drive")
        pu  = sum(1 for e in self.events if e.outcome == "pull_up")
        rs  = sum(1 for e in self.events if e.outcome == "roll_shot")
        if n == 0:
            return "0 pick-and-roll events"
        return f"{n} PnR event(s) — {dr}x drive  {pu}x pull_up  {rs}x roll_shot"
