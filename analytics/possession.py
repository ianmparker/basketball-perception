"""
analytics/possession.py — ball possession & carrier tracking.

Per-frame, given the ball and all player tracks (with their court positions
and team_ids), decide:
  * ball_carrier_track_id: which player currently has the ball, or None
  * possession_team_id:    which team currently has possession (1 / 2 / None)

Method:
  1. Find the closest player to the ball (court coords if available, else pixel).
  2. Require that distance below `pickup_dist` to consider a pickup.
  3. Hysteresis: the candidate must persist for `min_hold_frames` frames before
     becoming the new carrier; once anointed, the carrier "sticks" until either
     (a) a different player is closer for `min_switch_frames` consecutive frames,
     or (b) the ball leaves all players (>release_dist) for `release_frames`.
  4. possession_team_id = carrier's team_id (if any). When unowned, possession
     is held by the most recent owning team for `team_hold_frames` frames
     (pass / loose-ball window).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class _PlayerInput:
    track_id: int
    pos:      Tuple[float, float]   # court (ft) if court=True, else pixel
    team_id:  Optional[int]
    in_court: bool                  # True if pos is in court (ft) units


@dataclass
class PossessionState:
    carrier_track_id: Optional[int] = None
    possession_team:  Optional[int] = None  # 1, 2, or None
    # Internal hysteresis bookkeeping
    candidate_id:     Optional[int] = None
    candidate_streak: int = 0
    release_streak:   int = 0
    last_owning_team: Optional[int] = None
    team_hold_left:   int = 0
    # Coordinate space used for the most recent distance comparison:
    # 'court' (feet), 'pixel', or None when no ball/player signal existed.
    distance_mode: Optional[str] = None


class PossessionTracker:
    """
    Stateful per-frame possession tracker.

    Defaults are tuned for ~30 fps NBA broadcast court coords (feet):
      * pickup_dist        = 4.0 ft  → must be within ~1 body-radius
      * release_dist       = 7.0 ft  → carrier loses ball if everyone moves away
      * min_hold_frames    = 2       → 2-frame confirm before promoting candidate
      * min_switch_frames  = 4       → 4-frame confirm before stealing carrier
      * release_frames     = 6       → 6 frames untouched → no carrier
      * team_hold_frames   = 30      → keep possession_team during pass (~1 sec)

    Pixel-fallback distances (when no court_pos available) use pixel units;
    callers should set use_pixel_fallback=False if they only want court-confirmed
    possession.
    """

    def __init__(
        self,
        pickup_dist: float = 6.0,
        release_dist: float = 10.0,
        min_hold_frames: int = 2,
        min_switch_frames: int = 4,
        release_frames: int = 8,
        team_hold_frames: int = 30,
        use_pixel_fallback: bool = True,
        pixel_pickup_dist: float = 90.0,
        pixel_release_dist: float = 160.0,
    ):
        self.pickup_dist = float(pickup_dist)
        self.release_dist = float(release_dist)
        self.min_hold_frames = int(min_hold_frames)
        self.min_switch_frames = int(min_switch_frames)
        self.release_frames = int(release_frames)
        self.team_hold_frames = int(team_hold_frames)
        self.use_pixel_fallback = bool(use_pixel_fallback)
        self.pixel_pickup_dist = float(pixel_pickup_dist)
        self.pixel_release_dist = float(pixel_release_dist)
        self._state = PossessionState()

    # ── public API ────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state — call on scene cuts."""
        self._state = PossessionState()

    @property
    def state(self) -> PossessionState:
        return self._state

    def update(
        self,
        ball_court: Optional[Tuple[float, float]],
        ball_pixel: Optional[Tuple[float, float]],
        players_court: Iterable[Tuple[int, Tuple[float, float], Optional[int]]],
        players_pixel: Iterable[Tuple[int, Tuple[float, float], Optional[int]]],
        ball_visible: bool,
    ) -> Tuple[Optional[int], Optional[int]]:
        """
        Args:
            ball_court:    ball (x_ft, y_ft) or None
            ball_pixel:    ball (cx, cy) px or None
            players_court: iterable of (track_id, (x_ft, y_ft), team_id)
            players_pixel: iterable of (track_id, (cx, cy)px, team_id)
            ball_visible:  whether ball was *detected* this frame (vs purely coasted)

        Returns:
            (carrier_track_id, possession_team_id) — either may be None.
        """
        st = self._state

        # Pick the coordinate space we have data for (prefer court).
        use_court = ball_court is not None
        if use_court:
            st.distance_mode = "court"
            ball_pos = ball_court
            players: List[_PlayerInput] = [
                _PlayerInput(tid, pos, team, in_court=True)
                for tid, pos, team in players_court
                if pos is not None
            ]
            pickup_d = self.pickup_dist
            release_d = self.release_dist
        elif self.use_pixel_fallback and ball_pixel is not None:
            st.distance_mode = "pixel"
            ball_pos = ball_pixel
            players = [
                _PlayerInput(tid, pos, team, in_court=False)
                for tid, pos, team in players_pixel
                if pos is not None
            ]
            pickup_d = self.pixel_pickup_dist
            release_d = self.pixel_release_dist
        else:
            # No ball signal at all → decay
            st.distance_mode = None
            self._decay_team_hold()
            return st.carrier_track_id, self._effective_team()

        if not players:
            self._decay_team_hold()
            return st.carrier_track_id, self._effective_team()

        # Find closest player to ball
        closest = min(players, key=lambda p: _dist(p.pos, ball_pos))
        closest_d = _dist(closest.pos, ball_pos)

        # A position that was not actually observed this frame is a guess: coasted
        # and interpolated ball coordinates can be tens of pixels off, and picking
        # the "nearest player" to a guess produces phantom possessions. On such a
        # frame, hold any existing claim but never start or switch one.
        if not ball_visible:
            st.candidate_id = None
            st.candidate_streak = 0
            st.release_streak = 0
            self._decay_team_hold()
            return st.carrier_track_id, self._effective_team()

        # ── case A: no current carrier ────────────────────────────────────────
        if st.carrier_track_id is None:
            if closest_d <= pickup_d:
                if st.candidate_id == closest.track_id:
                    st.candidate_streak += 1
                else:
                    st.candidate_id = closest.track_id
                    st.candidate_streak = 1
                if st.candidate_streak >= self.min_hold_frames:
                    self._promote(closest)
            else:
                st.candidate_id = None
                st.candidate_streak = 0
            self._decay_team_hold()
            return st.carrier_track_id, self._effective_team()

        # ── case B: we have a current carrier ─────────────────────────────────
        carrier = next((p for p in players if p.track_id == st.carrier_track_id), None)
        carrier_d = _dist(carrier.pos, ball_pos) if carrier is not None else float("inf")

        # B1: someone else is now closer → start switch counter
        if closest.track_id != st.carrier_track_id and closest_d <= pickup_d and closest_d < carrier_d:
            if st.candidate_id == closest.track_id:
                st.candidate_streak += 1
            else:
                st.candidate_id = closest.track_id
                st.candidate_streak = 1
            if st.candidate_streak >= self.min_switch_frames:
                self._promote(closest)
                st.release_streak = 0
                self._decay_team_hold()
                return st.carrier_track_id, self._effective_team()
        else:
            st.candidate_id = None
            st.candidate_streak = 0

        # B2: ball drifting away → start release counter
        if carrier is None or carrier_d > release_d:
            st.release_streak += 1
            if st.release_streak >= self.release_frames:
                # Loose ball — clear carrier but keep team_hold
                st.last_owning_team = st.possession_team
                st.team_hold_left = self.team_hold_frames
                st.carrier_track_id = None
                st.possession_team = None
                st.release_streak = 0
        else:
            st.release_streak = 0

        self._decay_team_hold()
        return st.carrier_track_id, self._effective_team()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _promote(self, p: _PlayerInput) -> None:
        st = self._state
        st.carrier_track_id = p.track_id
        st.possession_team = p.team_id
        if p.team_id is not None:
            st.last_owning_team = p.team_id
            st.team_hold_left = self.team_hold_frames
        st.candidate_id = None
        st.candidate_streak = 0

    def _decay_team_hold(self) -> None:
        if self._state.team_hold_left > 0:
            self._state.team_hold_left -= 1

    def _effective_team(self) -> Optional[int]:
        st = self._state
        if st.possession_team is not None:
            return st.possession_team
        if st.team_hold_left > 0:
            return st.last_owning_team
        return None
