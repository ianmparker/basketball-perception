"""
analytics/passes.py — pass & interception detection (Phase 12c).

Ported from abdullahtarek/basketball_analysis pass_and_interception_detector.

Algorithm (offline, post-processing pass over a full FrameResult sequence):
  Walk frames keeping the previous "active" ball-carrier (last frame whose
  ball_carrier_track_id was non-None).  When a new carrier appears that differs
  from the previous one, classify the transition by team:

    * same team_id  → completed PASS
    * different team_id → INTERCEPTION (also called "turnover" by some)
    * unknown team on either side → ignored

  Each event is logged at the frame the *new* carrier was first detected,
  attributed to the team_id of the player who *received* the ball (i.e. the
  team that completed the pass / made the interception).

Output:
  PassInterceptionResult with:
    - passes:        List[PassEvent]
    - interceptions: List[InterceptionEvent]
    - per_frame_pass:         List[Optional[int]] of length N (team_id or None)
    - per_frame_interception: List[Optional[int]] of length N (team_id or None)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence


@dataclass
class PassEvent:
    frame_idx:     int
    team_id:       int
    from_track_id: int
    to_track_id:   int


@dataclass
class InterceptionEvent:
    frame_idx:     int
    team_id:       int          # team that intercepted
    from_track_id: int          # player who lost the ball
    to_track_id:   int          # player who intercepted


@dataclass
class PassInterceptionResult:
    passes:                  List[PassEvent]         = field(default_factory=list)
    interceptions:           List[InterceptionEvent] = field(default_factory=list)
    per_frame_pass:          List[Optional[int]]     = field(default_factory=list)
    per_frame_interception:  List[Optional[int]]     = field(default_factory=list)

    def pass_count(self, team_id: int) -> int:
        return sum(1 for p in self.passes if p.team_id == team_id)

    def interception_count(self, team_id: int) -> int:
        return sum(1 for i in self.interceptions if i.team_id == team_id)


def detect_passes_and_interceptions(frame_results: Sequence) -> PassInterceptionResult:
    """Scan a full sequence of FrameResults; return passes & interceptions.

    A "transition" is recorded at the frame where ball_carrier_track_id changes
    from `prev` to `curr` (both non-None, prev != curr).  Frames with no
    carrier (None) don't reset the previous-carrier memory — they just bridge
    the gap (e.g. ball in flight during a pass).
    """
    n = len(frame_results)
    out = PassInterceptionResult(
        per_frame_pass=[None] * n,
        per_frame_interception=[None] * n,
    )

    prev_carrier_id:  Optional[int] = None
    prev_carrier_team: Optional[int] = None

    for i, fr in enumerate(frame_results):
        carrier_id = getattr(fr, "ball_carrier_track_id", None)
        if carrier_id is None:
            # No active possession this frame → keep prev memory, move on.
            continue

        # Resolve current carrier's team from the frame's player list.
        carrier_team = _team_for_track(fr, carrier_id)

        if (prev_carrier_id is not None
                and carrier_id != prev_carrier_id
                and prev_carrier_team is not None
                and carrier_team is not None):
            if carrier_team == prev_carrier_team:
                out.passes.append(PassEvent(
                    frame_idx=i,
                    team_id=carrier_team,
                    from_track_id=prev_carrier_id,
                    to_track_id=carrier_id,
                ))
                out.per_frame_pass[i] = carrier_team
            else:
                out.interceptions.append(InterceptionEvent(
                    frame_idx=i,
                    team_id=carrier_team,
                    from_track_id=prev_carrier_id,
                    to_track_id=carrier_id,
                ))
                out.per_frame_interception[i] = carrier_team

        prev_carrier_id = carrier_id
        prev_carrier_team = carrier_team

    return out


def _team_for_track(frame_result, track_id: int) -> Optional[int]:
    """Look up team_id for a given track_id in a FrameResult."""
    for p in getattr(frame_result, "players", []) or []:
        if getattr(p, "track_id", None) == track_id:
            return getattr(p, "team_id", None)
    return None
