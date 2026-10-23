"""Clock-anchor alignment for video.

A broadcast PBP clock does not map linearly to video frames across stoppages.
Callers must provide explicit anchors: (frame, period, clock_s) pairs. Between
two anchors in the same period, interpolate on the game clock. Without anchors,
return None; never invent a frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from state_layer.espn import PbpPlay


@dataclass(frozen=True)
class ClockAnchor:
    frame: int
    period: int
    clock_s: float


def frame_for_clock(period: int, clock_s: float,
                    anchors: Sequence[ClockAnchor]) -> Optional[int]:
    same = sorted(
        (a for a in anchors if a.period == period),
        key=lambda a: a.clock_s,
        reverse=True,
    )
    if not same:
        return None
    if len(same) == 1:
        return int(same[0].frame)
    # The list is descending clock. Find a bracketing pair.
    for hi, lo in zip(same, same[1:]):
        if lo.clock_s <= clock_s <= hi.clock_s:
            if hi.clock_s == lo.clock_s:
                return int(hi.frame)
            ratio = (hi.clock_s - clock_s) / (hi.clock_s - lo.clock_s)
            return int(round(hi.frame + ratio * (lo.frame - hi.frame)))
    # Outside the anchored clock range: return None. Do not fabricate a frame
    # from a nearest anchor; that would silently misalign plays.
    return None


def align_plays(plays: Iterable[PbpPlay],
                anchors: Sequence[ClockAnchor]) -> List[dict]:
    out = []
    for play in plays:
        out.append({
            **play.to_dict(),
            "frame": frame_for_clock(play.period, play.clock_s, anchors),
        })
    return out


def anchors_from_manual(clock_anchors: Iterable[dict]) -> List[ClockAnchor]:
    return [
        ClockAnchor(frame=int(a["frame"]), period=int(a["period"]),
                    clock_s=float(a["clock_s"]))
        for a in clock_anchors
    ]
