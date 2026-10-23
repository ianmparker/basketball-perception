"""Possession accuracy metrics for Phase 15 eval harness.

Two scoring levels:

* **Team-level** (always available): for every frame inside a GT possession
  segment, did the pipeline predict the correct `possession_team_id`?
* **Jersey-level** (requires `jersey_map.per_track`): did the pipeline's
  `ball_carrier_track_id` resolve to the correct jersey number?

Both report:
- `n_frames_scored` — frames where GT specifies a carrier
- `correct`        — frames where prediction matches GT
- `wrong`          — frames where prediction is non-null but disagrees
- `missing`        — frames where prediction is null (no carrier predicted)
- `accuracy`       — correct / n_frames_scored

`missing` counts as not-correct. We separate it from `wrong` so a recall-vs-
precision tradeoff stays visible.

No external deps. Pure Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


@dataclass
class PossessionScore:
    level:           str          # "team" | "jersey"
    n_frames_scored: int
    correct:         int
    wrong:           int
    missing:         int
    per_team:        Dict[int, dict] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n_frames_scored if self.n_frames_scored else 0.0

    def to_dict(self) -> dict:
        return {
            "level":           self.level,
            "n_frames_scored": self.n_frames_scored,
            "correct":         self.correct,
            "wrong":           self.wrong,
            "missing":         self.missing,
            "accuracy":        round(self.accuracy, 4),
            "per_team":        self.per_team,
        }


def _expand_segments(segments: Sequence[dict], key: str) -> Dict[int, object]:
    """Materialise [{start_f, end_f, <key>: ...}] into a per-frame dict."""
    out: Dict[int, object] = {}
    for seg in segments:
        start = int(seg["start_f"])
        end   = int(seg["end_f"])
        val   = seg.get(key)
        if val is None:
            continue
        for f in range(start, end + 1):
            out[f] = val
    return out


def score_possession_team(
    gt_segments:    Sequence[dict],
    pred_per_frame: Sequence[Optional[int]],
) -> PossessionScore:
    """Team-level possession accuracy.

    Args:
        gt_segments:    list of `{"start_f": int, "end_f": int, "team": int}`
                        from the label file's `possession` list.
        pred_per_frame: predicted `possession_team_id` for each frame index
                        (`None` allowed). Length should be ≥ max(end_f)+1; we
                        only score frames covered by GT segments.
    """
    gt_by_frame = _expand_segments(gt_segments, "team")

    correct = wrong = missing = 0
    per_team: Dict[int, Dict[str, int]] = {}
    for f, gt_team in gt_by_frame.items():
        pt = per_team.setdefault(int(gt_team), {"n": 0, "correct": 0, "wrong": 0, "missing": 0})
        pt["n"] += 1
        pred = pred_per_frame[f] if 0 <= f < len(pred_per_frame) else None
        if pred is None:
            missing += 1
            pt["missing"] += 1
        elif int(pred) == int(gt_team):
            correct += 1
            pt["correct"] += 1
        else:
            wrong += 1
            pt["wrong"] += 1

    # Round per-team accuracy for readability
    for team_id, agg in per_team.items():
        agg["accuracy"] = round(agg["correct"] / agg["n"], 4) if agg["n"] else 0.0

    return PossessionScore(
        level="team",
        n_frames_scored=len(gt_by_frame),
        correct=correct,
        wrong=wrong,
        missing=missing,
        per_team={int(k): v for k, v in per_team.items()},
    )


def score_possession_jersey(
    gt_segments:    Sequence[dict],
    pred_track_per_frame: Sequence[Optional[int]],
    jersey_map:     Dict[str, Sequence[int]],
) -> PossessionScore:
    """Jersey-level possession accuracy.

    Args:
        gt_segments:           list of `{"start_f", "end_f", "player_jersey"}`.
                               Frames with `player_jersey: null` are skipped.
        pred_track_per_frame:  predicted `ball_carrier_track_id` per frame.
        jersey_map:            `{"<jersey_number>": [track_id, ...]}` from the
                               label file. A predicted track_id is "correct"
                               iff it is in the GT jersey's track-id list.
    """
    gt_by_frame = _expand_segments(gt_segments, "player_jersey")

    # Invert jersey_map → {track_id: jersey_number} for lookup
    track_to_jersey: Dict[int, int] = {}
    for jersey_str, track_ids in jersey_map.items():
        try:
            jersey = int(jersey_str)
        except (TypeError, ValueError):
            continue
        for tid in track_ids:
            track_to_jersey[int(tid)] = jersey

    correct = wrong = missing = 0
    for f, gt_jersey in gt_by_frame.items():
        pred_track = pred_track_per_frame[f] if 0 <= f < len(pred_track_per_frame) else None
        if pred_track is None:
            missing += 1
            continue
        pred_jersey = track_to_jersey.get(int(pred_track))
        if pred_jersey is not None and pred_jersey == int(gt_jersey):
            correct += 1
        else:
            wrong += 1

    return PossessionScore(
        level="jersey",
        n_frames_scored=len(gt_by_frame),
        correct=correct,
        wrong=wrong,
        missing=missing,
    )


__all__ = [
    "PossessionScore",
    "score_possession_team",
    "score_possession_jersey",
]
