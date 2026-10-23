"""Event precision / recall / F1 at frame tolerance.

Greedy 1-1 matching on |pred_frame − gt_frame|: sort all candidate (pred, gt)
pairs by frame distance, accept any pair under tolerance where neither side is
already matched. Matches SoccerNet action-spotting eval's spirit (point events
matched within a temporal window), simplified for our smaller label set — we
report P/R/F1 at a single tolerance rather than mAP over a sweep of tolerances.
That's enough to detect regressions; swap in full AP-at-δ once we have 10+
labeled clips.

No external dependencies. Pure Python + frame numbers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple


@dataclass
class EventScore:
    event_type: str
    n_gt:        int
    n_pred:      int
    tp:          int
    fp:          int
    fn:          int
    tolerance_f: int
    matches: List[Tuple[int, int, int]] = field(default_factory=list)  # (gt_frame, pred_frame, abs_distance)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.n_gt if self.n_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "n_gt":       self.n_gt,
            "n_pred":     self.n_pred,
            "tp":         self.tp,
            "fp":         self.fp,
            "fn":         self.fn,
            "precision":  round(self.precision, 4),
            "recall":     round(self.recall, 4),
            "f1":         round(self.f1, 4),
            "tolerance_f": self.tolerance_f,
        }


def score_events(
    gt_frames:   Sequence[int],
    pred_frames: Sequence[int],
    tolerance_f: int,
    event_type:  str = "event",
) -> EventScore:
    """Greedy 1-1 matching at frame tolerance.

    Args:
        gt_frames:    ground-truth event frames.
        pred_frames:  predicted event frames.
        tolerance_f:  abs(pred - gt) ≤ this → match.
        event_type:   label for the returned score.
    """
    gt_list   = sorted(int(f) for f in gt_frames)
    pred_list = sorted(int(f) for f in pred_frames)

    # All candidate pairs under tolerance, sorted by distance
    candidates: List[Tuple[int, int, int]] = []
    for gi, gf in enumerate(gt_list):
        for pi, pf in enumerate(pred_list):
            d = abs(pf - gf)
            if d <= tolerance_f:
                candidates.append((d, gi, pi))
    candidates.sort()

    matched_gt:   set = set()
    matched_pred: set = set()
    matches: List[Tuple[int, int, int]] = []
    for d, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        matches.append((gt_list[gi], pred_list[pi], d))

    tp = len(matches)
    fn = len(gt_list)   - tp
    fp = len(pred_list) - tp

    return EventScore(
        event_type=event_type,
        n_gt=len(gt_list),
        n_pred=len(pred_list),
        tp=tp, fp=fp, fn=fn,
        tolerance_f=tolerance_f,
        matches=sorted(matches),
    )
