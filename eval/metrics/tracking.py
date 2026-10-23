"""Tracking metrics: MOTA/IDF1 plus a lightweight player-count check.

`score_tracking` follows the MOTChallenge definitions:

* per frame, match predicted boxes to ground-truth boxes by IoU using an
  optimal (Hungarian) assignment at a fixed threshold
* MOTA = 1 - (FN + FP + IDSW) / num_gt_detections
* IDF1 uses the standard global ID matching: maximise the number of frames in
  which a GT id is matched to a predicted id, then
  IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN)

`score_player_count` exists because we usually have spot-check counts
("10 players in this frame") long before we have frame-by-frame boxes. It is
cheap and catches the failure mode where the tracker silently drops half the
court — but read it carefully: the observed count is *every tracked person*,
which on broadcast footage includes referees and the bench, so an over-count is
normal. `track_fragmentation` separates "many real people" from "ID churn".

No TrackEval dependency: these are the equations, implemented directly, so the
harness stays installable in CI.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # scipy already arrives with scikit-learn
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - keeps the module importable
    linear_sum_assignment = None

Box = Tuple[float, float, float, float]  # x1, y1, x2, y2


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _optimal_pairs(scores: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(-scores)
        return [(int(r), int(c)) for r, c in zip(rows, cols) if scores[r, c] >= threshold]

    pairs: List[Tuple[int, int]] = []
    used_r, used_c = set(), set()
    for flat in np.argsort(-scores, axis=None):
        r, c = divmod(int(flat), scores.shape[1])
        if r in used_r or c in used_c or scores[r, c] < threshold:
            continue
        used_r.add(r)
        used_c.add(c)
        pairs.append((r, c))
    return pairs


@dataclass
class TrackingScore:
    mota: float
    idf1: float
    idp: float
    idr: float
    tp: int
    fp: int
    fn: int
    id_switches: int
    num_gt: int
    num_pred: int

    def to_dict(self) -> dict:
        return {
            "mota": round(self.mota, 4),
            "idf1": round(self.idf1, 4),
            "idp": round(self.idp, 4),
            "idr": round(self.idr, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "id_switches": self.id_switches,
            "num_gt": self.num_gt,
            "num_pred": self.num_pred,
        }


def score_tracking(
    gt: Dict[int, List[Tuple[int, Box]]],
    pred: Dict[int, List[Tuple[int, Box]]],
    *,
    iou_threshold: float = 0.5,
) -> TrackingScore:
    """Score tracking against per-frame (track_id, box) ground truth."""
    frames = sorted(set(gt) | set(pred))
    tp = fp = fn = id_switches = 0
    num_gt = sum(len(gt.get(f, [])) for f in frames)
    num_pred = sum(len(pred.get(f, [])) for f in frames)

    last_match: Dict[int, int] = {}
    pair_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    for frame in frames:
        g = gt.get(frame, [])
        p = pred.get(frame, [])
        if not g or not p:
            fn += len(g)
            fp += len(p)
            continue

        scores = np.zeros((len(g), len(p)), dtype=np.float64)
        for i, (_, gb) in enumerate(g):
            for j, (_, pb) in enumerate(p):
                scores[i, j] = iou(gb, pb)

        pairs = _optimal_pairs(scores, iou_threshold)
        matched_g = {i for i, _ in pairs}
        matched_p = {j for _, j in pairs}

        tp += len(pairs)
        fn += len(g) - len(matched_g)
        fp += len(p) - len(matched_p)

        for i, j in pairs:
            gt_id, pred_id = g[i][0], p[j][0]
            pair_counts[(gt_id, pred_id)] += 1
            if gt_id in last_match and last_match[gt_id] != pred_id:
                id_switches += 1
            last_match[gt_id] = pred_id

    mota = 1.0 - (fn + fp + id_switches) / num_gt if num_gt else 0.0

    idtp = 0
    if pair_counts:
        gt_ids = sorted({k[0] for k in pair_counts})
        pred_ids = sorted({k[1] for k in pair_counts})
        counts = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
        gi = {gid: i for i, gid in enumerate(gt_ids)}
        pi = {pid: j for j, pid in enumerate(pred_ids)}
        for (gid, pid), n in pair_counts.items():
            counts[gi[gid], pi[pid]] = n
        idtp = int(sum(counts[r, c] for r, c in _optimal_pairs(counts, 1.0)))

    idfn = num_gt - idtp
    idfp = num_pred - idtp
    denom = 2 * idtp + idfp + idfn
    idf1 = (2 * idtp / denom) if denom else 0.0
    idp = (idtp / (idtp + idfp)) if (idtp + idfp) else 0.0
    idr = (idtp / (idtp + idfn)) if (idtp + idfn) else 0.0

    return TrackingScore(
        mota=mota, idf1=idf1, idp=idp, idr=idr,
        tp=tp, fp=fp, fn=fn, id_switches=id_switches,
        num_gt=num_gt, num_pred=num_pred,
    )


@dataclass
class PlayerCountScore:
    checks: int
    exact: int
    total_abs_error: int
    worst: Optional[int]
    # (frame, expected_players, observed_persons) per check. The direction of the
    # error matters: over-counting means referee/bench/ghost tracks, under-counting
    # means dropped players. An absolute error hides which one happened.
    observations: List[Tuple[int, int, int]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.exact / self.checks if self.checks else 0.0

    @property
    def mean_abs_error(self) -> float:
        return self.total_abs_error / self.checks if self.checks else 0.0

    def to_dict(self) -> dict:
        return {
            "checks": self.checks,
            "exact": self.exact,
            "accuracy": round(self.accuracy, 4),
            "mean_abs_error": round(self.mean_abs_error, 3),
            "worst_abs_error": self.worst,
            "observations": [
                {"frame_idx": f, "expected_players": e, "observed_persons": o}
                for f, e, o in self.observations
            ],
        }


def score_player_count(
    checks: Sequence[Tuple[int, int]],
    actual_by_frame: Dict[int, int],
) -> PlayerCountScore:
    """Compare spot-check player counts against tracked persons in the frame.

    ``actual_by_frame`` counts *every tracked person*, not just players on court:
    the detector has no notion of "on court". On broadcast footage the observed
    count therefore includes referees, bench and coaches, so an over-count is
    expected and is not by itself a tracking failure. Use ``observations`` to see
    the direction, and ``track_fragmentation`` to tell real people from fragments.
    """
    exact = 0
    total_err = 0
    worst: Optional[int] = None
    used = 0
    observations: List[Tuple[int, int, int]] = []

    for frame, expected in checks:
        if frame not in actual_by_frame:
            continue
        used += 1
        observed = int(actual_by_frame[frame])
        err = abs(observed - int(expected))
        total_err += err
        worst = err if worst is None else max(worst, err)
        observations.append((int(frame), int(expected), observed))
        if err == 0:
            exact += 1

    return PlayerCountScore(
        checks=used,
        exact=exact,
        total_abs_error=total_err,
        worst=worst,
        observations=observations,
    )


def track_fragmentation(
    per_frame_ids: Sequence[Sequence[int]],
) -> Dict[str, int]:
    """Summarise track-ID churn.

    A person that keeps its ID for the whole clip is one long track. Many short
    tracks in a scene with few people means ID switches or ghost detections.
    """
    life: Dict[int, int] = {}
    for ids in per_frame_ids:
        for track_id in set(ids):
            life[track_id] = life.get(track_id, 0) + 1

    n_frames = len(per_frame_ids) or 1
    return {
        "unique_tracks": len(life),
        "long_tracks": sum(1 for n in life.values() if n >= n_frames * 0.5),
        "fragments": sum(1 for n in life.values() if n < n_frames * 0.1),
    }


__all__ = [
    "iou",
    "TrackingScore",
    "score_tracking",
    "PlayerCountScore",
    "score_player_count",
    "track_fragmentation",
]
