"""Eval diagnostics: noise floor, leakage check, and grouped scorecards.

These are the cheap, mechanical guards from the CS 7641 upgrade list:

* ``resolvable_f1_delta`` (B-17) answers "given this many labeled events, how
  big must an F1 difference be before it means anything?"
* ``shuffle_sanity_check`` (B-08) scrambles the labels and re-scores.  If the
  harness still finds signal, the metric or the split is broken.
* ``summarize_by_group`` (B-03) aggregates over clips, not frames, and can
  group by game or footage type.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from eval.metrics import score_events_frame_tolerance


def resolvable_f1_delta(
    n_events: int,
    f1: float = 0.5,
    z: float = 1.96,
) -> Optional[float]:
    """Smallest measurable F1 difference, as a proportion (not percentage).

    Uses the standard error of the difference between two independent F1
    estimates at the given event count:

        SE_diff = sqrt(2 · F1 · (1 − F1) / n)

    A 95% interval is ``z · SE_diff``.  This is deliberately conservative:
    it says only that smaller differences cannot be distinguished from noise
    with this many labels.
    """
    if n_events <= 0:
        return None
    p = float(f1)
    # A 0.0 or 1.0 score carries no usable variance estimate (e.g. no
    # predictions at all).  Use the maximum-entropy fallback instead of
    # pretending a perfect/empty score has a tight interval.
    if p <= 0.0 or p >= 1.0:
        p = 0.5
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    se_diff = math.sqrt(2.0 * p * (1.0 - p) / float(n_events))
    return z * se_diff


@dataclass
class ShuffleResult:
    """Outcome of one shuffled-label re-score."""

    event_type: str
    n_gt: int
    observed_f1: float
    shuffled_f1: float
    status: str                 # 'ok' | 'warning' | 'skipped'
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "n_gt": self.n_gt,
            "observed_f1": round(self.observed_f1, 4),
            "shuffled_f1": round(self.shuffled_f1, 4),
            "status": self.status,
            "reason": self.reason,
        }


def shuffle_sanity_check(
    predicted_frames: Sequence[int],
    ground_truth_frames: Sequence[int],
    tolerance_f: int,
    *,
    event_type: str = "event",
    seed: int = 0,
    min_events: int = 5,
    warn_f1: float = 0.20,
    n_permutations: int = 20,
    frame_limit: Optional[int] = None,
) -> ShuffleResult:
    """Re-score against scrambled labels and flag suspiciously high F1.

    Labels are replaced by random frame indices drawn uniformly from the clip,
    keeping the same number of events.  (A permutation of the labels would
    preserve the *set* of frames, so a detector predicting that set would still
    score 1.0 — a useless check.)  With too few events the scramble cannot
    separate signal from chance, so the check is skipped rather than reported
    as a pass.
    """
    gt = list(int(f) for f in ground_truth_frames)
    pred = list(int(f) for f in predicted_frames)
    observed = score_events_frame_tolerance(pred, gt, tolerance_f=tolerance_f)

    if len(gt) < min_events:
        return ShuffleResult(
            event_type=event_type,
            n_gt=len(gt),
            observed_f1=observed.f1,
            shuffled_f1=0.0,
            status="skipped",
            reason=f"needs >={min_events} GT events to scramble",
        )

    limit = int(frame_limit) if frame_limit and frame_limit > 0 else (
        max(gt + pred) + 1 if (gt or pred) else 1
    )
    rng = random.Random(seed)
    shuffled_f1_values = []
    for _ in range(max(1, int(n_permutations))):
        shuffled = [rng.randrange(limit) for _ in gt]
        shuffled_f1_values.append(
            score_events_frame_tolerance(
                pred, shuffled, tolerance_f=tolerance_f
            ).f1
        )
    shuffled_f1 = sum(shuffled_f1_values) / len(shuffled_f1_values)

    status = "warning" if shuffled_f1 > warn_f1 else "ok"
    reason = (
        f"scrambled F1 {shuffled_f1:.2f} > {warn_f1:.2f} "
        f"(mean of {len(shuffled_f1_values)} draws); "
        "check for leakage or an over-loose matcher"
        if status == "warning" else ""
    )
    return ShuffleResult(
        event_type=event_type,
        n_gt=len(gt),
        observed_f1=observed.f1,
        shuffled_f1=shuffled_f1,
        status=status,
        reason=reason,
    )


def bootstrap_mean_ci(
    values: Iterable[float],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI for the mean.  Returns (None, None) for n<2."""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.quantile(means, (1.0 - ci) / 2.0))
    hi = float(np.quantile(means, 1.0 - (1.0 - ci) / 2.0))
    return lo, hi


@dataclass
class GroupSummary:
    """Clip-level aggregate for one game or footage type."""

    group: str
    n_clips: int
    mean_f1: float
    ci_low: Optional[float]
    ci_high: Optional[float]


def summarize_by_group(
    per_clip: Dict[str, object],
    attr: str,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> List[GroupSummary]:
    """Aggregate per-clip average event F1, grouped by ``attr``.

    ``per_clip`` maps clip stem → EvalReport-like object with ``average_event_f1``
    and the grouping attribute.  Error bars are over clips, never frames.
    """
    groups: Dict[str, List[float]] = {}
    for report in per_clip.values():
        key = str(getattr(report, attr, "") or "unknown")
        groups.setdefault(key, []).append(float(report.average_event_f1()))

    out: List[GroupSummary] = []
    for key in sorted(groups):
        values = groups[key]
        lo, hi = bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
        out.append(
            GroupSummary(
                group=key,
                n_clips=len(values),
                mean_f1=sum(values) / len(values),
                ci_low=lo,
                ci_high=hi,
            )
        )
    return out
