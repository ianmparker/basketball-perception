---
id: measurement.evaluation.metrics
title: Metrics and scorecards for perception, tracking, and events
layer: measurement
track: measurement.evaluation
status: active
confidence: high
updated: 2026-10-21
sources: [arxiv-2009.07736, arxiv-1603.00831, arxiv-2304.05170, arxiv-2208.08190]
code: [eval/, eval/metrics/, docs/]
prereqs: [measurement.experiment.noise]
tags: [metrics, HOTA, IDF1, MOTA, F1, scorecard]
---

# Metrics and scorecards for perception, tracking, and events

**Claim.** A metric is a decision rule. If you cannot say what action changes when the
number moves, it is decoration. Pick one primary metric per task, report its sample size
and grouping, and keep the rest as diagnostics.

## Metric families

### Detection

- **Precision / recall / F1 at an IoU threshold** — the operating-point view.
- **AP / mAP** — threshold-independent; good for comparing detectors, less useful for a
  fixed pipeline setting.
- **Recall@k px for the ball** — the right metric for a tiny object where IoU is unstable.

Report the operating point actually used by the pipeline, not only mAP.

### Tracking

- **MOTA** = `1 - (FN + FP + IDSW) / GT`. Detection-dominated; can hide association.
- **IDF1** = `2·IDTP / (2·IDTP + IDFP + IDFN)`. Identity-preserving recall.
- **HOTA** = `sqrt(DetA · AssA)` integrated over localization thresholds. Separates
  detection and association; use it as the primary tracking metric.

### Events

- **Event P/R/F1 with a temporal tolerance** — the repo's primary event metric. Report
  the strict tolerance first (5 frames), then 15/30 as a localization sweep.
- **mAP@δ** — if we adopt a spotting model; not directly comparable to event F1.
- **Duplicate rate** — a precision failure that a generous tolerance hides.

### Calibration and geometry

- **Reprojection median / RMS / p95 / max in feet**, plus inlier count.
- **Temporal jitter** of a static court point in court coordinates.
- **Downstream sensitivity**: how much does 1 ft of calibration error change speed or
  shot distance?

### Assignment (team, possession, role)

- Accuracy and confusion matrix, per game.
- **Abstention rate**: a model that says "unknown" is often more useful than a confident
  wrong label.
- **Adjusted Rand index** for clustering, if no labels.

### Value models (xPTS, EPV)

- **Log loss** and **Brier score** for probabilities.
- **Calibration curve** and expected calibration error.
- **Out-of-sample by game/season**, never by shot.

## One scorecard, four views

```
clip view      → did this clip regress?
game view      → is the result consistent within a game?
footage view   → does it hold on gym vs broadcast?
macro view     → per-class F1, so a rare class cannot hide
```

The harness already produces the clip and group views. Add the per-class table and the
regime breakdown.

## Primary vs diagnostic

| Task | Primary | Diagnostics |
|---|---|---|
| Ball | recall@15 px + precision at operating point | frame coverage, fragments, coast rate |
| Players | mAP@50 + count error | per-class AP, role confusion |
| Tracking | HOTA | MOTA, IDF1, ID switches, track churn |
| Events | event F1 @ 5 frames | P/R, F1 @ 15/30, duplicates |
| Calibration | p95 reprojection ft | median, RMS, jitter, inliers |
| Team ID | accuracy per game | confusion, abstention |
| xPTS | log loss + calibration | Brier, AUC (secondary) |

Frame coverage is never a primary metric. It is a diagnostic that explains a primary.

## Regression gates

A change ships only if:

1. the primary metric does not drop beyond its paired confidence interval;
2. no per-class F1 drops by more than the noise floor;
3. the per-regime breakdown does not hide a collapse in one regime;
4. the tests pass and the scorecard is attached.

If a change improves the average by hurting a regime, that is a product decision, not an
automatic merge.

## Repo mapping

- `eval/score.py` — driver; emits the per-clip table, localization sweep, noise floor,
  person-count checks, track churn, possession mode, shuffled-label check, grouped
  scorecards.
- `eval/metrics/events.py` — event P/R/F1 at tolerance.
- `eval/metrics/detection.py` — event + team assignment scores.
- `eval/metrics/tracking.py` — MOTA, IDF1, ID switches.
- `eval/metrics/possession.py` — team/jersey possession accuracy.
- `eval/results/` — per-run output (gitignored).

## Failure modes

- **Metric shopping.** Reporting the tolerance that looks best. Freeze the primary.
- **Averaging over regimes.** A broadcast win can hide a gym collapse.
- **Ignoring n.** An F1 on 12 events is an anecdote.
- **MOTA as the only tracking number.** It rewards detection and hides ID swaps.
- **Uncalibrated probabilities.** A 0.9 that means 0.6 is not a probability.

## Open questions

1. What is the primary metric for the coach product: event F1 or decision value?
2. Should tracking be scored on players only, or players + referees + bench separately?
3. How do we score an event that is correct but attributed to the wrong player?
4. What is the minimum per-class count before a class enters the scorecard?

## Proposed experiments

- **EX-120** Add a per-class event table and a per-regime breakdown to `eval/score.py`.
- **EX-121** Add HOTA and a role-aware person count to the tracking scorecard.
- **EX-122** Define the ship gate as a machine-readable JSON and enforce it in CI.
