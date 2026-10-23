---
id: perception.events.screen-pnr-result
title: Screen/PnR proposal-stage result
layer: perception
track: perception.events
status: active
confidence: high
updated: 2026-10-21
sources: [arxiv-2105.07404, arxiv-2607.21267, arxiv-2608.19646, arxiv-2404.05392, arxiv-2503.00147]
code: [events/, eval/, docs/screen-pnr-ontology.md]
prereqs: [perception.events.detection, perception.tracking.mot]
tags: [screen, pick-and-roll, proposal-stage, scorecard, negative-result]
---

# Screen/PnR proposal-stage result

**Claim.** The learned single-frame screen/PnR model is a useful **candidate generator**,
not a product detector. On 60 broadcast clips it finds most real events at permissive
thresholds but produces many false candidates, cannot assign the screener/cutter pair,
and has no interval boundaries. The correct decision is **proposal-only: keep it as stage
one, do not integrate it as a final event detector.**

## Setup

- 60 MultiSports `test_videos` clips, event starts scored at **±15 frames**.
- Screen: 59 clips / 74 ground-truth events. `c008` is excluded because it has a
  PnR-defense tube but no screen tube.
- PnR defense: 60 clips / 64 ground-truth events.
- The rule baseline is the existing pipeline detector; the learned model is a YOLO
  single-frame detector trained for 40 epochs.
- Internal validation mAP@50 was 0.515 overall (screen 0.539, PnR 0.491), but that is
  **frame-level** validation, not the event scorecard below.

## Held-out scorecard

| Event | Model | GT | Pred | TP | FP | FN | P | R | F1 | 95% CI | Noise floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Screen | Rule | 74 | 222 | 27 | 195 | 47 | 0.122 | 0.365 | 0.182 | [0.123, 0.250] | 0.124 |
| Screen | YOLO | 74 | 336 | 50 | 286 | 24 | 0.149 | 0.676 | 0.244 | [0.199, 0.293] | 0.124 |
| PnR defense | YOLO | 64 | 425 | 51 | 374 | 13 | 0.120 | 0.797 | 0.209 | [0.167, 0.256] | 0.141 |
| PnR defense | Rule | — | — | — | — | — | — | — | — | no valid baseline | — |

The PnR rule baseline is not comparable: the pipeline's `PnREvent` is an **offensive
screen + shot** event, while the benchmark label is **defensive contact**. That is an
ontology mismatch, not a model result. Do not report a rule comparison for PnR until the
definitions are aligned.

## The operating point is the decision

The headline 0.25 threshold is a proposal-stage point, not the best standalone detector.
The PR curve shows why:

| Event | Threshold | P | R | F1 |
|---|---:|---:|---:|---:|
| Screen | 0.05 | 0.051 | 0.757 | 0.095 |
| Screen | 0.25 | 0.149 | 0.676 | 0.244 |
| Screen | 0.40 | 0.307 | 0.527 | 0.388 |
| Screen | 0.50 | 0.526 | 0.405 | **0.458** |
| Screen | 0.60 | 0.765 | 0.176 | 0.286 |
| PnR | 0.05 | 0.040 | 0.859 | 0.077 |
| PnR | 0.25 | 0.120 | 0.797 | 0.209 |
| PnR | 0.40 | 0.205 | 0.703 | 0.318 |
| PnR | 0.50 | 0.266 | 0.578 | **0.364** |
| PnR | 0.60 | 0.354 | 0.359 | 0.357 |

At a permissive threshold, screen recall is 0.757 at precision 0.051; PnR-defense recall
is 0.859 at precision 0.040. That is the proposal-stage signature: most real events are
somewhere in the candidate set, but the set is large.

For a second stage, the metric is **candidate recall and candidates per clip**, not F1.
At 0.05 the model produces 336 screen candidates for 74 events (~5.6 per clip) and 425
PnR candidates for 64 events (~7.1 per clip). A pair/temporal classifier has to filter
that set without losing the real events.

## Did it beat the rule?

For screens, the point estimate is higher (F1 0.244 vs 0.182), but the paired difference
is **+0.061, 95% CI [−0.013, +0.133]**, with `P(YOLO > rule) = 0.944`. The difference is
below the **0.124 noise floor**, and the interval includes zero.

The honest statement:

> The learned model is a better candidate generator than the rule in point estimate, but
> the evidence is not sufficient to call it a statistically meaningful improvement, and
> neither version is a product detector.

## Why proposal-only

1. High recall at a permissive threshold → useful for a second stage.
2. Low precision at every usable recall point → not safe to surface as final events.
3. No pair assignment → cannot say who screened, who handled the ball, or who was
   screened. `ScreenEvent` needs a screener–cutter pair; this model has one-player tubes.
4. Start-frame only → cannot claim event boundaries or duration.
5. Broadcast-only evidence → no gym/Hudl validation.

## What would change the decision

- A pair/temporal second stage that improves precision at recall ≥ 0.6.
- Upstream tracking/team/possession quality good enough to supply the pair inputs.
- Pair labels and interval labels on a target-domain set.
- Gym/Hudl validation, not only MultiSports broadcast.

## Repo mapping

- `events/` — screen/PnR detector code and the candidate generator.
- `eval/metrics/events.py` — event P/R/F1 at tolerance.
- `docs/screen-pnr-ontology.md` — domain-expert-approved definitions, pair assignment,
  and interval semantics.
- `data/results/screen_pnr_scorecard.md` and `.json` — the scorecard.
- `data/results/multisports_v1_yolo_events.json` — predictions.
- Branch `task/multisports-screen-pnr`, commit `a4a73b6`.

## Failure modes

- **Ontology mismatch.** The PnR rule baseline measures a different event than the
  benchmark. Align definitions before any comparison.
- **Start-frame scoring.** A model can score on starts and still be unusable for clips or
  reports. Intervals are required.
- **Single-player tubes.** MultiSports labels one player; a screen is relational. A
  single-tube classifier cannot assign the pair.
- **Threshold shopping.** Reporting 0.25 hides the best-F1 point at 0.50. Report both and
  label which is for proposals and which is for standalone detection.
- **Broadcast-only evidence.** No claim about Hudl/gym footage.

## Open questions

1. What candidate budget per clip can a second stage handle (top 3, top 5, all)?
2. What is the recall of the top-k candidates per clip at the permissive threshold?
3. Can tracking/team/possession supply reliable screener–cutter–defender triples?
4. What is the event-level precision of a pair/temporal classifier at recall ≥ 0.6?
5. Do the same definitions transfer to Hudl footage without re-labeling?

## Proposed experiments

- **EX-170** Screen/PnR proposal-stage result — **done**, outcome `proposal-only`. See
  `data/results/screen_pnr_scorecard.json`.
- **EX-171** Pair/temporal screen classifier — **blocked** on tracking/team/possession
  scorecard and pair labels. Input: candidate clip + screener/cutter/defender tracks +
  interval. Output: confirmed event, pair assignment, confidence. Gate: precision at
  recall ≥ 0.6, pair accuracy, interval IoU.
