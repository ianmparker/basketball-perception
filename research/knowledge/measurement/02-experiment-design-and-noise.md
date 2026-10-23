---
id: measurement.experiment.noise
title: Experiment design, grouped splits, and the noise floor
layer: measurement
track: measurement.experiment
status: active
confidence: high
updated: 2026-10-21
sources: [murphy2022, arxiv-2009.07736, arxiv-1603.00831, arxiv-2304.05170]
code: [eval/, tests/]
prereqs: [measurement.evaluation.metrics]
tags: [statistics, bootstrap, leakage, power, labels]
---

# Experiment design, grouped splits, and the noise floor

**Claim.** Most "improvements" in computer vision are noise plus a leak. The job of this
note is to make the noise visible before the claim is made.

## The unit of independence

Frames are not independent. Adjacent frames are nearly identical; clips from the same
game share teams, arena, lighting, and camera; a possession is a single tactical event.
The independent unit is the **clip**, nested in the **game**, nested in the **arena**.

Rules:

- split by game, never by frame;
- if a game appears in train and test, the test score is optimistic;
- if the same possession appears in two clips, group it;
- if the same arena appears in train and test, report a per-arena breakdown.

The eval harness already groups by `game_id` and `footage_type` and bootstraps over
clips. Keep it that way. `eval/README.md` states the rule explicitly: "18,420 frames from
12 clips is 12 samples."

## Bootstrap confidence intervals

For a metric `θ` computed on `n` clips:

1. resample clips with replacement `B = 2000` times;
2. compute `θ*` on each resample;
3. report the 2.5th and 97.5th percentiles.

For a paired comparison (baseline vs candidate on the same clips), resample **clip
indices** and compute the difference on each resample. Do not compare two independent
CIs by eye; the paired interval is tighter and answers the right question.

With `n < 10` clips, the interval is wide and the bootstrap is shaky. Say so.

## The noise floor

If the benchmark has `E` labeled events, the smallest resolvable F1 difference is
roughly `1/E` in probability space, and larger after the F1 transform. The harness prints
a noise-floor statement; use it verbatim in the PR:

> "With n=NN events, differences under X F1 points are not measurable."

Do not report a 2-point F1 gain on 40 events. Do not report a 0.3-point gain at all
without a paired interval that excludes zero.

## Leakage checklist

- [ ] Same game in train and test?
- [ ] Same possession split across clips?
- [ ] Adjacent frames in both splits?
- [ ] Same arena/lighting in both splits?
- [ ] Event label derived from the outcome we are predicting?
- [ ] Threshold tuned on the test clips?
- [ ] Normalization statistics computed on the full dataset?
- [ ] Pretrained model saw the test data (foundation-model contamination)?

The last one is increasingly relevant: CLIP/DINOv2 pretraining corpora may include
basketball broadcast frames. Treat zero-shot results as a **prior**, not a clean
held-out evaluation.

## Label noise

- Measure inter-annotator agreement per class (Cohen's κ).
- Keep an adjudicated gold set; never train on it and test on it.
- If κ < 0.6, fix the definition before modeling.
- Report label counts per class and per game; a class with 3 examples is anecdotal.

## Power

Before running an experiment, ask: how many clips/events would I need to detect the
effect I expect? A rough rule: to detect a 5-point F1 difference with reasonable power,
you need on the order of hundreds of events, not dozens. If the budget is 30 events,
design a qualitative failure analysis instead of a significance test.

## Reporting template

```
Hypothesis:        <one sentence>
Metric:            <primary metric, tolerance, grouping>
Baseline:          <value, n clips, n events, CI>
Candidate:         <value, n clips, n events, CI>
Paired difference: <value, 95% CI>
Noise floor:       <harness statement>
Leakage check:     <what was held out>
Verdict:           keep / revert / needs labels
```

This is the same shape the `perception-researcher` agent is required to produce.

## Repo mapping

- `eval/score.py` — grouped scorecards, bootstrap, noise floor, shuffled-label check.
- `eval/diagnostics.py` — the statistical checks.
- `tests/test_eval_diagnostics.py`, `tests/test_eval_reproducibility.py` — the regression
  tests that keep the harness honest.
- `research/experiments/backlog.json` — hypotheses enter here before they become code.

## Failure modes

- **Frame-level confidence intervals.** Wrong unit; the interval is far too narrow.
- **p-hacking.** Trying five thresholds and reporting the best one.
- **Cherry-picked clips.** Evaluate on the clips that were easy to label.
- **Shuffled-label check ignored.** If scrambled labels score well, the matcher or the
  tolerance is leaking.
- **Comparing to a stale artifact.** The harness warns after 7 days; refresh after any
  pipeline change.

## Open questions

1. How many games do we need for a stable per-game F1 estimate?
2. Is the bootstrap over clips valid with 6–12 clips, or should we use a Bayesian
   hierarchical model?
3. How do we account for foundation-model pretraining contamination?
4. What is the right primary metric for a coaching product: event F1, or decision value?

## Proposed experiments

- **EX-100** Add a paired bootstrap comparison to `eval/score.py` for baseline vs
  candidate on the same clips.
- **EX-101** Run a power analysis on the current labeled events; publish the minimum
  detectable effect.
- **EX-102** Measure inter-annotator agreement on 50 events across two annotators.
