---
id: basketball.analytics.shot-quality
title: Shot quality, expected points, and possession value
layer: basketball
track: basketball.analytics
status: active
confidence: medium
updated: 2026-10-21
sources: [arxiv-1408.0777, arxiv-1107.5793, arxiv-1908.05745, arxiv-2006.07513, arxiv-1609.04849, arxiv-2405.01182, nba-stats-api]
code: [analytics/, events/, visualization/]
prereqs: [basketball.tactics.spacing, measurement.evaluation]
tags: [EPV, xPTS, shot quality, value, calibration]
---

# Shot quality, expected points, and possession value

**Claim.** "Good shot" is not a vibe. It is a probability: `P(make | state)`, multiplied
by the points at stake. Once that number exists, every coaching question — shot
selection, lineup construction, play design — becomes comparable on one scale.

## The two-layer model

### Layer 1: shot quality

For a shot from state `s` (location, shot type, contest, shooter, clock, score):

```
xPTS(s) = P(make | s) · points(s)          points ∈ {2, 3}
```

Fit `P(make | s)` with logistic regression or a gradient-boosted model. Start with
interpretable features:

- distance and angle to the rim;
- shot type (layup, mid-range, 3, dunk, free throw);
- defender distance and contest angle at release;
- shooter's historical accuracy (shrunk toward league mean);
- assisted vs unassisted;
- time remaining and score margin;
- motion before the shot (catch-and-shoot vs off-the-dribble).

Evaluate with **log loss and Brier score**, not accuracy, and report a calibration curve.
A model that says 60% and is right 45% of the time is worse than a well-calibrated 50%.

### Layer 2: possession value (EPV)

Cervone et al. define the expected possession value as the expected points from the
current state, integrating over the stochastic evolution of the possession:

```
EPV(s_t) = E[ points from t to the end of the possession | s_t ]
```

At a shot, `EPV` equals `xPTS`. At a turnover, it is 0 (or the opponent's expected
points, if you model both sides). At a made basket, it is the points already scored.

The paper's key move is **multiresolution**: a macro model over possession outcomes
(shot, turnover, foul) combined with a micro model over player/ball trajectories. You do
not need the full model to get value. A coarse version is:

```
EPV_zone(s) = P(score | zone, clock bucket, defense type) · expected points
```

estimated from play-by-play, then refined with tracking features. This is a legitimate
v0 and gives the coach a number this season.

## Selection bias is the trap

You only observe shots that were taken. A model trained on observed shots learns
`P(make | shot taken)`, not `P(make | shot available)`. The difference is selection:
players pass up bad shots and take good ones, so the observed set is not random.

Consequences:

- do not claim "this player is a bad shooter" from shot quality alone;
- do not use the outcome of the shot as a feature (leakage);
- for counterfactuals ("what if he drove?"), you need a model of the decision, not just
  the shot. That is the hard research frontier, not a v1 deliverable.

The shot-selection paper formalizes the trade-off: a shot is good if its expected points
exceed the value of the alternative action. That alternative is unobserved. Be explicit
about the assumption.

## A practical build order

1. **Baseline by zone.** League-average points per shot by court zone (restricted area,
   paint, mid-range, corner 3, wing 3, top 3). No ML.
2. **Logistic xPTS.** Add distance, angle, contest, shot type. Evaluate calibration.
3. **Player shrinkage.** Hierarchical model: player effect shrunk toward league mean
   (the Bayesian shot-chart literature is the reference).
4. **Tracking features.** Defender distance, closeout speed, shot clock, action type.
5. **Possession EPV.** Zone-based value + transition context; validate against points
   per possession.
6. **Counterfactual / decision value.** Only after 1–5 are trustworthy.

## v0 baseline: NBA 2023-24 (Denver)

The first baseline is implemented in `analytics/xpts.py`, fed by
`scripts/nba_shot_ingest.py` (NBA Stats API). It uses the NBA's own shot zones and a
shrunk zone FG%, then evaluates on a **game-grouped** split.

Data: 7,279 Denver shots across 82 games; 66 train games / 16 test games.

| Zone | Share | FG% | xPTS |
|---|---:|---:|---:|
| Restricted Area | 32.2% | 0.651 | 1.303 |
| Paint (non-RA) | 18.8% | 0.488 | 0.975 |
| Mid-Range | 13.8% | 0.455 | 0.910 |
| Corner 3 | 8.7% | 0.384 | 1.153 |
| Above the Break 3 | 26.1% | 0.376 | 1.127 |
| Backcourt | 0.3% | 0.000 | 0.000 |

Held-out metrics (1,429 shots): **Brier 0.2354, log loss 0.6636**. Calibration is close
overall, but the middle bins are overconfident: predicted 0.401 vs observed 0.333, and
predicted 0.477 vs observed 0.435.

The player comparison is the argument for shrinkage. Stephen Curry 2023-24 (1,445 shots):
55.1% of his attempts are above-the-break 3s and 5.4% are corner 3s; his zone xPTS is
1.278 (restricted area), 1.005 (paint), 0.831 (mid-range), 1.615 (corner 3, only 78
attempts), and 1.187 (above the break). His held-out Brier is 0.2362 and log loss 0.6653,
but the calibration bins are worse than the team model: predicted 0.408 vs observed 0.344,
and predicted 0.429 vs observed 0.295. Player-level estimates need hierarchical shrinkage,
not a bigger model.

**What this establishes:** the zone baseline is sane, and the value ordering matches the
expected modern profile — rim > corner 3 ≈ above-break 3 > paint > mid-range.

**What it does not establish:** contest, shot type, shooter skill, or a player-level
calibrated model.

### Logistic check (EX-081)

A logistic model with distance, distance², angle, 3PT, corner-3, and restricted-area
indicators was trained on the same game-grouped split.

| Model | Denver Brier | Denver log loss | Curry Brier | Curry log loss |
|---|---:|---:|---:|---:|
| Zone baseline | 0.2354 | 0.6636 | 0.2362 | 0.6653 |
| Logistic | 0.2345 | 0.6613 | 0.2351 | 0.6630 |

The gain is ~0.001 Brier and ~0.002 log loss — within noise, and not a reason to ship.
The middle calibration bins remain overconfident (Denver predicted 0.395 vs observed
0.326; Curry predicted 0.428 vs observed 0.328). The NBA zone labels already encode most
of the distance/angle information.

**What this implies:** the next gain must come from **contest/defender distance**,
**shooter shrinkage**, or **action context**, not more location features. A player-level
model needs hierarchical shrinkage, not a bigger classifier.

## Repo mapping

- `analytics/` — the natural home for `xpts.py` and `epv.py`.
- `events/` — shot location and outcome feed the model.
- `visualization/` — shot charts and value heatmaps for coaches.
- `eval/` — calibration and log loss belong in the scorecard, not in a notebook.

## Failure modes

- **Leakage.** Using made/missed, rebound, or post-shot features to predict the shot.
- **Ignoring shooter.** A 30% shooter and a 40% shooter from the same spot are not the
  same shot; shrink, don't ignore.
- **Overfitting a small sample.** One season of one team is ~8k shots; a deep model will
  memorize. Start linear.
- **Averaging over eras.** The 3-point line moved; league efficiency changed. Use
  season-relative features.
- **Presenting EPV as truth.** It is a model with uncertainty. Show intervals.

## Open questions

1. What is the smallest feature set that beats a zone-average baseline on log loss?
2. How much does contest distance add over shot location and type?
3. Can we estimate defender distance reliably enough from broadcast tracking to matter?
4. What is the coaching interface: a single number, a shot chart, or a ranked list of
   decisions?

## Proposed experiments

- **EX-080** Build the zone-average xPTS baseline from public play-by-play; report log
  loss and calibration. — **done**.
- **EX-081** Add logistic xPTS with distance/angle/contest; compare by game-grouped CV. —
  **done**; the location-only model is neutral versus the zone baseline.
- **EX-082** Compute a coarse possession EPV from zones + clock; validate against actual
  points per possession.
- **EX-083** Produce a coach-facing shot chart with xPTS color scale; user-test with one
  coach.
- **EX-084** Contest-aware xPTS — add defender distance and closeout at release; blocked
  until defender tracking near the shooter is reliable.
