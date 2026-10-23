---
id: basketball.tactics.spacing
title: Spacing, coverages, and tactical inference from tracking
layer: basketball
track: basketball.tactics
status: active
confidence: medium
updated: 2026-10-21
sources: [arxiv-1405.0231, arxiv-1612.05502, goldsberry2019, taylor2016]
code: [events/detect_screens.py, analytics/, visualization/]
prereqs: [perception.tracking.mot, perception.calibration.homography]
tags: [spacing, pick-and-roll, drop, switch, zone, matchup]
---

# Spacing, coverages, and tactical inference from tracking

**Claim.** Tactics are **relationships between players over time**, not labels on a
single frame. A 2-3 zone and a man defense can look identical for one frame; the
difference is who is responsible for which space across the possession.

## What tracking gives you

After homography and tracking, a possession is a set of court-space trajectories:

```
state_t = { (x_i, y_i, vx_i, vy_i, team_i, role_i) for each visible player } + ball
```

Everything below is a function of that state and its history. No new pixels required.

## Spacing

Useful descriptive metrics:

- **Convex hull area** of the five offensive players (floor spacing).
- **Mean pairwise distance** (a robust proxy for hull area).
- **Paint touches**: ball or player inside the key.
- **Rim pressure**: drives within 4 ft of the rim.
- **Closeout distance**: defender distance to the receiver at catch.
- **Corner occupancy**: are both corners filled (5-out) or one (4-out-1-in)?
- **Slot/wing/top alignment**: a coarse formation label.

Spacing is not good or bad by itself. Five-out maximizes driving lanes but can leave the
rim unprotected on a miss; two-big lineups trade spacing for offensive rebounding. The
metric must be evaluated against the team's intent, not against a universal target.

## Defensive coverages

| Coverage | Observable signature | Confounder |
|---|---|---|
| Man | each defender has a stable nearest offensive assignment | switches look like man |
| Drop | screener's defender sits between the roller and the rim, below the screen level | deep drop vs hedge |
| Hedge / show | screener's defender jumps above the screen level, then recovers | brief hedge vs blitz |
| Switch | two defenders exchange assignments after the screen | accidental cross-match |
| Ice / down | ball handler's defender forces toward the baseline; screener's defender stays high | side of the court |
| Trap / blitz | two defenders within ~6 ft of the ball handler | help defense |
| 2-3 zone | two guards high, three players low; no stable man assignments | matchup zone |
| 1-3-1 | one point, three wings, one baseline | diamond press |
| Full-court press | defenders above halfcourt at inbound | late-game defense |

### Man vs zone, quantitatively

Build the **assignment matrix** `A[i, j] = distance(off_i, def_j)` and solve a Hungarian
assignment each frame. Then compute:

- **assignment stability**: how often the same defender is matched to the same offensive
  player across the possession;
- **nearest-defender distance distribution**: man defense has a tight, stable mode; zone
  has a broader distribution and more switching;
- **paint occupancy**: zone defenders cluster in the paint without a specific matchup.

A classifier on these features is more defensible than a single-frame CNN, and it
degrades gracefully when tracking is sparse.

### Detecting a screen, then a coverage

`events/detect_screens.py` already looks for a stationary screener and a nearby cutter.
Extend the event with the **screener's defender's behavior**:

```
screen event + defender depth relative to screen level + recovery direction
    → drop | hedge | switch | ice | blitz
```

This is a small, interpretable extension of an existing detector. It is also the input a
coach actually wants: "they dropped, our roller was open, we didn't throw it."

## Matchups and defensive skill

Franks et al. show that defensive impact is spatially structured: a defender's value
depends on where they deter shots, not just on steals and blocks. Practical implications:

- assign each defender a **spatial influence map** (shots deterred/allowed by zone);
- evaluate help defense by the distance and timing of rotations, not by steals;
- beware of lineup confounding: a good defender on a bad unit can look bad.

Defensive player classification (the 2016 paper) is an early prototype of role labels
from tracking. Use it as a feature idea, not a ground truth.

## Repo mapping

- `events/detect_screens.py`, `screen_detector_v2.py` — screen detection.
- `events/pnr_detector.py` — pick-and-roll.
- `analytics/` — pass/possession and player metrics.
- `visualization/visualize_court_2d.py` — the canvas for spacing/coverage overlays.
- `visualization/transform_trajectories.py` — court-space trajectories and heatmaps.

## Failure modes

- **Classifying coverage from one frame.** Use a possession-level window (≥ 1 s).
- **Ignoring who has possession.** Offense/defense assignment is a prerequisite; a
  mislabeled possession flips every tactical label.
- **Camera angle bias.** A sideline camera compresses the far side; court coordinates fix
  the geometry but not the occlusion.
- **Small samples.** One possession is not a tendency. Report per-game and per-lineup
  counts with uncertainty.
- **Confusing a switch with a mismatch.** A switch is an event; a mismatch is a
  persistent size/skill advantage. They need different labels.

## Open questions

1. What is the minimum tracking quality (players visible, ID stability) for a coverage
   classifier to beat chance?
2. Can we detect "drop vs hedge" from the screener's defender alone, or do we need the
   ball handler's path?
3. How do we label ground truth for coverages without asking a coach to annotate every
   possession?
4. Which spacing metric best predicts a coaching outcome (shot quality, turnover rate)?

## Proposed experiments

- **EX-070** Build a possession-level matchup assignment from Hungarian + temporal
  smoothing; measure assignment stability on 20 possessions.
- **EX-071** Label 50 pick-and-roll possessions as drop/hedge/switch/ice/blitz; train a
  small classifier on the features above.
- **EX-072** Compute convex hull area and paint touches per possession; correlate with
  shot quality (xPTS) once `basketball.analytics.shot-quality` has a baseline.
