---
id: systems.footage.regimes
title: Footage regimes — broadcast, fixed gym, and handheld
layer: systems
track: systems.runtime
status: active
confidence: high
updated: 2026-10-21
sources: [arxiv-2208.08190, arxiv-2304.05170, arxiv-2503.18282, szeliski2022]
code: [data/, registry/, inference/, eval/]
prereqs: [perception.calibration.homography]
tags: [broadcast, gym, camera, domain shift, deployment]
---

# Footage regimes — broadcast, fixed gym, and handheld

**Claim.** "Basketball video" is at least three different sensing problems. A model
tuned on NBA broadcast is not a model for a high-school gym. Treat the footage regime as
a first-class field, not a note in a README.

## The three regimes

| Property | NBA broadcast | Fixed gym camera | Handheld |
|---|---|---|---|
| Camera motion | pans, zooms, cuts | none | pans, shakes |
| Resolution on players | high near the ball, low far | low everywhere | variable |
| Ball size | 15–30 px | 8–20 px | variable |
| Court visibility | partial, occluded | full | partial |
| Graphics/overlays | scorebug, replays | none | none |
| Calibration | per shot / per frame | one per session | hard |
| Audio | commentary | ambient | ambient |
| Labeled public data | DeepSportRadar, SportsMOT | scarce | scarce |
| Current repo support | keypoint model works | manual homography works | not supported |

## Why mixing regimes breaks evaluation

- A model trained on broadcast learns the scorebug, the crowd, and the arena palette. On
  a gym clip those cues vanish, and accuracy drops even when the players look the same.
- A threshold tuned on broadcast (ball `conf=0.25`) is not calibrated for gym footage.
- Tracking metrics on broadcast benefit from zoom: players are large near the action. Gym
  footage has all players small at once.
- Broadcast cuts create identity discontinuities that a fixed camera never has.

**Therefore:** every dataset, clip, and scorecard carries `footage_type`. The eval
harness already groups by it. Never report a single number across regimes without the
breakdown.

## Calibration strategy per regime

- **Broadcast:** per-shot calibration; detect cuts; recalibrate on the new shot; use line
  tracking within a shot. Do not interpolate `H` across a cut.
- **Fixed gym:** one manual or keypoint calibration per session; verify with a
  reprojection check; store the matrix with the session ID.
- **Handheld:** either stabilize first (estimate camera motion and warp to a reference
  frame) or declare it out of scope. A shaky camera plus a moving subject is a research
  project, not a sprint.

## Data-platform implications

Record with every clip:

```
footage_type: nba_broadcast | fiba_broadcast | fixed_gym | handheld | synthetic
camera: {height_ft, focal_length_mm?, position: sideline|baseline|corner|overhead}
calibration: {method: manual|keypoint|line|none, rmse_ft, p95_ft, session_id}
ruleset: nba_2025_26 | fiba_2026 | ncaa_...
source_id: registry asset_id
```

This is metadata, not video; it belongs in the registry record and the `ClipResult`
metadata. It is also the key to a meaningful per-regime scorecard.

## Deployment implications

- Broadcast analysis can be offline (a game is 2+ hours; a 10-minute turnaround is fine).
- A gym product may want near-real-time feedback (a coach on the sideline). That changes
  the latency budget, model size, and hardware.
- Edge inference favors small detectors (YOLO-n/s) and RTMPose; cloud inference allows
  larger models and multi-view fusion.
- Do not promise real-time until the offline pipeline is accurate. Accuracy first, then
  optimize.

## Repo mapping

- `data/` — raw clips and processed datasets; add `footage_type` to manifests.
- `registry/` — source records; add camera/calibration metadata.
- `inference/` — thresholds and model selection should be regime-aware.
- `eval/` — group by `footage_type`; the harness already supports it.

## Failure modes

- **Training on broadcast, demoing on a gym.** The classic domain-shift failure.
- **One threshold for all regimes.** Calibrate per regime or expose it as a parameter.
- **Ignoring the scorebug.** Broadcast graphics can be read for clock/score, but they also
  contaminate features. Crop or mask them.
- **Assuming a fixed camera on broadcast.** The README already warns; enforce it in code.
- **Handheld as an afterthought.** Declare scope explicitly.

## Open questions

1. Which regime is the product's beachhead: NBA broadcast (data-rich, crowded) or fixed
   gym (data-poor, real coaching need)?
2. Can a single detector work across regimes with per-regime thresholds, or do we need
   separate models?
3. How much does a synthetic broadcast-to-gym augmentation pipeline buy?
4. What is the minimum camera spec (height, lens, resolution) for a usable gym product?

## Proposed experiments

- **EX-110** Add `footage_type` and calibration metadata to the registry and `ClipResult`;
  backfill the current clips.
- **EX-111** Evaluate the current detector on a fixed-gym clip and a broadcast clip;
  report the per-regime gap.
- **EX-112** Test per-regime confidence thresholds vs one global threshold on the same
  benchmark.
- **EX-113** Prototype a camera-motion stabilization pass for handheld footage; decide
  go/no-go.
