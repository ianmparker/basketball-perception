---
id: perception.ball.tracking
title: Ball detection and trajectory recovery
layer: perception
track: perception.tracking
status: active
confidence: high
updated: 2026-10-21
sources: [arxiv-2110.06864, arxiv-2003.12039, arxiv-2406.19655, fontanella2006]
code: [scripts/detect_ball_positions.py, tracking/, inference/pipeline.py]
prereqs: [perception.calibration.homography]
tags: [ball, detection, tracking, occlusion, physics]
---

# Ball detection and trajectory recovery

**Claim.** The ball is the smallest, fastest, most-occluded object in the scene and the
one every event depends on. The current pipeline's headline number — 35.1% frame
coverage at `conf=0.25` — is a **detection coverage** number, not recall. Treating it as
recall overstates the system and hides the real failure: the ball is missing when it
matters (release, rim, rebound).

## Why the ball is hard

At full-court framing a 1920 px frame spans ~94 ft, so 1 ft ≈ 20 px. A size-7 ball is
~0.78 ft across: **~16 px**. At half-court framing it is ~30 px. Add:

- motion blur: 20 mph = 29 ft/s; at 1/60 s exposure the ball moves ~0.5 ft ≈ 10 px;
- occlusion by players, the rim, and the backboard;
- broadcast cuts and camera motion;
- similar-colored objects (skin, wood floor, orange shoes, crowd).

This is the canonical small-object, long-tail detection problem. Focal loss and
multi-scale feature pyramids exist for this regime; a detector trained on mostly static
players will not learn it by accident.

## Detection, tracking, and the difference

A per-frame detector gives a set of candidate boxes. A tracker imposes identity and
continuity. For the ball, three designs are on the table:

1. **Detect-then-associate** (current): YOLO per frame, then pick/smooth. Cheap, but
   recall collapses when the detector misses a stretch.
2. **Track-before-detect**: propagate a motion model and search a local window; a weak
   detection can re-acquire a track. This is where optical flow and a Kalman prior earn
   their keep.
3. **Learned tracking**: a recurrent/transformer model over frames (TrackNet-style).
   Higher ceiling, needs dense labels and compute.

ByteTrack's contribution is relevant even for one object: its second association stage
matches **low-score boxes** to tracks that are still alive. Our detector at `conf=0.25`
throws those boxes away. A ball-specific tracker should keep them.

## Motion model

In court coordinates the ball is a projectile:

```
p(t) = p0 + v0 t + 0.5 g t²,     g = (0, 0, -32.17 ft/s²)
```

But a monocular floor homography cannot see height. Mapping a ball pixel through `H`
gives the **intersection of the camera ray with the floor**, not the ball's true 3D
position. For a ball at height `h`, that floor point is displaced away from the camera,
and the error grows with `h` and with distance from the camera. Consequences:

- ball court coordinates during flight are a *shadow* projection, biased long;
- shot distance should be measured from the **shooter's** court position, not the ball's
  projected floor point;
- a shot arc in court coordinates is not a clean parabola unless height is modeled.

Use pixel-space dynamics for tracking, and court-space dynamics only for floor-level
events (player positions, rebound landing). If height matters, estimate it with a second
view, a known rim height (10 ft), or a learned monocular height head — do not pretend the
floor homography gives it.

## Smoothing without inventing

The two-pass smoother (argmax-confidence picker + pandas interpolation) improves the
visual, but interpolation across a long gap is fabrication. Gate it:

- maximum gap (frames) that may be bridged;
- maximum implied speed (ft/s) given the gap;
- no interpolation across a detected shot release or a scene cut;
- mark interpolated samples in the output (`BallState.interpolated` already exists) and
  exclude them from metrics.

A trajectory that is smooth and wrong is worse than a trajectory with holes, because
downstream event detectors will trust it.

## What to measure

1. **Per-frame recall at a distance tolerance** against labeled ball positions, with a
   visibility mask. Report by footage type and by game, not by frame count alone.
2. **Precision / false-positive rate** at the operating threshold, with examples.
3. **Track continuity**: longest gap, number of fragments, ID switches for the ball track.
4. **Event impact**: does better ball recall actually improve shot F1? That is the only
   number that ships.

Frame coverage is a diagnostic. It is not a benchmark.

## Repo mapping

- `scripts/detect_ball_positions.py` — YOLO per-frame extraction, confidence sweep.
- `scripts/compare_ball_detections.py` — threshold comparison; useful for operating-point
  selection, not for recall claims.
- `tracking/` — ByteTrack integration and Kalman motion model.
- `inference/pipeline.py` — two-pass smoothing, `BallState.coasted` / `interpolated`.
- `scripts/export_e2e_training_data.py` — `(frame, ball pixel) → court` pairs; note the
  floor-projection bias above when using these as targets.

## Failure modes

- **Threshold tuned on one clip.** The 0.25 operating point came from a single clip;
  re-select it on the grouped benchmark, and report the precision/recall curve.
- **Coasting mistaken for tracking.** `coasted=True` frames are predictions, not
  observations. Count them separately.
- **Interpolation across a pass.** A straight line through a pass looks like a shot
  trajectory to a naive detector.
- **Ball color assumptions.** HSV orange boosting helps in one arena and fails on a
  red-uniform court or under different white balance.
- **Confusing ball height with court position.** See above; this is the most expensive
  silent error in the pipeline.

## Open questions

1. What is the recall ceiling of the current detector on labeled ball positions, by
   visibility bucket (clear / partially occluded / fully occluded)?
2. Does adding low-score ByteTrack association recover enough ball frames to move shot
   F1, or does it add false tracks?
3. Can a gravity-constrained smoother reject physically impossible interpolations better
   than a max-speed gate?
4. Is a ball-specific ReID/embedding needed, or is motion continuity sufficient given
   there is only one ball?

## Proposed experiments

- **EX-010** Label ball positions on 3 clips (clear/occluded/out-of-frame) and report
  recall@5px, recall@15px, and precision at the current threshold.
- **EX-011** Add a low-score association pass for the ball only; compare shot F1 and
  false-positive events.
- **EX-012** Replace linear interpolation with a constant-acceleration model plus a
  gravity prior; measure event F1 and impossible-trajectory rate.
