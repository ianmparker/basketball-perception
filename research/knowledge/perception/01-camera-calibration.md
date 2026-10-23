---
id: perception.calibration.homography
title: Camera calibration and court homography
layer: perception
track: perception.calibration
status: active
confidence: high
updated: 2026-10-21
sources: [szeliski2022, hartley2004, arxiv-2207.11709, arxiv-2404.08401, arxiv-2404.09807]
code: [homography/, scripts/calibrate_from_points.py]
prereqs: []
tags: [homography, DLT, RANSAC, broadcast, camera]
---

# Camera calibration and court homography

**Claim.** Every court-space number this project produces — shot distance, player speed,
possession distance, spacing — is a homography times a pixel. Calibration error is not a
separate "vision" problem; it is a multiplicative error on every downstream metric.

## The model

A pinhole camera projects a 3D point `X` to image point `x`:

```
x ~ K [R | t] X          K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

The court is planar, so set `Z = 0`. Then `[R | t]` collapses to a 3×3 matrix and

```
x ~ H X,     H = K [r1 r2 t]        (8 degrees of freedom, scale-free)
```

`H` is what `cv2.findHomography` estimates and what `homography_matrix_*.json` stores.

### DLT in one paragraph

Each correspondence `(X_i, x_i)` gives two linear equations in the 9 entries of `H`.
Stack `2N × 9` into `A`, take the SVD, and use the last right singular vector as `h`.
Four points is the minimum; more points make it least-squares. **Normalize both point
sets** (translate centroid to origin, scale to mean distance √2) before the SVD or the
system is badly conditioned. This is Hartley's normalized DLT, and it is the difference
between a calibration that works and one that silently drifts.

### RANSAC

Sample 4 correspondences, fit `H`, count inliers whose reprojection error is below a
threshold, repeat, refit on inliers. RANSAC is only as good as the inlier threshold and
the fraction of correct correspondences. With 5 manually clicked points, RANSAC has
little to work with; with 18 keypoints and a line model it has a lot.

## What to measure

Report **median, RMS, p95, and max** reprojection error in court feet, plus the number of
inliers and the conditioning of `A`. RMS alone hides the tail, and the tail is what
breaks a possession-distance calculation. `scripts/calibrate_from_points.py` already
refuses to write a calibration worse than `--max-rms-ft` (default 5 ft); that gate is
necessary but not sufficient — add p95.

A useful second test: take a stationary court point across 100 frames and measure its
court-coordinate standard deviation. That is the **temporal jitter** that a per-frame
`H` injects into speed and distance.

## Broadcast is not one camera

The current learned keypoint model fires on NBA-style broadcast and returns nothing on
gym footage. More importantly, broadcast `H` changes on every pan, tilt, zoom, and cut.
Three regimes:

| Regime | `H` behavior | Calibration strategy |
|---|---|---|
| Fixed tripod / gym | constant | one manual or keypoint calibration per session |
| Broadcast, static shot | constant within a shot | calibrate per shot; re-use while lines agree |
| Broadcast, moving camera | `H(t)` | per-frame line/keypoint tracking, or learned calibration |

Fitting one `H` across a cut is a silent catastrophe: the overlay looks plausible on one
side of the cut and is wrong on the other. `inference/schema.py` records
`homography_quality` per frame for exactly this reason; treat a quality drop as a signal
to stop emitting court-space events, not as a number to average away.

## Learned calibration: what the literature actually supports

The tempting move is to regress the 9 entries of `H` from an image. It is poorly
conditioned: small pixel changes produce large entry changes, and MSE on `H` entries does
not correspond to geometric error. The better-supported designs are:

1. **Predict court landmarks, then solve DLT.** Keypoint heatmaps (the current 18-point
   model) or line-segment maps, followed by a geometric solve. TVCalib does this with
   line/keypoint correspondences; PnLCalib optimizes points and lines jointly.
2. **Predict a court-plane representation** (segmentation or bird's-eye feature map) and
   fit the transform to a template.
3. **Train with a reprojection loss**, not a matrix loss, if the network outputs `H`
   directly.

The calibration benchmark protocol paper argues that methods are often compared on
different splits and error definitions. Adopt one protocol before claiming a learned
homography beats the manual one.

## Repo mapping

- `homography/` — manual calibration, court template, matrix I/O.
- `scripts/calibrate_from_points.py` — headless calibration from ≥4 correspondences;
  reports RMS and refuses bad fits.
- `inference/pipeline.py` — applies `H` per frame and computes `court_pos`.
- `inference/schema.py` — `HomographyQuality` (`source`, `inlier_ratio`, `rmse_ft`).

## Failure modes

- **One `H` per video on moving-camera footage.** The README already warns about this;
  the pipeline should enforce it.
- **Court feet vs meters.** FIBA documentation is metric, the repo is feet. Convert once,
  at ingest, and store the unit in the calibration file.
- **Keypoint model returns plausible but wrong points** on non-NBA courts. Gate on
  reprojection error and line agreement, not on keypoint confidence alone.
- **Averaging quality across frames** hides the frames where the event fired. Events must
  carry the quality of their own frame.

## Current blocker (2026-10-21)

The initial environment is Python 3.11 with torch 2.11.0+cu128, CUDA 12.8,
and the RTX 5070 Ti is visible. The court keypoint weights exist at
`runs/pose/court_kp_v18_resume/weights/best.pt`, and both the learned keypoint model and
`scripts/calibrate_from_points.py` can run.

The remaining blocker is **manual point selection**: a fixed camera needs 4–8 court points
read off a frame (or a GUI click). The headless path:

1. Dump a frame:
   `python scripts/calibrate_from_points.py --video <game> --preview frame.jpg --frame <n>`
2. Pick 4+ correspondences and pass `--match INDEX:PX,PY`.
3. Record median/RMS/p95 and the segment boundaries.

The learned model is the first attempt; if it finds fewer than 4 confident keypoints on
the gym/Hudl footage, fall back to manual points. **Tested on Nepal U19 at 5, 15, 25, 35,
and 45 minutes: 0 keypoints.** It does not transfer to this footage.

For a fixed camera, the fastest path is the click tool:

```powershell
python scripts/click_calibrate.py --video <game> --frame <n> --out outputs/calibration/<game>_cal.json
```

Click the landmarks in the order shown. `n` skips a landmark that is not visible, `u`
undoes, `s` solves, `q` quits. The tool maps each click to the correct `COURT_KPS_FT`
index and refuses an RMS worse than 5 ft unless `--force` is passed.

A broadcast calibration is **not** reusable here: a homography is tied to one camera
pose and focal length. For a fixed segment, one matrix is valid. If the camera pans or
zooms, calibrate a single static segment and do not apply that matrix to other segments.

See `docs/ENVIRONMENT.md` and `python scripts/check_env.py` for the working setup.

## Open questions

1. What is the minimum correspondence set for a fixed gym camera that keeps p95 error
   under 1 ft? (Current manual result: 0.696 ft RMS on 5 points.)
2. Does per-frame `H` jitter dominate the speed noise floor at 30 fps? Measure before
   tuning the tracker.
3. Can line-only calibration work on a court whose keypoints are occluded by players?
4. Which footage regime should the first learned-calibration model target — broadcast
   (data-rich, moving) or gym (data-poor, static)?

## Proposed experiments

- **EX-001** Add p95 and temporal-jitter reporting to `calibrate_from_points.py`; gate
  calibration acceptance on both.
- **EX-002** Compare manual 4-point, 8-point, and keypoint-model calibration on the same
  fixed-camera clip, using the benchmark protocol.
- **EX-003** Measure downstream speed error vs calibration error on a synthetic moving
  point with known court trajectory.
- **EX-210** Calibrate the three Phase B games (Nepal, Bellingham, Lima/Pahoa) — **blocked**
  on the torch environment and manual point selection; target median/RMS/p95 < 1 ft.
