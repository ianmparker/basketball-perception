---
id: perception.tracking.mot
title: Multi-object tracking and identity in basketball
layer: perception
track: perception.tracking
status: active
confidence: high
updated: 2026-10-21
sources: [arxiv-1602.00763, arxiv-2110.06864, arxiv-2206.14651, arxiv-2203.14360, arxiv-2009.07736, arxiv-2304.05170, arxiv-2406.19655]
code: [tracking/, eval/metrics/tracking.py, scripts/export_mot.py]
prereqs: [perception.detection, perception.calibration.homography]
tags: [MOT, ByteTrack, HOTA, IDF1, ReID, occlusion]
---

# Multi-object tracking and identity in basketball

**Claim.** Tracking is the identity layer. A coach cannot say "the weak-side defender
rotated late" if the same player is track 17 in one possession and track 42 in the next.
Detection accuracy sets the ceiling; association quality decides whether the ceiling is
usable.

## Formulation

Given detections `D_t = {d_t^i}` and tracks `T_{t-1} = {tr_{t-1}^j}`, build a cost matrix
and solve an assignment:

```
C[i, j] = λ_iou · (1 - IoU(d_t^i, tr_{t-1}^j))
        + λ_app · cosine_dist(embed(d_t^i), embed(tr_{t-1}^j))
        + λ_motion · Mahalanobis(d_t^i, predict(tr_{t-1}^j))
```

The Hungarian algorithm (Jonker-Volgenant in practice) solves the linear assignment.
Unmatched detections start tracks; unmatched tracks coast on the motion model and die
after `max_age`.

### SORT family

- **SORT**: Kalman filter on `[u, v, s, r, u̇, v̇, ṡ]` + IoU + Hungarian. Strong baseline,
  no appearance.
- **DeepSORT**: adds a CNN appearance embedding and a cosine gallery. Helps re-acquire
  after occlusion; hurts when jerseys look alike.
- **ByteTrack**: first associates high-score boxes, then uses **low-score boxes** for the
  remaining tracks. The insight is that a low-confidence box is often a real occluded
  object, not noise. This matters for our sparse ball detections and for players behind
  screens.
- **BoT-SORT**: adds global motion compensation (GMC) — warps the previous frame's boxes
  by estimated camera motion before association — plus a better Kalman state and ReID.
  Broadcast footage pans and zooms constantly; GMC is not optional there.
- **OC-SORT**: observation-centric updates; reduces Kalman drift during long occlusions.
- **StrongSORT**: DeepSORT with modern ReID, EMA appearance, and post-processing.

### Transformer trackers

TrackFormer / MOTR / TransTrack use track queries and attention instead of explicit
assignment. They are elegant and end-to-end, but need more data and compute than this
project currently has. Read them for the query formulation; do not make them the next
sprint.

## The basketball-specific problems

1. **Camera motion.** Broadcast pans, zooms, and cuts. Without GMC, IoU association
   collapses on a pan. A cut is worse: identities should reset unless the game state
   proves continuity.
2. **Crowding.** Ten players plus referees, plus bench and crowd near the sidelines. The
   eval harness already warns that person counts include referees and bench; that is a
   role-classification problem, not a tracker bug.
3. **Similar appearance.** Two teammates in the same jersey are nearly identical to a
   color histogram. ReID must use pose/gait, jersey number, or spatial continuity.
4. **Occlusion.** Screens, drives, and rebounding create multi-second occlusions. Kalman
   alone will drift; appearance alone will swap. Use both, and let the track die rather
   than confidently attach to the wrong player.
5. **Possession changes.** A steal or defensive rebound is a legitimate identity
   discontinuity for the *ball*, not for the players.

## Metrics that separate failure modes

- **MOTA** = `1 - (FN + FP + IDSW) / GT`. One number, dominated by detection errors;
  hides association quality.
- **IDF1** = `2·IDTP / (2·IDTP + IDFP + IDFN)`. Identity-preserving recall; better for
  "did we keep the same ID".
- **HOTA** = `sqrt(DetA · AssA)` averaged over localization thresholds. Decomposes
  detection and association, which is exactly the diagnostic we need.

Report all three, per game and per footage type. A tracker that improves MOTA by
detecting more people but fragments IDs is not an improvement for coaching.

## Repo mapping

- `tracking/byte_track.py` — Kalman + two-stage association.
- `tracking/bytetrack_integration.py` — pipeline integration.
- `eval/metrics/tracking.py` — MOTA, IDF1, ID switches, IoU matching.
- `scripts/export_mot.py` — MOTChallenge rows for TrackEval.
- `eval/score.py` — track churn: unique tracks, long tracks (≥50% of clip), fragments
  (<10%). This is the right instinct; formalize it as a table.

## Failure modes

- **Counting referees as players.** Fix with role classification, not by raising the
  detection threshold.
- **Trusting a swapped ID.** A confident wrong association is worse than a dropped track.
- **Evaluating on frames.** The unit is the clip/game; see `measurement.experiment`.
- **No GMC on broadcast.** Association metrics will look fine on static clips and fail
  on the real footage.
- **ReID trained on pedestrians.** Market-1501 embeddings do not transfer to baggy
  basketball uniforms; fine-tune or use a sports ReID set.

## Open questions

1. How much of the current ID churn is caused by camera motion versus occlusion versus
   detection gaps? Instrument each separately.
2. Does jersey-color ReID help or hurt when both teams wear similar colors (the repo
   already supports `--team1-dark` for this case)?
3. What is the right `max_age` for basketball? A screen can hide a player for ~1 s
   (30 frames); a cut should reset immediately.
4. Can we use court position as a prior for association? Players move on a 94×50 ft
   plane with plausible speeds; a 40 ft jump in one frame is impossible.

## Proposed experiments

- **EX-020** Add HOTA to `eval/metrics/tracking.py` and report it beside MOTA/IDF1.
- **EX-021** Enable BoT-SORT-style GMC on a broadcast clip with pans; measure ID switches.
- **EX-022** Train/tune a basketball ReID embedding on DeepSportRadar or SportsMOT;
  compare with color-only association.
- **EX-023** Add role classification (player / referee / bench) and re-run person-count
  checks with the corrected denominator.
