---
id: perception.detection.objects
title: Object detection for players, ball, rim, and referees
layer: perception
track: perception.detection
status: active
confidence: high
updated: 2026-10-21
sources: [arxiv-1506.02640, arxiv-1612.03144, arxiv-1708.02002, arxiv-2005.12872, arxiv-2304.08069]
code: [models/, src/, data/]
prereqs: []
tags: [detection, YOLO, DETR, small objects, mAP]
---

# Object detection for players, ball, rim, and referees

**Claim.** Detection is a **class-imbalance and scale** problem before it is an
architecture problem. The ball is ~16 px, the players are ~100–300 px, and the referee is
rare. A single global confidence threshold over all classes is therefore a design error.

## What the repo runs

- YOLOv8n/m family, 17-class unified dataset (7,860 images), ball/player/ref/rim/court
  keypoints.
- YOLOv8x-pose for COCO-17 keypoints.
- A learned 18-keypoint court model for homography.

The detector is fine. The gaps are in **class balance, evaluation, and operating point**:

- The ball has 5,188 annotations vs 12,169 players; the ball is also 10–20× smaller in
  area. Focal loss exists for this.
- `mAP@50` averages over classes and IoU thresholds; it does not tell you the ball recall
  at the pipeline's threshold.
- Referee and bench detections are counted as people in tracking checks. That is a class
  problem, not a tracker problem.

## The architecture families

| Family | Idea | Trade-off |
|---|---|---|
| YOLO (one-stage) | dense grid, direct box + class prediction | fast, strong baseline; small objects need multi-scale |
| Faster R-CNN (two-stage) | region proposals then classify | accurate, slower; useful accuracy ceiling |
| FPN | multi-scale feature fusion | the standard fix for small objects |
| Focal loss | down-weight easy background | essential for extreme foreground/background imbalance |
| DETR | set prediction with queries, no NMS | elegant, slower to converge; needs more data |
| Deformable DETR | sparse attention at reference points | fixes DETR convergence; good for small objects |
| RT-DETR | real-time DETR | an alternative to YOLO when latency allows |

For this project, the practical sequence is: YOLO + FPN + focal loss + per-class
thresholds. DETR is a research direction, not a sprint.

## Evaluation that matches the pipeline

Report, per class:

- **AP@50** and **AP@50:95** as detector comparisons;
- **precision/recall at the operating threshold** actually used;
- **recall by object size** (small/medium/large) — this is where the ball lives;
- **count error** per frame (players detected vs expected);
- **false-positive examples** (crowd, scoreboard, shoes).

The eval harness already warns that person counts include referees and bench. Add a
`role` head or a post-hoc classifier before treating a count as a player count.

## Per-class thresholds are not a hack

A ball at `conf=0.25` and a player at `conf=0.25` are different operating points. The ball
needs a lower threshold for recall; the player can afford a higher one for precision.
Expose thresholds per class in the pipeline config and tune them on the grouped
benchmark, not on one clip.

## Repo mapping

- `models/` — training scripts and runs.
- `src/` — pipeline source modules.
- `data/raw/` — the merged 17-class dataset; `data/ingestion/merge_datasets.py` maps
  classes.
- `inference/pipeline.py` — detection thresholds and class handling.
- `eval/metrics/detection.py` — event + team assignment; add per-class AP.

## Failure modes

- **One threshold for all classes.** The ball disappears while the players are fine.
- **mAP as the only number.** It hides the operating-point behavior the pipeline uses.
- **Arena memorization.** The detector learns the court paint and scorebug; test on a
  held-out arena.
- **Label noise in the merged set.** Three datasets with different class definitions were
  merged into 17 classes; audit the mapping before trusting per-class AP.
- **Ignoring referees.** They are people; the tracker will track them. Classify the role.

## Open questions

1. What is the ball AP@50:95 vs players, and how much does focal loss + FPN close the gap?
2. Does a per-class threshold recover ball recall without a false-positive explosion?
3. Can a lightweight role classifier (player/ref/bench) be trained from the existing
   boxes, or do we need new labels?
4. Is the merged 17-class taxonomy actually consistent, or are we training on noise?

## Proposed experiments

- **EX-130** Report per-class AP@50 and AP@50:95 on the current validation split.
- **EX-131** Sweep per-class ball thresholds and plot precision/recall on the grouped
  benchmark.
- **EX-132** Train a role classifier on player/ref crops and re-run the person-count
  checks.
