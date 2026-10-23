# Evaluation

The project treats evaluation as a first-class artifact. Every claim in the README is
backed by a committed scorecard in `eval/`, and the scorecards include the runs that
failed.

## What is measured

| Layer | Metric | Where |
|---|---|---|
| Ball detection | mAP@50, precision, recall | `docs/MODEL_EVALUATION_YOLOV8N_5EPOCH.md`, training runs |
| Court calibration | Median/mean reprojection error (ft) | `homography/learned/v18_reproj_eval.json` |
| Tracking | Players per frame, MOTA/IDF1/ID switches | `eval/metrics/tracking.py` |
| Events | Precision/recall/F1 at frame and second tolerances | `eval/metrics/events.py` |
| Possession | Team/jersey possession accuracy | `eval/metrics/possession.py` |
| Team ID | Assignment counts, balance, unassigned frames | `eval/metrics/detection.py` |

## Event scorecards

Shot events are scored twice: at frame tolerances (±5/15/30 frames) and at second
tolerances (±1/2/3 s). The two views answer different questions:

- Frame tolerance: did the detector find the event near the annotated frame?
- Second tolerance: does the event survive the annotation and clip-alignment noise?

Every scorecard reports precision and recall with **Wilson 95% confidence intervals**,
plus the TP/FP/FN counts. With small clip sets, the interval matters more than the
point estimate.

### Two clocks

A shot has at least two relevant times: the release/attempt and the play-by-play
result. Candidate labels can sit several seconds before the PBP event, so a symmetric
match window is a misaligned view. `eval/pbp_shots/two_clock.md` records the
interpretation; release-frame calibration remains open.

### Noise floor and audit

- A shuffled-label check establishes what the metric looks like without signal.
- Label audits remove misaligned clips before scoring; the removals are recorded in
  the scorecard.
- Grouped scorecards keep clips from the same game or footage regime together, so a
  single game cannot dominate the result.

## Current results

Full-game PBP benchmark (162 positive clips, 28 negative clips):

| Tolerance | Precision (95% CI) | Recall (95% CI) | F1 |
|---|---|---|---|
| ±1 s | 0.25 (0.18–0.34) | 0.17 (0.12–0.24) | 0.20 |
| ±2 s | 0.39 (0.31–0.48) | 0.27 (0.21–0.34) | 0.32 |
| ±3 s | 0.55 (0.46–0.64) | 0.38 (0.31–0.46) | 0.45 |

These numbers are the reason the event layer is described as a work in progress. The
detection, tracking, and calibration layers are materially stronger.

## Running the harness

```bash
python eval/score.py --help
python scripts/score_broadcast_events.py
```

To add a benchmark:

1. Put labels in `eval/labels/` using `eval/schema.md`.
2. Add the clip to the manifest with source, license, fps, and length.
3. Run the scorer and commit the scorecard, including the known failures.
