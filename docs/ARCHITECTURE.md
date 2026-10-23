# Architecture

The pipeline is a set of composable stages around a versioned result schema. Every
stage can be run on its own, and the expensive GPU stages can be cached.

```
video
  |
  v
[ decode + frame cap ] -> [ court keypoints ] -> homography (or manual calibration)
  |                            |
  |                            +-> court coordinates
  v
[ player detection ] -> [ multi-object tracking ] -> [ team ID ]
  |                            |                        |
  |                            +-> player tracks        +-> possession
  v
[ ball detection ] -> [ ball smoothing / interpolation ] -> [ event detectors ]
  |                                                              |
  +------------------------------+-------------------------------+
                                 v
                    [ render + tactical view + report ]
```

## Modules

| Module | Responsibility |
|---|---|
| `inference/pipeline.py` | Owns model weights, trackers, and the per-frame state machine. |
| `inference/run_video.py` | CLI entry point, presets, and output wiring. |
| `inference/schema.py` | Versioned `ClipResult` pydantic schema. |
| `inference/stubs.py` | Per-frame detection cache keyed by video + model identity. |
| `tracking/` | ByteTrack/BoT-SORT integration, ball Kalman filter, simple tracker. |
| `homography/` | Manual homography tool and the learned 18-keypoint court model. |
| `events/` | Shot, screen, pick-and-roll, and dunk detectors with gates. |
| `analytics/` | Team ID, possession, shot features, four factors, xPTS. |
| `visualization/` | Court rendering, tactical view, overlays, heatmaps. |
| `state_layer/` | Game state, lineups, possession ledger, ESPN ingestion. |
| `eval/` | Metrics, scoring harness, diagnostics, committed scorecards. |
| `scripts/` | Demo, calibration, labeling, evals, training entry points. |
| `research/` | Bibliography, knowledge notes, experiment backlog. |
| `registry/` | Provenance records for external data sources. |

## Design decisions

**One schema, many stages.** `inference/schema.py` defines the `ClipResult` that every
stage writes and every consumer reads. The demo page, evaluator, and event exporter
all depend on the schema rather than on internal stage objects.

**Two-pass ball tracking.** Pass 1 collects per-frame detections. Pass 2 smooths the
ball with hindsight (argmax-confidence picker, interpolation, render). This is slower
than streaming but produces a stable track through occlusions. `--streaming` disables
it for latency-sensitive experiments.

**Cache the expensive stages.** `inference/stubs.py` caches court keypoints, player
detections, ball detections, and pose by `(video, model, mtime)`. Iterating on the
event layer does not re-run the GPU detectors.

**Calibration has a fallback.** The learned court model works on NBA-style broadcast.
When it finds no keypoints, a fixed-camera manual homography is used. The manual tool
refuses a calibration worse than a configurable reprojection RMS, so a bad click set
does not silently ship.

**Events are gated, not free.** Shot candidates pass court-view, pose, and motion
gates. The gates are deliberately reported in the scorecards: a gate that improves
precision by removing real shots is a regression, not a win.

**Evaluation is part of the code.** Metrics, label checks, confidence intervals, and
the noise floor live in `eval/` rather than in a notebook. A change without a
scorecard is not reviewable.

## Failure handling

- Model paths are validated before inference; a missing checkpoint produces a clear
  error rather than a traceback halfway through a clip.
- `inference/device_guard.py` warns when a GPU is present but PyTorch is CPU-only.
- The demo caps frames and transcodes to H.264 so the browser can play the result.
- The page only shows fields that are reliable; invalid shot locations are dropped and
  flagged in the timeline.
