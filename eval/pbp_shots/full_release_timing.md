# PBP Release-Window Sensitivity

The candidate audit showed that the PBP event is not the release. This table
re-scores the **same cached predictions** with shifted match windows. It is a
sensitivity analysis, not a new detector run.

Candidate offsets for **real shots**: n=20, median -3.21 s, mean -3.00 s, range -5.53 to -0.37 s.
Candidate offsets for **not-shots**: n=4, median +1.15 s, range -1.30 to +1.67 s.

| Match window | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median (frames) |
|---|---:|---:|---:|---|---|---:|---:|
| symmetric ±1 s | 28 | 84 | 134 | 0.25 (0.18-0.34) | 0.17 (0.12-0.24) | 0.20 | 6 |
| symmetric ±2 s | 44 | 68 | 118 | 0.39 (0.31-0.49) | 0.27 (0.21-0.34) | 0.32 | 3 |
| symmetric ±3 s | 62 | 50 | 100 | 0.55 (0.46-0.64) | 0.38 (0.31-0.46) | 0.45 | -12 |
| release -5.5/+1 s | 69 | 43 | 93 | 0.62 (0.52-0.70) | 0.43 (0.35-0.50) | 0.50 | -50 |
| release -5.0/+1 s | 68 | 44 | 94 | 0.61 (0.51-0.69) | 0.42 (0.35-0.50) | 0.50 | -49 |
| release -4.5/+1 s | 67 | 45 | 95 | 0.60 (0.51-0.68) | 0.41 (0.34-0.49) | 0.49 | -49 |
| release -5.5/0 s | 57 | 55 | 105 | 0.51 (0.42-0.60) | 0.35 (0.28-0.43) | 0.42 | -66 |

## Read this carefully

- The shifted window improves the event score because the detector fires before
  the PBP event. It is not evidence that the detector got better.
- Two clocks: the orange candidate is the release/attempt anchor; the blue PBP
  event is the made/missed result. The offset is release-to-result, not detector
  error. `ShotEvent.frame_start` is a rise trigger that can be slightly before
  release; release-frame labels refine that sub-second term.
- The window was chosen from the same 20 shots that produced the audit, so it is
  an in-sample sensitivity result. Validate it on a second game before claiming a
  stable release-time correction.
- Some predictions within the window may be duplicate phases of the same shot or
  adjacent real shots. Candidate dedupe is still required.
