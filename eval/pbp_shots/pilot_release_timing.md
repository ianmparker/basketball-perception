# PBP Release-Window Sensitivity

The candidate audit showed that the PBP event is not the release. This table
re-scores the **same cached predictions** with shifted match windows. It is a
sensitivity analysis, not a new detector run.

Candidate offsets for **real shots**: n=20, median -3.21 s, mean -3.00 s, range -5.53 to -0.37 s.
Candidate offsets for **not-shots**: n=4, median +1.15 s, range -1.30 to +1.67 s.

| Match window | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median (frames) |
|---|---:|---:|---:|---|---|---:|---:|
| symmetric ±1 s | 4 | 15 | 25 | 0.21 (0.09-0.43) | 0.14 (0.05-0.31) | 0.17 | 20 |
| symmetric ±2 s | 7 | 12 | 22 | 0.37 (0.19-0.59) | 0.24 (0.12-0.42) | 0.29 | 20 |
| symmetric ±3 s | 9 | 10 | 20 | 0.47 (0.27-0.68) | 0.31 (0.17-0.49) | 0.38 | -12 |
| release -5.5/+1 s | 12 | 7 | 17 | 0.63 (0.41-0.81) | 0.41 (0.26-0.59) | 0.50 | -66 |
| release -5.0/+1 s | 12 | 7 | 17 | 0.63 (0.41-0.81) | 0.41 (0.26-0.59) | 0.50 | -66 |
| release -4.5/+1 s | 12 | 7 | 17 | 0.63 (0.41-0.81) | 0.41 (0.26-0.59) | 0.50 | -66 |
| release -5.5/0 s | 11 | 8 | 18 | 0.58 (0.36-0.77) | 0.38 (0.23-0.56) | 0.46 | -85 |

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
