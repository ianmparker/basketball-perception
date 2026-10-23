# PBP Release-Window Sensitivity

The candidate audit showed that the PBP event is not the release. This table
re-scores the **same cached predictions** with shifted match windows. It is a
sensitivity analysis, not a new detector run.

Candidate offsets for **real shots**: n=20, median -3.21 s, mean -3.00 s, range -5.53 to -0.37 s.
Candidate offsets for **not-shots**: n=4, median +1.15 s, range -1.30 to +1.67 s.

| Match window | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median (frames) |
|---|---:|---:|---:|---|---|---:|---:|
| symmetric ±1 s | 3 | 11 | 16 | 0.21 (0.08-0.48) | 0.16 (0.06-0.38) | 0.18 | -12 |
| symmetric ±2 s | 5 | 9 | 14 | 0.36 (0.16-0.61) | 0.26 (0.12-0.49) | 0.30 | -12 |
| symmetric ±3 s | 7 | 7 | 12 | 0.50 (0.27-0.73) | 0.37 (0.19-0.59) | 0.42 | -27 |
| release -5.5/+1 s | 10 | 4 | 9 | 0.71 (0.45-0.88) | 0.53 (0.32-0.73) | 0.61 | -66 |
| release -5.0/+1 s | 10 | 4 | 9 | 0.71 (0.45-0.88) | 0.53 (0.32-0.73) | 0.61 | -66 |
| release -4.5/+1 s | 10 | 4 | 9 | 0.71 (0.45-0.88) | 0.53 (0.32-0.73) | 0.61 | -66 |
| release -5.5/0 s | 9 | 5 | 10 | 0.64 (0.39-0.84) | 0.47 (0.27-0.68) | 0.55 | -85 |

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
