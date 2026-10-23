# Non-Shot Filter — Cross-Period Holdout

Train on three quarters, test on the held-out quarter. Labels are clip-level:
all candidates in a positive clip are positive; all candidates in a balanced
no-FG clip are negative. This is still one game and not a second broadcast.

Rows: 246; clips: 222; candidates: 246
Overall holdout AUC: 0.717

## Per-quarter holdout

| Held-out period | Rows | Positive | Negative | AUC | Event F1 @0.3 (±2 s) |
|---:|---:|---:|---:|---:|---:|
| Q1 | 70 | 58 | 12 | 0.672 | 0.57 |
| Q2 | 70 | 56 | 14 | 0.722 | 0.58 |
| Q3 | 59 | 54 | 5 | 0.730 | 0.46 |
| Q4 | 47 | 43 | 4 | 0.866 | 0.45 |

## Overall holdout event score

| Filter threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 92 | 53 | 36 | 0.63 | 0.72 | 0.67 |
| 0.4 | 88 | 47 | 40 | 0.65 | 0.69 | 0.67 |
| 0.5 | 81 | 38 | 53 | 0.68 | 0.60 | 0.64 |
| 0.6 | 76 | 33 | 61 | 0.70 | 0.56 | 0.62 |
| 0.7 | 66 | 28 | 75 | 0.70 | 0.47 | 0.56 |

At ±3 s, raw rise80 on the same held-out records: P 0.43 / R 0.65 / F1 0.52.
At ±2 s, raw rise80: P 0.32 / R 0.49 / F1 0.39.

Do not adopt on one game. The filter is promising only if it survives a
second broadcast and a frozen threshold.
