# Broadcast Event Scorecard

**Labels:** `labeled`<br>
**Runs:** 3<br>
**Judged candidates:** 7<br>
**Gather only:** 0<br>
**Unlabeled:** 0

### Tolerance ±5 frames

| Run | Regime | Judged | Gather only | GT | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| kyrie_current | NBA broadcast | 2 | 0 | 2 | 2 | 0 | 0 | 1.00 (0.34-1.00) | 1.00 (0.34-1.00) | 1.00 | — |
| kyrie_legacy | NBA broadcast | 3 | 0 | 2 | 2 | 1 | 0 | 0.67 (0.21-0.94) | 1.00 (0.34-1.00) | 0.80 | — |
| lima_q2b | HS top-down | 2 | 0 | 0 | 0 | 2 | 0 | 0.00 (0.00-0.66) | 0.00 (0.00-0.00) | 0.00 | — |
| **HS top-down** | — | 2 | 0 | 0 | 0 | 2 | 0 | 0.00 (0.00-0.66) | 0.00 (0.00-0.00) | 0.00 | — |
| **NBA broadcast** | — | 5 | 0 | 4 | 4 | 1 | 0 | 0.80 (0.38-0.96) | 1.00 (0.51-1.00) | 0.89 | — |
| **All runs (mixed regimes)** | — | 7 | 0 | 4 | 4 | 3 | 0 | 0.57 (0.25-0.84) | 1.00 (0.51-1.00) | 0.73 | — |

### Tolerance ±15 frames

| Run | Regime | Judged | Gather only | GT | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| kyrie_current | NBA broadcast | 2 | 0 | 2 | 2 | 0 | 0 | 1.00 (0.34-1.00) | 1.00 (0.34-1.00) | 1.00 | — |
| kyrie_legacy | NBA broadcast | 3 | 0 | 2 | 2 | 1 | 0 | 0.67 (0.21-0.94) | 1.00 (0.34-1.00) | 0.80 | — |
| lima_q2b | HS top-down | 2 | 0 | 0 | 0 | 2 | 0 | 0.00 (0.00-0.66) | 0.00 (0.00-0.00) | 0.00 | — |
| **HS top-down** | — | 2 | 0 | 0 | 0 | 2 | 0 | 0.00 (0.00-0.66) | 0.00 (0.00-0.00) | 0.00 | — |
| **NBA broadcast** | — | 5 | 0 | 4 | 4 | 1 | 0 | 0.80 (0.38-0.96) | 1.00 (0.51-1.00) | 0.89 | — |
| **All runs (mixed regimes)** | — | 7 | 0 | 4 | 4 | 3 | 0 | 0.57 (0.25-0.84) | 1.00 (0.51-1.00) | 0.73 | — |

### Tolerance ±30 frames

| Run | Regime | Judged | Gather only | GT | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| kyrie_current | NBA broadcast | 2 | 0 | 2 | 2 | 0 | 0 | 1.00 (0.34-1.00) | 1.00 (0.34-1.00) | 1.00 | — |
| kyrie_legacy | NBA broadcast | 3 | 0 | 2 | 2 | 1 | 0 | 0.67 (0.21-0.94) | 1.00 (0.34-1.00) | 0.80 | — |
| lima_q2b | HS top-down | 2 | 0 | 0 | 0 | 2 | 0 | 0.00 (0.00-0.66) | 0.00 (0.00-0.00) | 0.00 | — |
| **HS top-down** | — | 2 | 0 | 0 | 0 | 2 | 0 | 0.00 (0.00-0.66) | 0.00 (0.00-0.00) | 0.00 | — |
| **NBA broadcast** | — | 5 | 0 | 4 | 4 | 1 | 0 | 0.80 (0.38-0.96) | 1.00 (0.51-1.00) | 0.89 | — |
| **All runs (mixed regimes)** | — | 7 | 0 | 4 | 4 | 3 | 0 | 0.57 (0.25-0.84) | 1.00 (0.51-1.00) | 0.73 | — |

## Notes

- A true positive matches a labeled real shot to a prediction within the tolerance.
- A false positive is a labeled false-positive candidate that still has a prediction.
- A false negative is a labeled real shot with no prediction in tolerance.
- Gather-only candidates are excluded from precision/recall: the clip shows the
  start of a shot but not the release.
- The label set currently contains only predicted candidates. A real shot the
  detector missed is not in `labels.json`, so **recall is optimistic**. Add missed
  shots with `predicted_frame: null` to measure it.
- Do **not** compare the NBA broadcast aggregate with the HS top-down probe.
  They are different camera regimes; the top-down result is a transfer check.
- Offsets use `release_frame` only; candidates labeled without a release frame count
  as matches but do not contribute to the offset distribution.
- With this few events, the confidence intervals are wide; report the counts, not
  just F1.
