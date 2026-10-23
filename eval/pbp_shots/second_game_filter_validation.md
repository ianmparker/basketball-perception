# Second-Game Non-Shot Filter Validation

**Train:** game 1, 162 positives + 60 balanced negatives.  
**Test:** game 2, second-game validation subset.  
**Threshold:** frozen at 0.3; no tuning on game 2.

Test candidates: 24 (15 positive, 9 negative)  
Second-game AUC: 0.659

## Event-level, game 2

| Window | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| filtered ±3 s @ 0.3 | 9 | 11 | 23 | 0.45 | 0.28 | 0.35 |

Raw rise80 comparison on the same game-2 records:
- ±3 s: P 0.42 / R 0.28 / F1 0.33
- ±2 s: P 0.33 / R 0.22 / F1 0.27

## Verdict

**Fail / weak transfer.** Do not adopt the filter. Keep rise80 as the candidate generator and investigate second-broadcast domain shift.

The filter is only useful if it improves precision/F1 without collapsing positive
recall on the second broadcast. It did not meet that bar here.
