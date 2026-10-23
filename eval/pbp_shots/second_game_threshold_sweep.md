# PBP Shot Detector Threshold Sweep

Source: `outputs\pbp_shots_second\results.json`

Positives: 36

Negatives: 30

FPS: 30

| MIN_RISE_PX | ±1s P/R/F1 | ±2s P/R/F1 | ±3s P/R/F1 | Positive clips with any shot | Negative clips with any shot |
|---:|---|---|---|---:|---:|
| 60 | 0.18/0.19/0.19 | 0.32/0.33/0.32 | 0.40/0.42/0.41 | 20/36 | 10/30 |
| 80 | 0.17/0.11/0.13 | 0.33/0.22/0.27 | 0.42/0.28/0.33 | 14/36 | 9/30 |
| 100 | 0.12/0.06/0.08 | 0.19/0.08/0.12 | 0.31/0.14/0.19 | 9/36 | 7/30 |
| 120 | 0.11/0.03/0.04 | 0.22/0.06/0.09 | 0.33/0.08/0.13 | 5/36 | 4/30 |

## Notes

- This replays the cached smoothed frames, not the full video pipeline.
- The PBP frame is the event clock, not the release frame.
- A lower rise threshold should recover layups/rim attempts but may add false positives; the table shows the trade-off.
- Negative counts depend on the negative set; the original 7-negative set was not fully clean (see candidate_audit.md).
