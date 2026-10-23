# PBP Shot Detector Threshold Sweep

Source: `outputs\pbp_shots_full\results.json`

Positives: 162

Negatives: 28

FPS: 30

| MIN_RISE_PX | ±1s P/R/F1 | ±2s P/R/F1 | ±3s P/R/F1 | Positive clips with any shot | Negative clips with any shot |
|---:|---|---|---|---:|---:|
| 60 | 0.20/0.38/0.26 | 0.32/0.60/0.42 | 0.40/0.75/0.53 | 144/162 | 15/28 |
| 80 | 0.23/0.33/0.27 | 0.35/0.49/0.41 | 0.47/0.65/0.54 | 135/162 | 12/28 |
| 100 | 0.22/0.22/0.22 | 0.36/0.36/0.36 | 0.50/0.51/0.51 | 109/162 | 9/28 |
| 120 | 0.25/0.17/0.20 | 0.39/0.27/0.32 | 0.55/0.38/0.45 | 81/162 | 4/28 |

## Notes

- This replays the cached smoothed frames, not the full video pipeline.
- The PBP frame is the event clock, not the release frame.
- A lower rise threshold should recover layups/rim attempts but may add false positives; the table shows the trade-off.
- Negative counts depend on the negative set; the original 7-negative set was not fully clean (see candidate_audit.md).
