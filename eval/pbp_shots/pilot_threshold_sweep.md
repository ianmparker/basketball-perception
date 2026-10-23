# PBP Shot Detector Threshold Sweep

Source: `outputs\pbp_shots_pilot\results.json`

Positives: 29

Negatives: 17

FPS: 30

| MIN_RISE_PX | ±1s P/R/F1 | ±2s P/R/F1 | ±3s P/R/F1 | Positive clips with any shot | Negative clips with any shot |
|---:|---|---|---|---:|---:|
| 60 | 0.19/0.45/0.27 | 0.25/0.59/0.35 | 0.33/0.76/0.46 | 27/29 | 8/17 |
| 80 | 0.23/0.41/0.29 | 0.34/0.62/0.44 | 0.40/0.72/0.51 | 28/29 | 6/17 |
| 100 | 0.19/0.21/0.20 | 0.28/0.31/0.29 | 0.38/0.41/0.39 | 20/29 | 4/17 |
| 120 | 0.21/0.14/0.17 | 0.37/0.24/0.29 | 0.47/0.31/0.38 | 13/29 | 1/17 |

## Notes

- This replays the cached smoothed frames, not the full video pipeline.
- The PBP frame is the event clock, not the release frame.
- A lower rise threshold should recover layups/rim attempts but may add false positives; the table shows the trade-off.
- Negative counts depend on the negative set; the original 7-negative set was not fully clean (see candidate_audit.md).
