# PBP Shot Detector Threshold Sweep

Source: generated from the cached PBP benchmark run (raw `results.json` not included)

Positives: 19

Negatives: 7

FPS: 30

| MIN_RISE_PX | ±1s P/R/F1 | ±2s P/R/F1 | ±3s P/R/F1 | Positive clips with any shot | Negative clips with any shot |
|---:|---|---|---|---:|---:|
| 60 | 0.19/0.42/0.26 | 0.26/0.58/0.36 | 0.31/0.68/0.43 | 18/19 | 5/7 |
| 80 | 0.22/0.42/0.29 | 0.36/0.68/0.47 | 0.36/0.68/0.47 | 19/19 | 4/7 |
| 100 | 0.16/0.16/0.16 | 0.26/0.26/0.26 | 0.37/0.37/0.37 | 14/19 | 0/7 |
| 120 | 0.21/0.16/0.18 | 0.36/0.26/0.30 | 0.50/0.37/0.42 | 10/19 | 0/7 |

## Notes

- This replays the cached smoothed frames, not the full video pipeline.
- The PBP frame is the event clock, not the release frame.
- A lower rise threshold should recover layups/rim attempts but may add false positives; the table shows the trade-off.
- Negative counts depend on the negative set; the original 7-negative set was not fully clean (see candidate_audit.md).
