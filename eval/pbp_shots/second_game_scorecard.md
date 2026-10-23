# PBP Shot Event Evaluation

PBP event set: `401766463`.
Positives: 36 PBP field-goal attempts; negatives: 30 non-shot clips.
Clip fps: 30.0. Match tolerance is reported in seconds.

**Candidate-level timing caveat:** human labels in `candidate_audit.md` show real-shot
candidates a median 3.2 s before the PBP event. This symmetric event scorecard is a
misaligned view, not the final release metric.

## Event-level scorecard

| Tolerance | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median | Offset mean |
|---|---:|---:|---:|---|---|---:|---:|---:|
| ±1s (30f) | 1 | 8 | 35 | 0.11 (0.02-0.43) | 0.03 (0.01-0.14) | 0.04 | 24 | 24.0 |
| ±2s (60f) | 2 | 7 | 34 | 0.22 (0.06-0.55) | 0.06 (0.01-0.18) | 0.09 | 40 | 32.0 |
| ±3s (90f) | 3 | 6 | 33 | 0.33 (0.12-0.65) | 0.08 (0.03-0.22) | 0.13 | 40 | 49.0 |

## Clip-level check

- Positive clips with any shot event: 5/36
- Negative clips with any shot event: 4/30

## Per-event

| Event | Kind | Truth | GT frame | Pred frames | Error |
|---|---|---|---:|---|---|
| shot_0017 | positive | MADE | 210 | [] |  |
| shot_0018 | positive | MISS | 210 | [] |  |
| shot_0019 | positive | MISS | 210 | [] |  |
| shot_0020 | positive | MISS | 210 | [] |  |
| shot_0021 | positive | MISS | 210 | [] |  |
| shot_0022 | positive | MISS | 210 | [] |  |
| shot_0023 | positive | MADE | 210 | [] |  |
| shot_0024 | positive | MADE | 210 | [] |  |
| shot_0025 | positive | MADE | 210 | [] |  |
| shot_0026 | positive | MADE | 210 | [293] |  |
| shot_0057 | positive | MISS | 210 | [] |  |
| shot_0058 | positive | MISS | 210 | [] |  |
| shot_0063 | positive | MADE | 210 | [] |  |
| shot_0064 | positive | MISS | 210 | [] |  |
| shot_0065 | positive | MADE | 210 | [] |  |
| shot_0066 | positive | MADE | 210 | [] |  |
| shot_0067 | positive | MISS | 210 | [234] |  |
| shot_0069 | positive | MADE | 210 | [] |  |
| shot_0106 | positive | MISS | 210 | [] |  |
| shot_0107 | positive | MADE | 210 | [] |  |
| shot_0108 | positive | MADE | 210 | [] |  |
| shot_0109 | positive | MISS | 210 | [] |  |
| shot_0110 | positive | MISS | 210 | [] |  |
| shot_0111 | positive | MADE | 210 | [] |  |
| shot_0112 | positive | MISS | 210 | [250] |  |
| shot_0113 | positive | MADE | 210 | [] |  |
| shot_0114 | positive | MADE | 210 | [] |  |
| shot_0115 | positive | MISS | 210 | [] |  |
| shot_0148 | positive | MADE | 210 | [] |  |
| shot_0149 | positive | MADE | 210 | [] |  |
| shot_0151 | positive | MISS | 210 | [95] |  |
| shot_0152 | positive | MISS | 210 | [] |  |
| shot_0154 | positive | MADE | 210 | [] |  |
| shot_0155 | positive | MADE | 210 | [] |  |
| shot_0158 | positive | MISS | 210 | [] |  |
| shot_0159 | positive | MISS | 210 | [29] |  |
| bal_neg_001 | negative | NO SHOT | — | [] |  |
| bal_neg_002 | negative | NO SHOT | — | [] |  |
| bal_neg_003 | negative | NO SHOT | — | [] |  |
| bal_neg_004 | negative | NO SHOT | — | [128] |  |
| bal_neg_005 | negative | NO SHOT | — | [] |  |
| bal_neg_006 | negative | NO SHOT | — | [] |  |
| bal_neg_007 | negative | NO SHOT | — | [] |  |
| bal_neg_008 | negative | NO SHOT | — | [] |  |
| bal_neg_009 | negative | NO SHOT | — | [23] |  |
| bal_neg_010 | negative | NO SHOT | — | [] |  |
| bal_neg_011 | negative | NO SHOT | — | [] |  |
| bal_neg_012 | negative | NO SHOT | — | [] |  |
| bal_neg_013 | negative | NO SHOT | — | [] |  |
| bal_neg_014 | negative | NO SHOT | — | [] |  |
| bal_neg_015 | negative | NO SHOT | — | [] |  |
| bal_neg_016 | negative | NO SHOT | — | [] |  |
| bal_neg_017 | negative | NO SHOT | — | [] |  |
| bal_neg_018 | negative | NO SHOT | — | [] |  |
| bal_neg_019 | negative | NO SHOT | — | [63] |  |
| bal_neg_020 | negative | NO SHOT | — | [] |  |
| bal_neg_021 | negative | NO SHOT | — | [] |  |
| bal_neg_022 | negative | NO SHOT | — | [] |  |
| bal_neg_023 | negative | NO SHOT | — | [194] |  |
| bal_neg_024 | negative | NO SHOT | — | [] |  |
| bal_neg_025 | negative | NO SHOT | — | [] |  |
| bal_neg_026 | negative | NO SHOT | — | [] |  |
| bal_neg_027 | negative | NO SHOT | — | [] |  |
| bal_neg_028 | negative | NO SHOT | — | [] |  |
| bal_neg_029 | negative | NO SHOT | — | [] |  |
| bal_neg_030 | negative | NO SHOT | — | [] |  |

## Notes

- The PBP frame is the event clock, not the release frame; scorebug alignment corrects the video offset.
- Event-level metrics count extra predicted shots as false positives.
- The 30 negatives are non-shot clips with no PBP field goal in the window.
- This is one full game; the next step is to expand to all 186 PBP events and a second game.
