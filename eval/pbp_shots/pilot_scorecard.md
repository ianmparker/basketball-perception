# PBP Shot Event Evaluation

Full-game PBP source: `401810030` (Knicks 137 - Timberwolves 114).
Positives: 29 PBP field-goal attempts; negatives: 17 non-shot clips.
Clip fps: 30.0. Match tolerance is reported in seconds.

**Candidate-level timing caveat:** human labels in `candidate_audit.md` show real-shot
candidates a median 3.2 s before the PBP event. This symmetric event scorecard is a
misaligned view, not the final release metric.

## Event-level scorecard

| Tolerance | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median | Offset mean |
|---|---:|---:|---:|---|---|---:|---:|---:|
| ±1s (30f) | 4 | 15 | 25 | 0.21 (0.09-0.43) | 0.14 (0.06-0.31) | 0.17 | 20 | 1.25 |
| ±2s (60f) | 7 | 12 | 22 | 0.37 (0.19-0.59) | 0.24 (0.12-0.42) | 0.29 | 20 | 7.86 |
| ±3s (90f) | 9 | 10 | 20 | 0.47 (0.27-0.68) | 0.31 (0.17-0.49) | 0.38 | -12 | -10.67 |

## Clip-level check

- Positive clips with any shot event: 13/29
- Negative clips with any shot event: 1/17

## Per-event

| Event | Kind | Truth | GT frame | Pred frames | Error |
|---|---|---|---:|---|---|
| shot_0008 | positive | MISS | 180 | [] |  |
| shot_0011 | positive | MISS | 180 | [200, 252] |  |
| shot_0013 | positive | MISS | 180 | [114] |  |
| shot_0018 | positive | MADE | 180 | [] |  |
| shot_0020 | positive | MISS | 180 | [95] |  |
| shot_0022 | positive | MADE | 180 | [106, 168] |  |
| shot_0026 | positive | MADE | 180 | [30, 83] |  |
| shot_0057 | positive | MISS | 180 | [] |  |
| shot_0061 | positive | MISS | 180 | [83, 231] |  |
| shot_0089 | positive | MADE | 180 | [78] |  |
| shot_0100 | positive | MADE | 180 | [] |  |
| shot_0101 | positive | MISS | 180 | [] |  |
| shot_0106 | positive | MADE | 180 | [] |  |
| shot_0107 | positive | MISS | 180 | [] |  |
| shot_0124 | positive | MISS | 180 | [] |  |
| shot_0139 | positive | MADE | 180 | [78] |  |
| shot_0141 | positive | MISS | 180 | [] |  |
| shot_0149 | positive | MADE | 180 | [153] |  |
| shot_0164 | positive | MADE | 180 | [141] |  |
| shot_0021 | positive | MADE | 180 | [50] |  |
| shot_0024 | positive | MISS | 180 | [] |  |
| shot_0023 | positive | MADE | 180 | [] |  |
| shot_0063 | positive | MADE | 180 | [124, 204] |  |
| shot_0064 | positive | MISS | 180 | [] |  |
| shot_0116 | positive | MADE | 180 | [] |  |
| shot_0109 | positive | MISS | 180 | [218] |  |
| shot_0114 | positive | MADE | 180 | [] |  |
| shot_0162 | positive | MADE | 180 | [] |  |
| shot_0165 | positive | MISS | 180 | [] |  |
| neg_007 | negative | NO SHOT | — | [] |  |
| neg_010 | negative | NO SHOT | — | [] |  |
| neg_011 | negative | NO SHOT | — | [] |  |
| neg_012 | negative | NO SHOT | — | [] |  |
| pilot_neg_001 | negative | NO SHOT | — | [] |  |
| pilot_neg_002 | negative | NO SHOT | — | [21] |  |
| pilot_neg_003 | negative | NO SHOT | — | [] |  |
| pilot_neg_004 | negative | NO SHOT | — | [] |  |
| pilot_neg_005 | negative | NO SHOT | — | [] |  |
| pilot_neg_007 | negative | NO SHOT | — | [] |  |
| pilot_neg_010 | negative | NO SHOT | — | [] |  |
| pilot_neg_011 | negative | NO SHOT | — | [] |  |
| pilot_neg_012 | negative | NO SHOT | — | [] |  |
| pilot_neg_013 | negative | NO SHOT | — | [] |  |
| pilot_neg_014 | negative | NO SHOT | — | [] |  |
| pilot_neg_015 | negative | NO SHOT | — | [] |  |
| pilot_neg_016 | negative | NO SHOT | — | [] |  |

## Notes

- The PBP frame is the event clock, not the release frame; scorebug alignment corrects the video offset.
- Event-level metrics count extra predicted shots as false positives.
- The 17 negatives are non-shot clips with no PBP field goal in the window.
- This is one full game; the next step is to expand to all 186 PBP events and a second game.
