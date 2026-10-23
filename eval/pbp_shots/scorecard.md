# PBP Shot Event Evaluation

Full-game PBP source: `401810030` (Knicks 137 - Timberwolves 114).
Positives: 19 PBP field-goal attempts; negatives: 7 non-shot clips.
Clip fps: 30.0. Match tolerance is reported in seconds.
Audit: 5 clips reviewed; excluded positives: shot_0041; excluded negatives: neg_003, neg_006, neg_014.

**Candidate-level timing caveat:** human labels in `candidate_audit.md` show real-shot
candidates a median 3.2 s before the PBP event. This symmetric event scorecard is a
misaligned view, not the final release metric.

## Event-level scorecard

| Tolerance | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median | Offset mean |
|---|---:|---:|---:|---|---|---:|---:|---:|
| ±1s (30f) | 3 | 11 | 16 | 0.21 (0.08-0.48) | 0.16 (0.06-0.38) | 0.18 | -12 | -6.33 |
| ±2s (60f) | 5 | 9 | 14 | 0.36 (0.16-0.61) | 0.26 (0.12-0.49) | 0.30 | -12 | -1.4 |
| ±3s (90f) | 7 | 7 | 12 | 0.50 (0.27-0.73) | 0.37 (0.19-0.59) | 0.42 | -27 | -22.57 |

## Clip-level check

- Positive clips with any shot event: 10/19
- Negative clips with any shot event: 0/7

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
| neg_001 | negative | NO SHOT | — | [] |  |
| neg_002 | negative | NO SHOT | — | [] |  |
| neg_007 | negative | NO SHOT | — | [] |  |
| neg_010 | negative | NO SHOT | — | [] |  |
| neg_011 | negative | NO SHOT | — | [] |  |
| neg_012 | negative | NO SHOT | — | [] |  |
| neg_013 | negative | NO SHOT | — | [] |  |

## Notes

- The PBP frame is the event clock, not the release frame; scorebug alignment corrects the video offset.
- Event-level metrics count extra predicted shots as false positives.
- The 7 negatives are non-shot clips with no PBP field goal in the window.
- This is one full game; the next step is to expand to all 186 PBP events and a second game.
