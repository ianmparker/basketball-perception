# PBP Shot Event Evaluation

Full-game PBP source: `401810030` (Knicks 137 - Timberwolves 114).
Positives: 162 PBP field-goal attempts; negatives: 28 non-shot clips.
Clip fps: 30.0. Match tolerance is reported in seconds.

**Candidate-level timing caveat:** human labels in `candidate_audit.md` show real-shot
candidates a median 3.2 s before the PBP event. This symmetric event scorecard is a
misaligned view, not the final release metric.

## Event-level scorecard

| Tolerance | TP | FP | FN | Precision (95% CI) | Recall (95% CI) | F1 | Offset median | Offset mean |
|---|---:|---:|---:|---|---|---:|---:|---:|
| ±1s (30f) | 28 | 84 | 134 | 0.25 (0.18-0.34) | 0.17 (0.12-0.24) | 0.20 | 6 | 2.39 |
| ±2s (60f) | 44 | 68 | 118 | 0.39 (0.31-0.48) | 0.27 (0.21-0.34) | 0.32 | 3 | -3.36 |
| ±3s (90f) | 62 | 50 | 100 | 0.55 (0.46-0.64) | 0.38 (0.31-0.46) | 0.45 | -12 | -19.16 |

## Clip-level check

- Positive clips with any shot event: 81/162
- Negative clips with any shot event: 4/28

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
| shot_0001 | positive | MADE | 180 | [119] |  |
| shot_0002 | positive | MISS | 180 | [] |  |
| shot_0003 | positive | MISS | 180 | [86, 174, 240] |  |
| shot_0004 | positive | MISS | 180 | [] |  |
| shot_0005 | positive | MADE | 180 | [135, 296] |  |
| shot_0006 | positive | MADE | 180 | [12, 131] |  |
| shot_0007 | positive | MISS | 180 | [85, 264] |  |
| shot_0009 | positive | MISS | 180 | [68] |  |
| shot_0010 | positive | MISS | 180 | [] |  |
| shot_0012 | positive | MADE | 180 | [97, 293] |  |
| shot_0014 | positive | MADE | 180 | [] |  |
| shot_0015 | positive | MISS | 180 | [] |  |
| shot_0016 | positive | MISS | 180 | [] |  |
| shot_0017 | positive | MADE | 180 | [] |  |
| shot_0019 | positive | MADE | 180 | [] |  |
| shot_0025 | positive | MADE | 180 | [191] |  |
| shot_0027 | positive | MADE | 180 | [128] |  |
| shot_0028 | positive | MADE | 180 | [111, 190] |  |
| shot_0029 | positive | MADE | 180 | [] |  |
| shot_0030 | positive | MISS | 180 | [127, 241] |  |
| shot_0031 | positive | MADE | 180 | [165, 213] |  |
| shot_0032 | positive | MISS | 180 | [140] |  |
| shot_0033 | positive | MISS | 180 | [108] |  |
| shot_0034 | positive | MADE | 180 | [94] |  |
| shot_0035 | positive | MISS | 180 | [] |  |
| shot_0036 | positive | MADE | 180 | [] |  |
| shot_0044 | positive | MADE | 180 | [147] |  |
| shot_0045 | positive | MADE | 180 | [] |  |
| shot_0046 | positive | MADE | 180 | [] |  |
| shot_0047 | positive | MISS | 180 | [11, 247] |  |
| shot_0048 | positive | MADE | 180 | [194] |  |
| shot_0049 | positive | MADE | 180 | [168] |  |
| shot_0050 | positive | MISS | 180 | [] |  |
| shot_0051 | positive | MADE | 180 | [] |  |
| shot_0052 | positive | MADE | 180 | [] |  |
| shot_0053 | positive | MADE | 180 | [195] |  |
| shot_0055 | positive | MISS | 180 | [124] |  |
| shot_0056 | positive | MISS | 180 | [106] |  |
| shot_0058 | positive | MISS | 180 | [190] |  |
| shot_0059 | positive | MISS | 180 | [130] |  |
| shot_0060 | positive | MADE | 180 | [12] |  |
| shot_0062 | positive | MISS | 180 | [186] |  |
| shot_0065 | positive | MADE | 180 | [194] |  |
| shot_0066 | positive | MISS | 180 | [61, 285] |  |
| shot_0067 | positive | MADE | 180 | [] |  |
| shot_0068 | positive | MISS | 180 | [204] |  |
| shot_0069 | positive | MADE | 180 | [11] |  |
| shot_0070 | positive | MISS | 180 | [] |  |
| shot_0071 | positive | MADE | 180 | [176] |  |
| shot_0072 | positive | MISS | 180 | [] |  |
| shot_0073 | positive | MADE | 180 | [114] |  |
| shot_0074 | positive | MADE | 180 | [187, 280] |  |
| shot_0075 | positive | MISS | 180 | [103] |  |
| shot_0076 | positive | MISS | 180 | [103, 155, 277] |  |
| shot_0077 | positive | MADE | 180 | [111] |  |
| shot_0078 | positive | MISS | 180 | [36] |  |
| shot_0079 | positive | MADE | 180 | [] |  |
| shot_0080 | positive | MISS | 180 | [218] |  |
| shot_0085 | positive | MISS | 180 | [] |  |
| shot_0086 | positive | MISS | 180 | [] |  |
| shot_0087 | positive | MISS | 180 | [] |  |
| shot_0088 | positive | MISS | 180 | [] |  |
| shot_0090 | positive | MISS | 180 | [275] |  |
| shot_0091 | positive | MADE | 180 | [] |  |
| shot_0092 | positive | MISS | 180 | [12] |  |
| shot_0093 | positive | MISS | 180 | [] |  |
| shot_0094 | positive | MADE | 180 | [] |  |
| shot_0095 | positive | MISS | 180 | [] |  |
| shot_0096 | positive | MADE | 180 | [75] |  |
| shot_0097 | positive | MISS | 180 | [] |  |
| shot_0098 | positive | MISS | 180 | [] |  |
| shot_0099 | positive | MADE | 180 | [51, 165] |  |
| shot_0102 | positive | MADE | 180 | [] |  |
| shot_0103 | positive | MADE | 180 | [117] |  |
| shot_0104 | positive | MISS | 180 | [] |  |
| shot_0105 | positive | MADE | 180 | [] |  |
| shot_0108 | positive | MADE | 180 | [] |  |
| shot_0110 | positive | MISS | 180 | [] |  |
| shot_0111 | positive | MISS | 180 | [] |  |
| shot_0112 | positive | MADE | 180 | [73] |  |
| shot_0113 | positive | MADE | 180 | [88] |  |
| shot_0115 | positive | MISS | 180 | [] |  |
| shot_0117 | positive | MISS | 180 | [] |  |
| shot_0118 | positive | MISS | 180 | [] |  |
| shot_0120 | positive | MISS | 180 | [51, 174] |  |
| shot_0121 | positive | MADE | 180 | [22] |  |
| shot_0122 | positive | MADE | 180 | [117] |  |
| shot_0123 | positive | MADE | 180 | [] |  |
| shot_0125 | positive | MADE | 180 | [116] |  |
| shot_0126 | positive | MADE | 180 | [] |  |
| shot_0127 | positive | MISS | 180 | [219] |  |
| shot_0129 | positive | MADE | 180 | [] |  |
| shot_0130 | positive | MADE | 180 | [174, 255] |  |
| shot_0131 | positive | MADE | 180 | [] |  |
| shot_0132 | positive | MADE | 180 | [282] |  |
| shot_0133 | positive | MISS | 180 | [] |  |
| shot_0134 | positive | MADE | 180 | [] |  |
| shot_0138 | positive | MADE | 180 | [] |  |
| shot_0142 | positive | MADE | 180 | [] |  |
| shot_0143 | positive | MADE | 180 | [] |  |
| shot_0144 | positive | MISS | 180 | [90] |  |
| shot_0145 | positive | MISS | 180 | [] |  |
| shot_0146 | positive | MADE | 180 | [218] |  |
| shot_0147 | positive | MISS | 180 | [134] |  |
| shot_0148 | positive | MADE | 180 | [176] |  |
| shot_0150 | positive | MADE | 180 | [50] |  |
| shot_0151 | positive | MADE | 180 | [] |  |
| shot_0152 | positive | MISS | 180 | [104, 207] |  |
| shot_0153 | positive | MISS | 180 | [] |  |
| shot_0154 | positive | MISS | 180 | [] |  |
| shot_0155 | positive | MADE | 180 | [53] |  |
| shot_0156 | positive | MISS | 180 | [] |  |
| shot_0157 | positive | MADE | 180 | [] |  |
| shot_0158 | positive | MISS | 180 | [184, 248] |  |
| shot_0159 | positive | MADE | 180 | [94] |  |
| shot_0160 | positive | MADE | 180 | [224] |  |
| shot_0161 | positive | MISS | 180 | [183] |  |
| shot_0163 | positive | MADE | 180 | [] |  |
| shot_0166 | positive | MADE | 180 | [] |  |
| shot_0168 | positive | MADE | 180 | [29, 77, 199] |  |
| shot_0169 | positive | MISS | 180 | [] |  |
| shot_0170 | positive | MISS | 180 | [275] |  |
| shot_0171 | positive | MISS | 180 | [] |  |
| shot_0172 | positive | MISS | 180 | [187] |  |
| shot_0173 | positive | MADE | 180 | [] |  |
| shot_0174 | positive | MADE | 180 | [] |  |
| shot_0175 | positive | MISS | 180 | [] |  |
| shot_0176 | positive | MADE | 180 | [] |  |
| shot_0178 | positive | MADE | 180 | [] |  |
| shot_0179 | positive | MADE | 180 | [44, 164] |  |
| shot_0180 | positive | MISS | 180 | [] |  |
| shot_0181 | positive | MISS | 180 | [94] |  |
| shot_0182 | positive | MADE | 180 | [] |  |
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
| full_neg_001 | negative | NO SHOT | — | [] |  |
| full_neg_003 | negative | NO SHOT | — | [174] |  |
| full_neg_004 | negative | NO SHOT | — | [] |  |
| full_neg_005 | negative | NO SHOT | — | [276] |  |
| full_neg_007 | negative | NO SHOT | — | [] |  |
| full_neg_008 | negative | NO SHOT | — | [] |  |
| full_neg_009 | negative | NO SHOT | — | [] |  |
| full_neg_010 | negative | NO SHOT | — | [] |  |
| full_neg_011 | negative | NO SHOT | — | [189] |  |
| full_neg_012 | negative | NO SHOT | — | [] |  |
| full_neg_013 | negative | NO SHOT | — | [] |  |

## Notes

- The PBP frame is the event clock, not the release frame; scorebug alignment corrects the video offset.
- Event-level metrics count extra predicted shots as false positives.
- The 28 negatives are non-shot clips with no PBP field goal in the window.
- This is one full game; the next step is to expand to all 186 PBP events and a second game.
