# PBP Shot Verifier v1 — Candidate-Level Experiment

**Candidate generator:** `MIN_RISE_PX=80`  
**Candidates:** 36 (13 positive, 23 negative)  
**Clips:** 26  
**Leave-one-clip-out AUC:** 0.528

## Event-level after the verifier

| Verifier threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 8 | 11 | 3 | 0.42 | 0.73 | 0.53 |
| 0.4 | 7 | 9 | 5 | 0.44 | 0.58 | 0.50 |
| 0.5 | 6 | 6 | 9 | 0.50 | 0.40 | 0.44 |
| 0.6 | 5 | 6 | 10 | 0.46 | 0.33 | 0.39 |
| 0.7 | 4 | 4 | 13 | 0.50 | 0.23 | 0.32 |

## Verdict

**Null result.** The leave-one-clip-out AUC is at chance, so the verifier does
not rank true candidates above false ones on this feature set. Do not adopt it.
The threshold table is noise, not a tuning surface.

The candidate generator at rise80 already covers 20/20 positive clips; the
missing piece is a verifier that can use release/pose/trajectory evidence and
more than 42 candidates. Expanding to all 186 PBP events and a second game is
the prerequisite, not another model on these 30 clips.

## Top coefficients (full fit, not causal)

| Feature | Coefficient |
|---|---:|
| `is_layup` | +0.886 |
| `ball_speed_px` | -0.855 |
| `nearest_player_area` | -0.848 |
| `nearest_player_px` | +0.633 |
| `ball_near_hoop_px` | +0.369 |
| `is_three` | -0.368 |
| `is_mid` | -0.316 |
| `rise_px` | +0.308 |
| `ball_vert_frac` | -0.251 |
| `homog_inlier` | -0.150 |
| `pose_wrist_up` | -0.142 |
| `ball_height_frac` | +0.123 |

## Per-candidate

| Clip | Kind | Frame | Label | OOF probability |
|---|---|---:|---:|---:|
| shot_0008 | positive | 162 | 1 | 0.06 |
| shot_0011 | positive | 14 | 0 | 0.37 |
| shot_0011 | positive | 90 | 0 | 0.30 |
| shot_0011 | positive | 199 | 1 | 0.38 |
| shot_0011 | positive | 252 | 0 | 0.09 |
| shot_0013 | positive | 110 | 0 | 0.34 |
| shot_0013 | positive | 198 | 1 | 0.96 |
| shot_0018 | positive | 274 | 0 | 0.72 |
| shot_0020 | positive | 76 | 0 | 0.08 |
| shot_0020 | positive | 206 | 1 | 0.70 |
| shot_0022 | positive | 100 | 0 | 0.47 |
| shot_0022 | positive | 167 | 1 | 0.70 |
| shot_0026 | positive | 30 | 0 | 0.63 |
| shot_0026 | positive | 83 | 0 | 0.07 |
| shot_0057 | positive | 230 | 1 | 0.60 |
| shot_0061 | positive | 44 | 0 | 0.65 |
| shot_0061 | positive | 93 | 0 | 0.86 |
| shot_0061 | positive | 230 | 1 | 0.37 |
| shot_0089 | positive | 65 | 0 | 0.69 |
| shot_0100 | positive | 76 | 0 | 0.26 |
| shot_0100 | positive | 199 | 1 | 0.14 |
| shot_0101 | positive | 18 | 0 | 0.45 |
| shot_0106 | positive | 135 | 1 | 0.09 |
| shot_0107 | positive | 74 | 0 | 0.64 |
| shot_0107 | positive | 141 | 1 | 0.81 |
| shot_0124 | positive | 115 | 0 | 0.46 |
| shot_0124 | positive | 169 | 1 | 0.23 |
| shot_0139 | positive | 77 | 0 | 0.50 |
| shot_0141 | positive | 44 | 0 | 0.36 |
| shot_0149 | positive | 151 | 1 | 0.41 |
| shot_0164 | positive | 140 | 1 | 0.94 |
| neg_001 | negative | 44 | 0 | 0.25 |
| neg_002 | negative | 268 | 0 | 0.92 |
| neg_007 | negative | 56 | 0 | 0.58 |
| neg_007 | negative | 227 | 0 | 0.75 |
| neg_013 | negative | 40 | 0 | 0.34 |
