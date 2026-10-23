# PBP Shot Verifier v1 — Candidate-Level Experiment

**Candidate generator:** `MIN_RISE_PX=80`  
**Candidates:** 42 (13 positive, 29 negative)  
**Clips:** 30  
**Leave-one-clip-out AUC:** 0.512

## Event-level after the verifier

| Verifier threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 7 | 17 | 2 | 0.29 | 0.78 | 0.42 |
| 0.4 | 7 | 12 | 6 | 0.37 | 0.54 | 0.44 |
| 0.5 | 6 | 7 | 10 | 0.46 | 0.38 | 0.41 |
| 0.6 | 6 | 6 | 11 | 0.50 | 0.35 | 0.41 |
| 0.7 | 3 | 4 | 15 | 0.43 | 0.17 | 0.24 |

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
| `is_layup` | +0.937 |
| `nearest_player_area` | -0.811 |
| `ball_speed_px` | -0.759 |
| `nearest_player_px` | +0.686 |
| `ball_vert_frac` | -0.437 |
| `is_three` | -0.383 |
| `ball_near_hoop_px` | +0.331 |
| `is_mid` | -0.309 |
| `rise_px` | +0.212 |
| `ball_height_frac` | +0.194 |
| `pose_wrist_up` | -0.115 |
| `homog_inlier` | -0.107 |

## Per-candidate

| Clip | Kind | Frame | Label | OOF probability |
|---|---|---:|---:|---:|
| shot_0008 | positive | 162 | 1 | 0.07 |
| shot_0011 | positive | 14 | 0 | 0.22 |
| shot_0011 | positive | 90 | 0 | 0.35 |
| shot_0011 | positive | 199 | 1 | 0.29 |
| shot_0011 | positive | 252 | 0 | 0.06 |
| shot_0013 | positive | 110 | 0 | 0.25 |
| shot_0013 | positive | 198 | 1 | 0.98 |
| shot_0018 | positive | 274 | 0 | 0.72 |
| shot_0020 | positive | 76 | 0 | 0.10 |
| shot_0020 | positive | 206 | 1 | 0.64 |
| shot_0022 | positive | 100 | 0 | 0.53 |
| shot_0022 | positive | 167 | 1 | 0.66 |
| shot_0026 | positive | 30 | 0 | 0.55 |
| shot_0026 | positive | 83 | 0 | 0.03 |
| shot_0041 | positive | 116 | 0 | 0.33 |
| shot_0057 | positive | 230 | 1 | 0.69 |
| shot_0061 | positive | 44 | 0 | 0.54 |
| shot_0061 | positive | 93 | 0 | 0.91 |
| shot_0061 | positive | 230 | 1 | 0.24 |
| shot_0089 | positive | 65 | 0 | 0.67 |
| shot_0100 | positive | 76 | 0 | 0.31 |
| shot_0100 | positive | 199 | 1 | 0.20 |
| shot_0101 | positive | 18 | 0 | 0.44 |
| shot_0106 | positive | 135 | 1 | 0.06 |
| shot_0107 | positive | 74 | 0 | 0.56 |
| shot_0107 | positive | 141 | 1 | 0.87 |
| shot_0124 | positive | 115 | 0 | 0.40 |
| shot_0124 | positive | 169 | 1 | 0.23 |
| shot_0139 | positive | 77 | 0 | 0.46 |
| shot_0141 | positive | 44 | 0 | 0.39 |
| shot_0149 | positive | 151 | 1 | 0.42 |
| shot_0164 | positive | 140 | 1 | 0.92 |
| neg_001 | negative | 44 | 0 | 0.26 |
| neg_002 | negative | 268 | 0 | 0.93 |
| neg_003 | negative | 238 | 0 | 0.39 |
| neg_006 | negative | 3 | 0 | 0.74 |
| neg_006 | negative | 234 | 0 | 0.65 |
| neg_007 | negative | 56 | 0 | 0.34 |
| neg_007 | negative | 227 | 0 | 0.63 |
| neg_013 | negative | 40 | 0 | 0.48 |
| neg_014 | negative | 203 | 0 | 0.44 |
| neg_014 | negative | 252 | 0 | 0.20 |
