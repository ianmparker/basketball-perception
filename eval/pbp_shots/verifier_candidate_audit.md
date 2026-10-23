# PBP Shot Verifier — Candidate Audit

**Candidate generator:** `MIN_RISE_PX=80`  
**Labels:** human candidate audit  
**Candidates:** 25 (20 positive, 5 negative)  
**Clips:** 26  
**Leave-one-clip-out AUC:** 0.270

## Event-level after the verifier

| Verifier threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 3 | 10 | 8 | 0.23 | 0.27 | 0.25 |
| 0.4 | 3 | 10 | 8 | 0.23 | 0.27 | 0.25 |
| 0.5 | 3 | 10 | 8 | 0.23 | 0.27 | 0.25 |
| 0.6 | 3 | 9 | 8 | 0.25 | 0.27 | 0.26 |
| 0.7 | 3 | 7 | 10 | 0.30 | 0.23 | 0.26 |

## Verdict

**Below chance.** The verifier ranks non-shots above shots on this label set. With very few negative candidates, this is not a usable ranking; the label set is too small and imbalanced to fit a verifier.

This is a label-quality experiment, not a product model. The candidate set is
small and imbalanced; the next step is more audited candidates, a balanced
negative set, and a second broadcast, not threshold tuning.

## Top coefficients (full fit, not causal)

| Feature | Coefficient |
|---|---:|
| `event_confidence` | +1.032 |
| `scene_cut_near` | +0.835 |
| `is_three` | +0.643 |
| `ball_height_frac` | -0.468 |
| `ball_vert_frac` | +0.450 |
| `det_distance_ft` | +0.397 |
| `rise_px` | +0.370 |
| `is_mid` | -0.358 |
| `is_layup` | -0.332 |
| `ball_speed_px` | -0.282 |
| `ball_near_hoop_px` | -0.280 |
| `nearest_player_px` | -0.176 |

## Per-candidate

| Candidate | Clip | Kind | Frame | Label | OOF probability |
|---|---|---|---:|---:|---:|
| shot_0011_14 | shot_0011 | positive | 14 | 1 | 0.98 |
| shot_0011_90 | shot_0011 | positive | 90 | 1 | 0.81 |
| shot_0013_110 | shot_0013 | positive | 110 | 1 | 0.98 |
| shot_0020_76 | shot_0020 | positive | 76 | 1 | 0.98 |
| shot_0022_100 | shot_0022 | positive | 100 | 1 | 0.09 |
| shot_0022_167 | shot_0022 | positive | 167 | 1 | 0.04 |
| shot_0057_230 | shot_0057 | positive | 230 | 0 | 0.96 |
| shot_0061_44 | shot_0061 | positive | 44 | 1 | 0.53 |
| shot_0061_93 | shot_0061 | positive | 93 | 1 | 0.53 |
| shot_0061_230 | shot_0061 | positive | 230 | 0 | 0.70 |
| shot_0089_65 | shot_0089 | positive | 65 | 1 | 0.64 |
| shot_0100_76 | shot_0100 | positive | 76 | 1 | 0.97 |
| shot_0100_199 | shot_0100 | positive | 199 | 0 | 0.96 |
| shot_0101_18 | shot_0101 | positive | 18 | 1 | 0.63 |
| shot_0106_135 | shot_0106 | positive | 135 | 1 | 0.99 |
| shot_0107_74 | shot_0107 | positive | 74 | 1 | 0.97 |
| shot_0107_141 | shot_0107 | positive | 141 | 0 | 0.97 |
| shot_0124_115 | shot_0124 | positive | 115 | 1 | 0.84 |
| shot_0124_169 | shot_0124 | positive | 169 | 1 | 0.58 |
| shot_0139_77 | shot_0139 | positive | 77 | 1 | 0.29 |
| shot_0141_44 | shot_0141 | positive | 44 | 1 | 0.02 |
| shot_0149_151 | shot_0149 | positive | 151 | 1 | 0.18 |
| shot_0164_140 | shot_0164 | positive | 140 | 1 | 0.03 |
| neg_001_44 | neg_001 | negative | 44 | 1 | 0.57 |
| neg_007_56 | neg_007 | negative | 56 | 0 | 0.99 |
