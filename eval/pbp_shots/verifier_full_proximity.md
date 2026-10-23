# PBP Shot Verifier v1 — Candidate-Level Experiment

**Candidate generator:** `MIN_RISE_PX=80`  
**Labels:** selected-PBP proximity  
**Candidates:** 226 (90 positive, 136 negative)  
**Clips:** 190  
**Leave-one-clip-out AUC:** 0.566

## Event-level after the verifier

| Verifier threshold | TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.3 | 56 | 76 | 40 | 0.42 | 0.58 | 0.49 |
| 0.4 | 51 | 66 | 54 | 0.44 | 0.49 | 0.46 |
| 0.5 | 37 | 53 | 76 | 0.41 | 0.33 | 0.36 |
| 0.6 | 25 | 31 | 109 | 0.45 | 0.19 | 0.26 |
| 0.7 | 11 | 11 | 142 | 0.50 | 0.07 | 0.13 |

## Verdict

**Weak signal.** The audited candidate labels show a small separation, but it is not strong enough to adopt. Add data before tuning.

This is a label-quality experiment, not a product model. The candidate set is
small and imbalanced; the next step is more audited candidates, a balanced
negative set, and a second broadcast, not threshold tuning.

## Top coefficients (full fit, not causal)

| Feature | Coefficient |
|---|---:|
| `is_mid` | +0.530 |
| `ball_near_hoop_px` | -0.528 |
| `event_confidence` | -0.450 |
| `is_three` | +0.365 |
| `homog_inlier` | -0.343 |
| `rise_px` | -0.313 |
| `nearest_player_px` | +0.305 |
| `ball_height_frac` | +0.238 |
| `is_layup` | +0.229 |
| `nearest_player_area` | -0.156 |
| `ball_speed_px` | +0.123 |
| `det_distance_ft` | -0.106 |

## Per-candidate

| Candidate | Clip | Kind | Frame | Label | OOF probability |
|---|---|---|---:|---:|---:|
| shot_0008_162 | shot_0008 | positive | 162 | 1 | 0.57 |
| shot_0011_14 | shot_0011 | positive | 14 | 0 | 0.14 |
| shot_0011_90 | shot_0011 | positive | 90 | 0 | 0.59 |
| shot_0011_199 | shot_0011 | positive | 199 | 1 | 0.50 |
| shot_0011_252 | shot_0011 | positive | 252 | 0 | 0.15 |
| shot_0013_110 | shot_0013 | positive | 110 | 0 | 0.38 |
| shot_0013_198 | shot_0013 | positive | 198 | 1 | 0.48 |
| shot_0018_274 | shot_0018 | positive | 274 | 0 | 0.39 |
| shot_0020_76 | shot_0020 | positive | 76 | 0 | 0.49 |
| shot_0020_206 | shot_0020 | positive | 206 | 1 | 0.31 |
| shot_0022_100 | shot_0022 | positive | 100 | 0 | 0.24 |
| shot_0022_167 | shot_0022 | positive | 167 | 1 | 0.41 |
| shot_0026_30 | shot_0026 | positive | 30 | 0 | 0.59 |
| shot_0026_83 | shot_0026 | positive | 83 | 0 | 0.34 |
| shot_0057_230 | shot_0057 | positive | 230 | 1 | 0.23 |
| shot_0061_44 | shot_0061 | positive | 44 | 0 | 0.23 |
| shot_0061_93 | shot_0061 | positive | 93 | 0 | 0.37 |
| shot_0061_230 | shot_0061 | positive | 230 | 1 | 0.47 |
| shot_0089_65 | shot_0089 | positive | 65 | 0 | 0.62 |
| shot_0100_76 | shot_0100 | positive | 76 | 0 | 0.24 |
| shot_0100_199 | shot_0100 | positive | 199 | 1 | 0.29 |
| shot_0101_18 | shot_0101 | positive | 18 | 0 | 0.46 |
| shot_0106_135 | shot_0106 | positive | 135 | 1 | 0.22 |
| shot_0107_74 | shot_0107 | positive | 74 | 0 | 0.28 |
| shot_0107_141 | shot_0107 | positive | 141 | 1 | 0.47 |
| shot_0124_115 | shot_0124 | positive | 115 | 0 | 0.55 |
| shot_0124_169 | shot_0124 | positive | 169 | 1 | 0.51 |
| shot_0139_77 | shot_0139 | positive | 77 | 0 | 0.61 |
| shot_0141_44 | shot_0141 | positive | 44 | 0 | 0.26 |
| shot_0149_151 | shot_0149 | positive | 151 | 1 | 0.39 |
| shot_0164_140 | shot_0164 | positive | 140 | 1 | 0.66 |
| shot_0021_49 | shot_0021 | positive | 49 | 0 | 0.59 |
| shot_0021_103 | shot_0021 | positive | 103 | 0 | 0.41 |
| shot_0021_152 | shot_0021 | positive | 152 | 1 | 0.61 |
| shot_0024_266 | shot_0024 | positive | 266 | 0 | 0.58 |
| shot_0023_141 | shot_0023 | positive | 141 | 1 | 0.28 |
| shot_0023_198 | shot_0023 | positive | 198 | 1 | 0.50 |
| shot_0063_123 | shot_0063 | positive | 123 | 1 | 0.68 |
| shot_0063_204 | shot_0063 | positive | 204 | 1 | 0.66 |
| shot_0064_266 | shot_0064 | positive | 266 | 0 | 0.11 |
| shot_0116_177 | shot_0116 | positive | 177 | 1 | 0.60 |
| shot_0109_217 | shot_0109 | positive | 217 | 1 | 0.40 |
| shot_0114_98 | shot_0114 | positive | 98 | 0 | 0.76 |
| shot_0165_49 | shot_0165 | positive | 49 | 0 | 0.28 |
| shot_0001_119 | shot_0001 | positive | 119 | 0 | 0.73 |
| shot_0001_204 | shot_0001 | positive | 204 | 1 | 0.37 |
| shot_0002_93 | shot_0002 | positive | 93 | 0 | 0.18 |
| shot_0002_171 | shot_0002 | positive | 171 | 1 | 0.73 |
| shot_0003_86 | shot_0003 | positive | 86 | 0 | 0.53 |
| shot_0003_173 | shot_0003 | positive | 173 | 1 | 0.63 |
| shot_0003_240 | shot_0003 | positive | 240 | 1 | 0.49 |
| shot_0004_194 | shot_0004 | positive | 194 | 1 | 0.39 |
| shot_0005_134 | shot_0005 | positive | 134 | 1 | 0.67 |
| shot_0005_266 | shot_0005 | positive | 266 | 0 | 0.71 |
| shot_0006_8 | shot_0006 | positive | 8 | 0 | 0.30 |
| shot_0006_131 | shot_0006 | positive | 131 | 1 | 0.59 |
| shot_0007_80 | shot_0007 | positive | 80 | 0 | 0.58 |
| shot_0007_262 | shot_0007 | positive | 262 | 0 | 0.41 |
| shot_0009_66 | shot_0009 | positive | 66 | 0 | 0.34 |
| shot_0010_157 | shot_0010 | positive | 157 | 1 | 0.46 |
| shot_0012_97 | shot_0012 | positive | 97 | 0 | 0.24 |
| shot_0012_291 | shot_0012 | positive | 291 | 0 | 0.28 |
| shot_0014_228 | shot_0014 | positive | 228 | 1 | 0.25 |
| shot_0017_150 | shot_0017 | positive | 150 | 1 | 0.47 |
| shot_0025_191 | shot_0025 | positive | 191 | 1 | 0.62 |
| shot_0027_127 | shot_0027 | positive | 127 | 1 | 0.72 |
| shot_0027_193 | shot_0027 | positive | 193 | 1 | 0.46 |
| shot_0027_256 | shot_0027 | positive | 256 | 0 | 0.48 |
| shot_0028_111 | shot_0028 | positive | 111 | 0 | 0.62 |
| shot_0028_190 | shot_0028 | positive | 190 | 1 | 0.48 |
| shot_0029_17 | shot_0029 | positive | 17 | 0 | 0.38 |
| shot_0030_122 | shot_0030 | positive | 122 | 1 | 0.42 |
| shot_0030_236 | shot_0030 | positive | 236 | 1 | 0.59 |
| shot_0031_164 | shot_0031 | positive | 164 | 1 | 0.75 |
| shot_0031_211 | shot_0031 | positive | 211 | 1 | 0.57 |
| shot_0031_294 | shot_0031 | positive | 294 | 0 | 0.69 |
| shot_0032_40 | shot_0032 | positive | 40 | 0 | 0.68 |
| shot_0032_140 | shot_0032 | positive | 140 | 1 | 0.33 |
| shot_0033_150 | shot_0033 | positive | 150 | 1 | 0.66 |
| shot_0034_170 | shot_0034 | positive | 170 | 1 | 0.78 |
| shot_0035_110 | shot_0035 | positive | 110 | 0 | 0.53 |
| shot_0036_38 | shot_0036 | positive | 38 | 0 | 0.66 |
| shot_0044_146 | shot_0044 | positive | 146 | 1 | 0.64 |
| shot_0045_21 | shot_0045 | positive | 21 | 0 | 0.17 |
| shot_0045_239 | shot_0045 | positive | 239 | 1 | 0.63 |
| shot_0047_11 | shot_0047 | positive | 11 | 0 | 0.31 |
| shot_0047_169 | shot_0047 | positive | 169 | 1 | 0.52 |
| shot_0047_244 | shot_0047 | positive | 244 | 0 | 0.73 |
| shot_0048_194 | shot_0048 | positive | 194 | 1 | 0.21 |
| shot_0049_161 | shot_0049 | positive | 161 | 1 | 0.14 |
| shot_0050_111 | shot_0050 | positive | 111 | 0 | 0.63 |
| shot_0051_227 | shot_0051 | positive | 227 | 1 | 0.77 |
| shot_0053_188 | shot_0053 | positive | 188 | 1 | 0.50 |
| shot_0055_119 | shot_0055 | positive | 119 | 0 | 0.37 |
| shot_0055_166 | shot_0055 | positive | 166 | 1 | 0.50 |
| shot_0056_30 | shot_0056 | positive | 30 | 0 | 0.36 |
| shot_0056_106 | shot_0056 | positive | 106 | 0 | 0.81 |
| shot_0058_190 | shot_0058 | positive | 190 | 1 | 0.46 |
| shot_0059_123 | shot_0059 | positive | 123 | 1 | 0.56 |
| shot_0060_118 | shot_0060 | positive | 118 | 0 | 0.46 |
| shot_0062_183 | shot_0062 | positive | 183 | 1 | 0.38 |
| shot_0065_80 | shot_0065 | positive | 80 | 0 | 0.57 |
| shot_0065_191 | shot_0065 | positive | 191 | 1 | 0.41 |
| shot_0066_54 | shot_0066 | positive | 54 | 0 | 0.27 |
| shot_0066_133 | shot_0066 | positive | 133 | 1 | 0.32 |
| shot_0066_237 | shot_0066 | positive | 237 | 1 | 0.55 |
| shot_0066_285 | shot_0066 | positive | 285 | 0 | 0.46 |
| shot_0067_97 | shot_0067 | positive | 97 | 0 | 0.30 |
| shot_0068_69 | shot_0068 | positive | 69 | 0 | 0.23 |
| shot_0068_186 | shot_0068 | positive | 186 | 1 | 0.72 |
| shot_0069_6 | shot_0069 | positive | 6 | 0 | 0.58 |
| shot_0070_250 | shot_0070 | positive | 250 | 0 | 0.54 |
| shot_0071_176 | shot_0071 | positive | 176 | 1 | 0.39 |
| shot_0072_27 | shot_0072 | positive | 27 | 0 | 0.72 |
| shot_0073_113 | shot_0073 | positive | 113 | 0 | 0.48 |
| shot_0074_90 | shot_0074 | positive | 90 | 0 | 0.35 |
| shot_0074_185 | shot_0074 | positive | 185 | 1 | 0.65 |
| shot_0074_230 | shot_0074 | positive | 230 | 1 | 0.28 |
| shot_0074_277 | shot_0074 | positive | 277 | 0 | 0.18 |
| shot_0075_101 | shot_0075 | positive | 101 | 0 | 0.50 |
| shot_0075_186 | shot_0075 | positive | 186 | 1 | 0.63 |
| shot_0076_101 | shot_0076 | positive | 101 | 0 | 0.27 |
| shot_0076_153 | shot_0076 | positive | 153 | 1 | 0.55 |
| shot_0076_277 | shot_0076 | positive | 277 | 0 | 0.52 |
| shot_0077_108 | shot_0077 | positive | 108 | 0 | 0.19 |
| shot_0077_277 | shot_0077 | positive | 277 | 0 | 0.87 |
| shot_0078_36 | shot_0078 | positive | 36 | 0 | 0.63 |
| shot_0078_249 | shot_0078 | positive | 249 | 0 | 0.60 |
| shot_0079_41 | shot_0079 | positive | 41 | 0 | 0.43 |
| shot_0080_44 | shot_0080 | positive | 44 | 0 | 0.65 |
| shot_0080_216 | shot_0080 | positive | 216 | 1 | 0.28 |
| shot_0086_144 | shot_0086 | positive | 144 | 1 | 0.68 |
| shot_0087_163 | shot_0087 | positive | 163 | 1 | 0.76 |
| shot_0090_271 | shot_0090 | positive | 271 | 0 | 0.54 |
| shot_0092_8 | shot_0092 | positive | 8 | 0 | 0.40 |
| shot_0095_87 | shot_0095 | positive | 87 | 0 | 0.17 |
| shot_0095_285 | shot_0095 | positive | 285 | 0 | 0.52 |
| shot_0096_75 | shot_0096 | positive | 75 | 0 | 0.54 |
| shot_0099_23 | shot_0099 | positive | 23 | 0 | 0.33 |
| shot_0099_82 | shot_0099 | positive | 82 | 0 | 0.42 |
| shot_0099_165 | shot_0099 | positive | 165 | 1 | 0.33 |
| shot_0102_114 | shot_0102 | positive | 114 | 0 | 0.60 |
| shot_0103_117 | shot_0103 | positive | 117 | 0 | 0.69 |
| shot_0104_144 | shot_0104 | positive | 144 | 1 | 0.76 |
| shot_0105_73 | shot_0105 | positive | 73 | 0 | 0.34 |
| shot_0110_299 | shot_0110 | positive | 299 | 0 | 0.62 |
| shot_0111_58 | shot_0111 | positive | 58 | 0 | 0.61 |
| shot_0111_192 | shot_0111 | positive | 192 | 1 | 0.60 |
| shot_0112_68 | shot_0112 | positive | 68 | 0 | 0.52 |
| shot_0112_117 | shot_0112 | positive | 117 | 0 | 0.50 |
| shot_0113_40 | shot_0113 | positive | 40 | 0 | 0.50 |
| shot_0113_87 | shot_0113 | positive | 87 | 0 | 0.31 |
| shot_0115_41 | shot_0115 | positive | 41 | 0 | 0.64 |
| shot_0115_272 | shot_0115 | positive | 272 | 0 | 0.64 |
| shot_0117_87 | shot_0117 | positive | 87 | 0 | 0.69 |
| shot_0118_183 | shot_0118 | positive | 183 | 1 | 0.45 |
| shot_0118_230 | shot_0118 | positive | 230 | 1 | 0.44 |
| shot_0120_51 | shot_0120 | positive | 51 | 0 | 0.47 |
| shot_0120_126 | shot_0120 | positive | 126 | 1 | 0.43 |
| shot_0120_174 | shot_0120 | positive | 174 | 1 | 0.71 |
| shot_0121_22 | shot_0121 | positive | 22 | 0 | 0.62 |
| shot_0121_84 | shot_0121 | positive | 84 | 0 | 0.60 |
| shot_0122_109 | shot_0122 | positive | 109 | 0 | 0.19 |
| shot_0125_44 | shot_0125 | positive | 44 | 0 | 0.58 |
| shot_0125_110 | shot_0125 | positive | 110 | 0 | 0.48 |
| shot_0126_294 | shot_0126 | positive | 294 | 0 | 0.33 |
| shot_0127_218 | shot_0127 | positive | 218 | 1 | 0.68 |
| shot_0129_152 | shot_0129 | positive | 152 | 1 | 0.59 |
| shot_0130_174 | shot_0130 | positive | 174 | 1 | 0.50 |
| shot_0130_226 | shot_0130 | positive | 226 | 1 | 0.62 |
| shot_0132_282 | shot_0132 | positive | 282 | 0 | 0.80 |
| shot_0133_29 | shot_0133 | positive | 29 | 0 | 0.24 |
| shot_0133_124 | shot_0133 | positive | 124 | 1 | 0.48 |
| shot_0144_89 | shot_0144 | positive | 89 | 0 | 0.37 |
| shot_0145_253 | shot_0145 | positive | 253 | 0 | 0.47 |
| shot_0146_212 | shot_0146 | positive | 212 | 1 | 0.42 |
| shot_0147_80 | shot_0147 | positive | 80 | 0 | 0.58 |
| shot_0147_132 | shot_0147 | positive | 132 | 1 | 0.30 |
| shot_0148_168 | shot_0148 | positive | 168 | 1 | 0.43 |
| shot_0150_46 | shot_0150 | positive | 46 | 0 | 0.51 |
| shot_0151_212 | shot_0151 | positive | 212 | 1 | 0.40 |
| shot_0152_106 | shot_0152 | positive | 106 | 0 | 0.57 |
| shot_0152_206 | shot_0152 | positive | 206 | 1 | 0.80 |
| shot_0155_50 | shot_0155 | positive | 50 | 0 | 0.57 |
| shot_0155_103 | shot_0155 | positive | 103 | 0 | 0.28 |
| shot_0155_231 | shot_0155 | positive | 231 | 1 | 0.51 |
| shot_0156_107 | shot_0156 | positive | 107 | 0 | 0.25 |
| shot_0157_20 | shot_0157 | positive | 20 | 0 | 0.36 |
| shot_0158_56 | shot_0158 | positive | 56 | 0 | 0.59 |
| shot_0158_101 | shot_0158 | positive | 101 | 0 | 0.61 |
| shot_0158_184 | shot_0158 | positive | 184 | 1 | 0.62 |
| shot_0158_248 | shot_0158 | positive | 248 | 0 | 0.66 |
| shot_0159_65 | shot_0159 | positive | 65 | 0 | 0.36 |
| shot_0159_180 | shot_0159 | positive | 180 | 1 | 0.68 |
| shot_0160_223 | shot_0160 | positive | 223 | 1 | 0.86 |
| shot_0161_174 | shot_0161 | positive | 174 | 1 | 0.55 |
| shot_0163_261 | shot_0163 | positive | 261 | 0 | 0.30 |
| shot_0166_68 | shot_0166 | positive | 68 | 0 | 0.43 |
| shot_0168_29 | shot_0168 | positive | 29 | 0 | 0.47 |
| shot_0168_77 | shot_0168 | positive | 77 | 0 | 0.59 |
| shot_0168_198 | shot_0168 | positive | 198 | 1 | 0.40 |
| shot_0169_119 | shot_0169 | positive | 119 | 0 | 0.75 |
| shot_0170_263 | shot_0170 | positive | 263 | 0 | 0.62 |
| shot_0171_163 | shot_0171 | positive | 163 | 1 | 0.55 |
| shot_0172_185 | shot_0172 | positive | 185 | 1 | 0.49 |
| shot_0172_262 | shot_0172 | positive | 262 | 0 | 0.49 |
| shot_0173_241 | shot_0173 | positive | 241 | 0 | 0.54 |
| shot_0176_78 | shot_0176 | positive | 78 | 0 | 0.57 |
| shot_0179_41 | shot_0179 | positive | 41 | 0 | 0.63 |
| shot_0179_157 | shot_0179 | positive | 157 | 1 | 0.43 |
| shot_0181_94 | shot_0181 | positive | 94 | 0 | 0.65 |
| neg_007_56 | neg_007 | negative | 56 | 0 | 0.75 |
| neg_007_227 | neg_007 | negative | 227 | 0 | 0.49 |
| pilot_neg_001_27 | pilot_neg_001 | negative | 27 | 0 | 0.42 |
| pilot_neg_002_20 | pilot_neg_002 | negative | 20 | 0 | 0.54 |
| pilot_neg_002_77 | pilot_neg_002 | negative | 77 | 0 | 0.46 |
| pilot_neg_003_169 | pilot_neg_003 | negative | 169 | 0 | 0.48 |
| pilot_neg_004_79 | pilot_neg_004 | negative | 79 | 0 | 0.40 |
| pilot_neg_013_9 | pilot_neg_013 | negative | 9 | 0 | 0.83 |
| pilot_neg_013_64 | pilot_neg_013 | negative | 64 | 0 | 0.37 |
| full_neg_001_195 | full_neg_001 | negative | 195 | 0 | 0.43 |
| full_neg_003_172 | full_neg_003 | negative | 172 | 0 | 0.32 |
| full_neg_005_280 | full_neg_005 | negative | 280 | 0 | 0.28 |
| full_neg_007_223 | full_neg_007 | negative | 223 | 0 | 0.23 |
| full_neg_008_278 | full_neg_008 | negative | 278 | 0 | 0.63 |
| full_neg_011_178 | full_neg_011 | negative | 178 | 0 | 0.50 |
