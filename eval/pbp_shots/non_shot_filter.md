# Non-Shot Filter

**Date:** 2026-10-22  
**Status:** promising, **not adopted** — one game only.

## Dataset

- Full positive set: **162** aligned clips.
- Balanced negatives: **60** six-second live-play windows, 15 per quarter, each
  with no PBP field goal.
- Rise80 candidates: **246** — 211 from positive clips, 35 from balanced negatives.
- Labels: clip-level. Every candidate in a positive clip is positive; every
  candidate in a balanced negative clip is negative.

The negative set is still small and the positive candidate labels include duplicate
phases, so this is a screening experiment, not a deployment decision.

## Ranking

| Evaluation | AUC |
|---|---:|
| Leave-one-clip-out, all 222 clips | 0.728 |
| Cross-quarter holdout, Q1 | 0.672 |
| Cross-quarter holdout, Q2 | 0.722 |
| Cross-quarter holdout, Q3 | 0.730 |
| Cross-quarter holdout, Q4 | 0.866 |
| **All held-out quarters** | **0.717** |

The cross-quarter AUC is the more honest number. It is stable around 0.72, with Q1
the weakest quarter.

## Event-level result

Cross-quarter holdout, verifier threshold 0.3:

| Window | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ±3 s filtered | 92 | 53 | 36 | 0.63 | 0.72 | **0.67** |
| ±3 s raw rise80 | 105 | 141 | 57 | 0.43 | 0.65 | 0.52 |
| ±2 s raw rise80 | 79 | 167 | 83 | 0.32 | 0.49 | 0.39 |

The threshold table is not a tuning surface; 0.3–0.4 work similarly. A threshold
must be frozen before the second broadcast.

## What this means

1. A general non-shot filter is now plausible: AUC 0.717 and event F1 0.67 vs raw
   0.52 at ±3 s on held-out quarters.
2. The filter is not validated across games. The same broadcast/scorebug style and
   the same features could still leak context.
3. The positive labels are clip-level and include duplicates; the filter may be
   learning clip context rather than shot quality. Release labels should be
   checked separately.
4. The 60 balanced negatives are a useful hard set, but only 35 candidates; more
   negatives and a second broadcast are required.

## Next

1. Freeze the feature list and threshold 0.3.
2. Validate on the second NBA broadcast before adopting.
3. Add release/pose/trajectory features only after that validation.
4. Keep this filter separate from made-shot classification.

Artifacts: `non_shot_filter_clip.md`, `non_shot_filter_clip_3s.md`,
`non_shot_filter_release.md`, `non_shot_filter_holdout.md`.
