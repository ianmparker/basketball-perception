# Full Event-Detection Result

**Date:** 2026-10-22  
**Game:** `401810030` Knicks 137 – Timberwolves 114  
**Scope:** shot-event detection only, not made/missed and not release tracking.

## Dataset

- **162 positives**: 161 scorebug-timeline-aligned PBP field goals plus the
  audited `shot_0089`.
- **24 positives excluded**: no scorebug clock match. Most are quarter-end or
  period-boundary plays; see `all_timeline/validation_ocr.json`.
- **28 negatives**: 7 free throws, 12 turnovers, 7 fouls, 2 dead-ball clips.
  Each is a 10 s window with no PBP field goal.
- The negative set is **not balanced**: 10 s no-field-goal windows are scarce.
  Do not present this as a 186/186 benchmark.

## Clip-level event detection

| `MIN_RISE_PX` | Positive clips with any shot | Negative clips with any shot |
|---:|---:|---:|
| 60 | 144/162 (89%) | 15/28 (54%) |
| **80** | **135/162 (83%)** | **12/28 (43%)** |
| 100 | 109/162 (67%) | 9/28 (32%) |
| 120 (default) | 81/162 (50%) | 4/28 (14%) |

Rise80 is the candidate generator: it finds a shot in 83% of positive clips but
fires on 43% of negatives. The default is precision-first: 50% positive, 14%
negative.

## Event-level score

| Operating point | Precision | Recall | F1 |
|---|---:|---:|---:|
| default 120, ±3 s | 0.55 | 0.38 | 0.45 |
| rise80, ±2 s | 0.35 | 0.49 | 0.41 |
| rise80, ±3 s | 0.47 | 0.65 | 0.54 |
| rise60, ±3 s | 0.40 | 0.75 | 0.53 |

Full tables are in `full_threshold_sweep.md` and `full_scorecard.md`.

## Negative taxonomy

At rise80, 12/28 negatives fire:

| Negative class | Fired | Total |
|---|---:|---:|
| Free throws | **5** | 7 |
| Turnovers | 4 | 12 |
| Fouls | 3 | 7 |
| Dead ball | 0 | 2 |

At the default threshold, the four negative hits are 1 free throw, 2 turnovers,
and 1 foul. Free throws have the highest rate at rise80, but they are not the only
false-positive class; a general non-shot filter is still needed.

## Positive coverage by shot type at rise80

| Shot type | Clips with a candidate |
|---|---:|
| Three-pointer | 66/75 (88%) |
| Mid-range | 16/19 (84%) |
| Dunk | 9/11 (82%) |
| Layup | 20/25 (80%) |
| Other | 24/32 (75%) |

The detector is not strongly type-biased; the missed clips are spread across
types, with several blocks/closeouts and end-of-quarter plays.

## Candidate-filter experiment

The full rise80 result has 226 candidates. A leave-one-clip-out logistic regression
on the existing 19 features was trained with three label definitions:

| Label definition | Candidates | Positive | AUC | Event F1 at best threshold |
|---|---:|---:|---:|---:|
| PBP proximity (±2 s) | 226 | 90 | 0.566 | 0.49 |
| Clip-level shot/no-shot | 226 | 211 | 0.601 | 0.43 |
| Release window (−5.5/+1 s) | 226 | 159 | 0.686 | 0.45 |

The release-window label gives the best ranking, but it may be learning timing
rather than shot quality: positive candidates are defined by being in the release
window, and features such as ball height and velocity correlate with phase. These
event F1s are at ±2 s; raw rise80 is 0.39, so the gain is small and the AUC is
weak. **Do not adopt this verifier.** It is a useful signal that a general
non-shot filter may be learnable, but it needs a balanced negative set and a
held-out game.

The balanced-negative filter does better: 60 six-second no-FG windows give 246
candidates, LOO AUC 0.728, cross-quarter holdout AUC 0.717, and event F1 0.67 vs
raw 0.52 at ±3 s on held-out quarters. It is still one game and is not adopted;
see `non_shot_filter.md`.

See `verifier_full_proximity.md`, `verifier_full_clip.md`, and
`verifier_full_release.md`.

## Release-window sensitivity

Re-scoring the same cached predictions with a shifted window:

| Window | Precision | Recall | F1 |
|---|---:|---:|---:|
| symmetric ±3 s | 0.55 | 0.38 | 0.45 |
| release −5.5/+1 s | 0.62 | 0.43 | 0.50 |

This is an in-sample association result. The larger set reduces the apparent gain
from the pilot, which is why it needs a second-game check.

## What changed

1. **The pilot’s recall did not fully generalize.** Rise80 found 28/29 pilot
   positives; the full set finds 135/162 (83%). The harder clips include blocks,
   closeouts, and end-of-quarter actions.
2. **The negative set is cleaner but harder.** Default negative hits are 4/28
   (14%), not the pilot’s 1/17; rise80 is 12/28 (43%).
3. **Free throws are the top rate, not the whole problem.** Turnovers and fouls
   also fire; a single free-throw filter will not solve event specificity.
4. **The two-clock correction is real but smaller on the full set.** Release
   window F1 is 0.50 vs 0.45 at symmetric ±3 s.
5. **The event-detection problem is now measurable at scale.** The next step is a
   general non-shot verifier/filter and a balanced negative set, not more threshold
   tuning on 28 negatives.

## Next

1. Build a balanced negative set with 6 s windows or arbitrary no-FG windows from
   the full-game scorebug timeline.
2. Add a general non-shot verifier/filter; free-throw-only filtering is insufficient.
3. Resolve the 24 unaligned positives with per-event OCR fallback or an explicit
   exclusion/audit rule.
4. Validate the event detector on a second broadcast.
5. Keep made-shot classification and release tracking separate.
