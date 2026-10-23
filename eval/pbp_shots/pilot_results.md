# Event-Detection Pilot

**Date:** 2026-10-22  
**Purpose:** validate the clip manifest, scorebug alignment, negative selection, and
event-detection metric before the 186-event run.

## Dataset

| Split | Count | Notes |
|---|---:|---|
| Positives | 29 | 19 previously audited PBP field goals + 10 new PBP field goals |
| Negatives | 17 | 4 strict existing negatives + 13 new non-shot clips |
| Total | 46 | 10 s clips, 30 fps, 640 px |

The new positives aligned **10/10** to the scorebug clock. The new negatives aligned
**13/16**; three were excluded as unaligned (`pilot_neg_006`, `pilot_neg_008`,
`pilot_neg_009`).

Negative mix: **6 free throws, 8 turnovers, 3 fouls**. Every negative was checked
against all PBP field goals in the clip window: no field goal may occur in the
10-second window.

## Clip-level event detection

| `MIN_RISE_PX` | Positive clips with any shot | Negative clips with any shot |
|---:|---:|---:|
| 60 | 27/29 | 8/17 |
| **80** | **28/29** | **6/17** |
| 100 | 20/29 | 4/17 |
| 120 (default) | 13/29 | 1/17 |

At rise80, the detector finds a candidate in **28 of 29** positive clips. The
remaining problem is specificity: **6 of 17** negatives also fire.

## Negative taxonomy at rise80

| Negative class | Clips with a shot | Total |
|---|---:|---:|
| Free throws | **4** | 6 |
| Turnovers | 1 | 8 |
| Fouls | 1 | 3 |

**4 of the 6 negative false positives are free throws.** This is the same
free-throw class seen in the Kyrie demo. A free-throw/scorebug state filter is the
highest-value next event-detection improvement.

At the default threshold, only one negative fires: a free throw.

### Oracle free-throw exclusion

As an upper bound, remove the six free-throw negatives:

- Rise80 negative clips with a shot: **2/11** (one turnover, one foul), down from
  6/17.
- Positive clips unchanged: **28/29**.
- Default negative clips: **0/11**.

This is not a product filter; it is the ceiling a free-throw filter could reach on
this pilot. The filter still needs to be built and validated.

## Event-level score

Cached-prediction event score (PBP-event windows):

| Window | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ±3 s (default 120) | 9 | 10 | 20 | 0.47 | 0.31 | 0.38 |
| ±2 s (rise80) | 18 | 35 | 11 | 0.34 | 0.62 | 0.44 |
| ±3 s (rise80) | 21 | 32 | 8 | 0.40 | 0.72 | 0.51 |

The rise80 candidate generator has good recall; the event-level precision is limited
by extra candidates and by the two-clock problem. See
`pilot_threshold_sweep.md`.

## Release-window sensitivity

Re-scoring the same cached predictions with the asymmetric **−5.5/+1 s** window:

| Window | Precision | Recall | F1 |
|---|---:|---:|---:|
| symmetric ±3 s | 0.47 | 0.31 | 0.38 |
| **release −5.5/+1 s** | **0.63** | **0.41** | **0.50** |

This is still an in-sample association result, but it is directionally consistent
with the 19-clip audit. See `pilot_release_timing.md`.

## What the pilot established

1. **The negative construction works.** Default negatives are **1/17**, versus
   4/7 on the old contaminated set; rise80 is 6/17.
2. **Free throws dominate rise80 false positives.** Four of six negative hits are
   free throws. The next filter should be a free-throw/scorebug state guard.
3. **Event detection is a candidate-generation problem, not a recall problem.**
   Rise80 finds 28/29 positive clips. The missing piece is specificity and dedupe.
4. **The new positive alignment is good.** 10/10 positives aligned. Negative
   alignment is 13/16; the three unaligned clips should be replaced or audited.
5. **The symmetric PBP-event window still understates release performance.**
   The two-clock correction remains necessary.

## Scaling note

A 10-second window with **no PBP field goal** is scarce. After excluding the pilot
and existing negatives, only 13 new clean negatives were available; with the 17
pilot negatives, the total is 30. The strict 10 s event-detection set is therefore
not a balanced 186-positive/186-negative dataset.

For the 186-event run, use one of these:

1. **Shorter clips** (for example 6 s) for both positives and negatives, which
   makes clean negative windows easier to find.
2. **Arbitrary no-FG windows** sampled across the game, aligned with the full-game
   scorebug clock index.
3. **A larger negative pool** from additional games.

Do not call the current 30-negative set a balanced 186-event benchmark.

## Next

1. Add a free-throw/scorebug state filter and re-score the pilot negatives.
2. Replace the three unaligned negatives; expand the clean negative pool.
3. Run the 186-event manifest with the same event-detection metric.
4. Only after event detection is stable, train the made-shot head on PBP outcomes.
5. Validate on a second broadcast before claiming a stable number.
