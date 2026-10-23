# Broadcast demo evaluation

The demo page can say “2 shots, 5 passes” without saying whether those events are
real. This folder turns the demo into a number.

## What is here

| File | What it is |
|---|---|
| `scorecard.md` | The demo and failure record: what works, what is wrong, what to fix next. |
| `event_scorecard.md` | Event-level precision/recall/F1 at ±5/15/30 frames, with confidence intervals. |

The raw `manifest.json`, `labels.json`, and per-run `event_scorecard.json` are not
distributed; they reference research-only footage. The committed markdown files are the
audit trail. See `docs/DATA_POLICY.md`.

## Regimes

The runs are not the same camera regime:

| Regime | Use |
|---|---|
| `nba_broadcast` | The broadcast demo and its candidate precision |
| `hs_top_down` | A Phase B transfer probe, not a broadcast score |

Do not compare the broadcast number with the top-down probe. A good broadcast result
does not imply a good fixed-camera high-school result.

## Current status

Seven candidates were labeled by the owner: **4 real shots and 3 false positives**.
The broadcast-only subset is **4/5 = 0.80** (Wilson 95% CI 0.38–0.96); the top-down
probe is **0/2** and is read separately. Recall is not measured until missed shots are
added to the label set.

The false-positive taxonomy is in `scorecard.md`: a free throw, a pass/drive, and a
rim-area pass/rebound with a camera cut. Those are the categories the next detector
must handle.

## Label your own clips

```bash
python scripts/label_broadcast_events.py --port 8766
```

Open <http://127.0.0.1:8766/>. The page plays one candidate clip at a time and writes
labels to the local `labels.json` (gitignored because it points at footage).

- `1` = real shot, `0` = false positive, `S` = skip/unsure.
- `R` = replay, arrow keys = previous/next.
- Optional fields: release frame, shooter track, made/missed, notes.

After labeling, click **Score now** on the page, or run the scorer:

```bash
python scripts/score_broadcast_events.py
```

This writes `event_scorecard.md` and `event_scorecard.json`.

## How scoring works

- A true positive matches a real shot to a prediction within the tolerance.
- A false positive is a labeled false-positive candidate that still has a prediction.
- A false negative is a real shot with no prediction in tolerance.
- Offsets use `release_frame` only; a label without a release frame counts as a match
  but does not contribute to the offset distribution.
- With seven events, the confidence intervals are wide. Report the counts and the
  interval, not just F1.

## Why this exists

A demo is a candidate view, not an accuracy claim. The labels turn the demo into a
measurable claim, and the failure record keeps the claim honest.
