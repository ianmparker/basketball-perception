# PBP Shot Benchmark

This is the event-level shot benchmark. It is separate from the small visual demo.

## Source

- One full NBA regular-season broadcast (research-only; not included in this repo).
- Play-by-play: 186 field-goal attempts with frame, made/missed, shot type, distance,
  shooter, team, and next rebound. The raw PBP extract and per-clip `ClipResult` JSON
  are not distributed; the aggregate scorecards are.
- Benchmark v1 after the alignment audit: **19 positives** and **7 clean negatives**,
  10 s clips at 30 fps.
- The candidate-level audit found one of those seven negatives (`neg_001`) contains a
  real shot and two more (`neg_002`, `neg_013`) are unverified. See
  `candidate_audit.md`; the 7-negative set is the pre-candidate-audit benchmark.
- The **full-game event-detection** result uses 162 positives and 28 negatives;
  see `full_results.md`. The 19 + 7 scorecard is the audited subset, not the full
  benchmark.

## Files

| File | What it is |
|---|---|
| `scorecard.md` | Audited subset: event-level P/R/F1 at ±1/2/3 s, per-event predictions, audit exclusions |
| `full_scorecard.md` | Full-game 162-positive / 28-negative event-level score |
| `pilot_results.md` | 46-clip event-detection pilot: dataset, negative taxonomy, findings |
| `threshold_sweep.md` | `MIN_RISE_PX` 60/80/100/120 on the cached frames |
| `two_clock.md` | Release anchor vs PBP result anchor, and the separate release/made-shot tasks |
| `release_timing.md` | Symmetric vs shifted release-window sensitivity on cached predictions |
| `non_shot_filter.md` | Balanced-negative non-shot filter result |
| `non_shot_filter_holdout.md` | Cross-quarter filter validation |
| `second_game_validation.md` | Second-broadcast event/filter validation |
| `verifier_v1.md` | Historical 42-candidate null result |
| `verifier_v1_audited.md` | Null result after applying the alignment audit (36 candidates) |

Other markdown files in this directory are supporting runs. Raw label JSON, per-clip
manifests, and `ClipResult` JSON are not distributed; human-readable audit summaries
are. See `docs/DATA_POLICY.md`.

## Run

```powershell
# Full benchmark; uses the audit by default and cached predictions
python scripts/eval_pbp_shots.py --tolerances 1,2,3

# Force a pipeline rerun
python scripts/eval_pbp_shots.py --refresh --tolerances 1,2,3

# Threshold sweep over the cached frames (seconds)
python scripts/sweep_pbp_shot_thresholds.py --rises 60,80,100,120
```

## Audit

Five clips were reviewed:

| Clip | Verdict | Action |
|---|---|---|
| `shot_0041` | misaligned | excluded from positives |
| `shot_0089` | aligned | kept |
| `neg_003` | adjacent real shot | excluded from negatives |
| `neg_006` | misaligned | excluded from negatives |
| `neg_014` | adjacent real shot | excluded from negatives |

After the audit: **19 positives, 7 clean negatives.** At the default detector, the
clean negatives have **0/7 false-positive clips**; the previous 3 negative-clip
detections were all on misaligned or adjacent-shot clips.

## Candidate audit

All 36 rise80 candidates were reviewed at the candidate level:

| Priority | Real shot | Turnover | Not a shot | Unsure |
|---|---:|---:|---:|---:|
| Critical | 1 | 1 | 0 | 3 |
| Ambiguous | 14 | 0 | 0 | 4 |
| Positive | 5 | 0 | 4 | 4 |
| **Total** | **20** | **1** | **4** | **11** |

Candidate precision among the 25 non-unsure labels is **0.80** (95% CI 0.61–0.91),
but duplicate candidates are counted separately.

The important result is the timing: real-shot candidates have a median offset of
**−3.22 s** from the PBP event (range −5.53 to −0.37 s), while non-shot candidates
cluster at **+1.15 s**. The PBP event is not the release; the detector’s true
candidates are usually 2–5 s earlier. That makes the symmetric ±2/3 s event
scorecard misleading.

Re-scoring the same cached predictions with an asymmetric **-5.5/+1 s** window
gives P 0.71 / R 0.53 / F1 0.61, versus P 0.50 / R 0.37 / F1 0.42 at symmetric
±3 s. This is an in-sample sensitivity result; validate it on a second game before
treating it as a stable correction. See `release_timing.md`.

The orange candidate and the blue PBP event are two different anchors: release and
made/missed result. Release tracking and made-shot detection must be reported
separately; see `two_clock.md`.

The candidate audit also changes the negative set:

- `neg_001` contains a real shot → not a clean negative.
- `neg_007` contains a turnover candidate → confirmed false positive; one other
  candidate is unsure.
- `neg_002` and `neg_013` have only unsure candidates → unverified.

So the earlier “4/7 clean negatives fire” is not a valid clean-negative false-positive
rate. At most one of the four is a confirmed false positive.

## Event-detection pilot (46 clips)

The pilot adds 10 new PBP positives and 13 aligned non-shot negatives to the
audited set: **29 positives, 17 negatives**. The new positives aligned 10/10; the
new negatives aligned 13/16 (three unaligned clips were excluded).

Clip-level event detection:

| `MIN_RISE_PX` | Positive clips with any shot | Negative clips with any shot |
|---:|---:|---:|
| 60 | 27/29 | 8/17 |
| **80** | **28/29** | **6/17** |
| 100 | 20/29 | 4/17 |
| 120 (default) | 13/29 | 1/17 |

At rise80, **28/29** positive clips have a candidate, but **6/17** negatives fire.
The negative taxonomy is the useful result: **4 of the 6** are free throws, 1 is a
turnover, and 1 is a foul. A free-throw/scorebug state filter is the next
event-detection improvement. At the default threshold, the only negative hit is a
free throw. An oracle free-throw exclusion would reduce rise80 negative clips from
**6/17 to 2/11** with no positive loss.

See `pilot_results.md`, `pilot_threshold_sweep.md`, and `pilot_release_timing.md`.

## Full 162-clip event detection

The full-game run uses **162 aligned positives** and **28 negatives**; 24
positives were excluded because the scorebug clock did not align, mostly
quarter-end plays. A full write-up is in `full_results.md`.

| `MIN_RISE_PX` | Positive clips with any shot | Negative clips with any shot |
|---:|---:|---:|
| 60 | 144/162 (89%) | 15/28 (54%) |
| **80** | **135/162 (83%)** | **12/28 (43%)** |
| 100 | 109/162 (67%) | 9/28 (32%) |
| 120 (default) | 81/162 (50%) | 4/28 (14%) |

At rise80, event-level performance is P 0.47 / R 0.65 / F1 0.54 at ±3 s. At the
default, it is P 0.55 / R 0.38 / F1 0.45. The pilot’s 28/29 recall did not fully
generalize; the full set includes harder blocks, closeouts, and quarter-end plays.

The negative taxonomy at rise80 is 5/7 free throws, 4/12 turnovers, 3/7 fouls,
and 0/2 dead-ball clips. Free throws are the top rate but not the whole problem;
a general non-shot filter is still needed.

An exploratory 226-candidate filter scored AUC 0.601 with clip-level labels and
0.686 with release-window labels, but its event F1 gain was small. The release-label
result may be learning timing. It is not adopted; see `full_results.md` and
`verifier_full_*.md`.

The balanced-negative filter is stronger: 60 six-second no-FG windows produce 246
candidates, LOO AUC **0.728**, cross-quarter holdout AUC **0.717**, and event F1
**0.67 vs raw 0.52** at ±3 s on held-out quarters. It is still one game, so it is
not adopted; see `non_shot_filter.md` and `non_shot_filter_holdout.md`.

## Second-game validation — failed transfer

Game 2 (ESPN `401766463`, Mavericks–Grizzlies) has 36 aligned positives and 30
balanced negatives. Event detection at rise80 finds a shot in only **14/36**
positive clips (39%) and fires on **9/30** negatives. Event F1 at ±3 s is 0.33.

The game-1 filter applied at the frozen 0.3 threshold scores AUC 0.659 and event
F1 0.35 vs raw 0.33 — no meaningful transfer. **Do not adopt the filter or claim
cross-broadcast generalization.** The detector itself is the larger problem. See
`second_game_validation.md`.

## Audited 19-clip subset result

At `MIN_RISE_PX=120` (default), audited set:

| Tolerance | Precision | Recall | F1 |
|---|---:|---:|---:|
| ±1 s | 0.21 | 0.16 | 0.18 |
| ±2 s | 0.36 | 0.26 | 0.30 |
| ±3 s | 0.50 | 0.37 | 0.42 |

At `MIN_RISE_PX=80`, audited set:

| Tolerance | Precision | Recall | F1 |
|---|---:|---:|---:|
| ±1 s | 0.22 | 0.42 | 0.29 |
| ±2 s | 0.36 | 0.68 | 0.47 |
| ±3 s | 0.36 | 0.68 | 0.47 |

`MIN_RISE_PX=80` finds a shot in **19/19** positive clips. The old **4/7** negative
count is not a valid clean-negative rate after the candidate audit (one contaminated,
one confirmed turnover, two unverified).

These event-level numbers use PBP-proximity labels. Because the PBP event is later
than the real shot, treat them as a **lower-bound / misaligned** view, not the final
release metric.

## Verifier v1 — null result

A leave-one-clip-out logistic regression on the 36 audited rise80 candidates (13
positive, 23 negative) with the existing 12 features plus court-view, scene-cut, and
predicted shot-type features scored **AUC 0.528** — chance. The historical
42-candidate set scored 0.512. The event-level threshold table is noise, not a
tuning surface. Do not adopt it.

The next verifier needs more than 36 candidates and better release/pose/trajectory
features. Expanding to all 186 PBP events and a second game is the prerequisite.

### Candidate-audit verifier

On the 25 human-labeled candidates (20 shot, 5 non-shot; 11 unsure excluded), the
same verifier scored **AUC 0.270** — below chance. With only 5 negatives, that is
not a stable estimate and not a usable model. The right conclusion is that the
candidate label set is too small and imbalanced, not that another threshold will
fix it. See `verifier_candidate_audit.md`.

## Caveats

- The PBP frame is the event clock, not the release frame. The candidate audit shows
  real-shot candidates a median 3.2 s earlier than the PBP event.
- A 10 s clip can contain another real shot from the same or adjacent possession.
  The candidate audit contains multiple real-shot candidates per clip; some are
  duplicate phases of the same shot.
- Eleven of 36 candidates are unsure. They are excluded from the candidate-level
  precision and verifier labels.
- One game is one game. Validate the timing offset, the negative set, and the
  verifier on a second broadcast before claiming a stable number.
