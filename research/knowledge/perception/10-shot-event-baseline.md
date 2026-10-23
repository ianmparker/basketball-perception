---
id: perception.events.shot-event-baseline
title: NBA shot-event baseline and scorebug alignment
layer: perception
track: perception.events
status: active
confidence: medium
updated: 2026-10-22
sources: [nba-stats-api, oliver2004, arxiv-2607.21267]
code: [analytics/shot_events.py, analytics/scorebug.py, scripts/nba_shot_event_manifest.py, scripts/align_shots_by_scorebug.py, scripts/eval_shot_attempts.py, scripts/broadcast_demo.py, scripts/build_demo_page.py, scripts/label_broadcast_events.py, scripts/score_broadcast_events.py]
prereqs: [perception.state.possession-ledger]
tags: [shot detection, events, scorebug, OCR, alignment, NBA]
---

# NBA shot-event baseline and scorebug alignment

**Claim.** NBA play-by-play gives clean shot/rebound event labels, but those labels are
not frame-accurate by themselves. Scorebug OCR is the missing bridge: on the 20-shot
validation set it aligned 18/20 events, correcting PBP-to-frame offsets of up to 13 s. The
existing heuristic shot detector then found a shot in 10/20 clips (50%). That is a
baseline, not a product result; layups and mid-range shots are the weak classes.

## What was built

- `analytics/shot_events.py` turns PBP field-goal attempts into event records with
  shooter, make/miss, shot value, shot type, distance, and next rebound.
- `scripts/nba_shot_event_manifest.py` builds all field-goal events and a made/missed-
  balanced validation set with clips.
- `analytics/scorebug.py` parses the broadcast scorebug clock and period.
- `scripts/align_shots_by_scorebug.py` searches around each PBP frame for the scorebug
  clock, then cuts a corrected clip.
- `scripts/eval_shot_attempts.py` runs the existing video pipeline and scores clip-level
  shot detection.

## Data and result

Game `401810030` (Knicks–Wolves), 2025-11-06.

- PBP field-goal attempts: **186**.
- Validation set: **20** clips, 10 made and 10 missed.
- Scorebug clock alignment: **18/20** (90%). Two events had no scorebug match:
  `shot_0041` (Q1 0.0 s) and `shot_0089` (Q3 673 s). The corrected offsets reached
  **390 frames (13 s)** in the five-clip pilot, which is large enough to explain the
  closeup/replay clips from the raw PBP alignment.
- Shot attempt detector: **10/20** clips had at least one detected shot (50%).
  - Made clips: 6/10.
  - Missed clips: 4/10.
  - 3PT: 4/6.
  - Layup: 0/2.
  - Dunk: 1/2.
  - Mid-range: 0/1.
  - Other: 5/9.
- Detection within +/-2 s of the PBP frame: 5/20. This is secondary; the PBP clock is
  not the exact release frame.
- Clean negative set: 10 scorebug-aligned turnover possessions with no preceding shot.
  The detector fired on **1/10** at the default threshold, **3/10** at `MIN_RISE_PX=80`,
  and **6/10** at `MIN_RISE_PX=60`. These counts use the pre-audit turnover cut; the
  audited benchmark below supersedes them.

## What this means

1. **The labels are usable; the alignment was not.** PBP is an excellent teacher for
   *what* happened. It is not a frame label. Scorebug OCR turns it into one.
2. **The current detector is jump-shot biased.** The trigger requires a sustained upward
   ball streak. Layups and at-rim actions do not satisfy it; mid-range is also weak in
   this sample.
3. **Do not overread 50%.** The sample is 20 clips from one game, the detector is a
   heuristic, and the scorebug ROI is tuned to this MSG/bottom-left broadcast. The
   direction is useful; the number is not a product claim.
4. **The next event component is at-rim/layup detection.** Without it, the shot chart
   will systematically miss the highest-value shots.

### Threshold vs clean negatives (pre-audit, historical)

> **Superseded (2026-10-22).** This first pass used a different 10-clip turnover cut.
> The audited benchmark below (19 positives + 7 clean negatives) is the current number:
> 0/7 clean negatives at 120, 4/7 at 80.

| Config | Positive recall | Clean-negative FP | Approx precision* | Approx F1* |
|---|---:|---:|---:|---:|
| `MIN_RISE_PX=120` (default) | 10/20 | 1/10 | 0.91 | 0.64 |
| `MIN_RISE_PX=80` | 20/20 | 3/10 | 0.87 | 0.93 |
| `MIN_RISE_PX=60` | 19/20 | 6/10 | 0.76 | 0.85 |

*Assuming 20 positives and 10 clean negatives; not a production estimate.

`MIN_RISE_PX=80` is the better **candidate generator** on this set: it keeps all positive
clips while preserving most of the clean negatives. It is not a shipping decision. A quick
pose-only probe (nearest player, wrist-above-elbow) kept 15/20 positives but only 1/10
negatives; it did not beat the raw rise80 F1. A single hand rule is not enough; the next
filter needs learned or multi-feature geometry. A second NBA broadcast is required before
changing the default.

### Candidate-filter probe (null result)

A leave-one-clip-out logistic regression was fit on 35 candidate rows (20 positive, 15
negative) from the rise80 detector. Candidate AUC was **0.31** (below chance), and the
best clip-level operating point was 19/20 recall with 3/10 clean-negative clips at
threshold 0.2 — essentially the same as raw rise80, with no usable ranking. The filter is
not adopted. The current feature set/labels are too weak; the next attempt needs more
clips, better candidate labels, and features that separate a shot from a drive/pass.

### Broadcast demo run (2026-10-22)

The current pipeline was run end-to-end on a 1200-frame (20 s) 1080p60 NBA broadcast
clip (`kyrie.mp4`) with the broadcast preset, `--offline-events`, and the current
defaults (replay-v3 ball detector, K-means team ID). A legacy run from 2026-10-20 on
the same clip is kept as a comparison.

| Run | Raw ball detected | Shots | Screens | PnRs | Passes | Interceptions |
|---|---:|---:|---:|---:|---:|---:|
| Kyrie current defaults | **73.4%** | 2 | 4 | 0 | 5 | 5 |
| Kyrie legacy run (2026-10-20) | 52.6% | 3 | 4 | 0 | 10 | 6 |
| Lima HS broadcast, 450 frames | 65.3% | 2 (1 layup) | 1 | 1 | 2 | 2 |

These are **counts, not precision/recall**. There are no event labels for these clips,
and the runs disagree on the event set, so the table does not rank the detectors.

The Lima run is the first broadcast demo with a detected layup (4.4 ft) and a PnR
event. It also rejected 35% of frames at the court-view gate because the segment
contains crowd/close-up shots, and team assignment collapsed to a 9.9% / 90.1% split.
Both are demo-selection and team-ID failures, not evidence about the event detector.

Two failure modes are visible and matter for the demo:

1. **Shot location is the ball at the gather, not the release.** One current-run shot
   projects to `[94, 50]`, the court corner, which means the ball was off-camera or the
   projection failed. The demo page drops boundary projections and labels them as
   “location unavailable.”
2. **Shooter attribution can pick the wrong track.** Frame inspection of the legacy
   run shows a rim attack attributed to a player at the wing. The shot chart is a
   candidate view, not a coach-trusted chart, until the shooter/rim geometry is fixed.

The demo artifacts are `outputs/broadcast_demo/` (local), with the manifest and
scorecard in `eval/broadcast_demo/`. The next useful step is to label the 2–3 shots
and report event-level precision/recall at a stated tolerance, then fix the release
location and shooter attribution before adding more event types.

### Labeled demo result (2026-10-22) — small demo check

> This seven-candidate set is a demo sanity check. The full-game PBP benchmark below is the event benchmark.

The owner labeled all 7 candidates with the criterion "a real shot attempt with the
release visible in the clip." **Lima is a fixed/elevated top-down HS camera, not an
NBA broadcast.** Read the two regimes separately.

| Regime | Runs | Real | False positive | Candidate precision (Wilson 95% CI) |
|---|---|---:|---:|---|
| **NBA broadcast** | Kyrie current + legacy | **4** | **1** | **0.80** (0.38–0.96) |
| **HS top-down probe** | Lima | 0 | 2 | 0.00 (0.00–0.66) |
| Mixed (not the headline) | All | 4 | 3 | 0.57 (0.25–0.84) |

Recall is not measured: `labels.json` contains only predicted candidates, so a
missed real shot is not represented. Add missed shots with `predicted_frame: null`
to measure it.

The false positives split by regime:

1. **NBA broadcast — free throw (Kyrie legacy, frame 439).** The event is 15.3 ft
   from the hoop and the scorebug reads `FREE THROWS 2/3`. The detector treats the
   free-throw motion as a field-goal attempt. This is a clean, fixable broadcast
   class.
2. **HS top-down — pass/drive (Lima, frame 288).** The predicted court location is
   `[0, 10]` (invalid), and the contact sheet shows ball movement with no release.
   The upward ball streak is not a shot.
3. **HS top-down — rim-area pass/rebound plus a camera cut (Lima, frame 356).** The
   ball is near the rim but there is no shot; the clip contains a scene cut.

The useful conclusion is not "the detector is 57% accurate." It is: **the NBA
broadcast path is a usable candidate generator (4/5 real), the HS top-down view does
not transfer (0/2), and the false-positive classes are identifiable.** The
broadcast next step is a free-throw guard and more labels. The HS top-down next step
belongs in Phase B: its own benchmark, calibration, and verifier. **Do not tune NBA
thresholds on the Lima clip.**

A quick check of the existing 12 candidate features shows they **do not separate the
classes**:

- The free throw has `pose_wrist_up=1`, `rise_px=242`, and `ball_near_hoop_px=718`;
  a real broadcast shot (legacy frame 1081) also has `pose_wrist_up=1` and
  `rise_px=163`.
- The HS rim false positive has `nearest_player_px=0.7`; a real layup can also put
  the ball on the player at the gather.
- The HS pass/drive has an invalid court location, which is a data-quality flag,
  not a shot/non-shot feature.

Fitting a classifier on 7 rows would overfit. The next step is **more broadcast
labels plus deterministic guards**: scorebug free-throw exclusion and a court-view
continuity check, then a release verifier once release frames exist.

### Full-game PBP benchmark (2026-10-22)

The event benchmark is the full-game PBP set from `401810030`
(Knicks–Timberwolves), not the seven hand-labeled Kyrie candidates.

- **Positives:** 19 PBP field-goal attempts after the alignment audit.
- **Negatives:** 7 clean turnover possessions after the alignment audit.
- **Clips:** 10 s, 30 fps. Event frame is the PBP event clock, not the release.
- **Audit:** five clips were reviewed. `shot_0041` was misaligned and removed;
  `shot_0089` was aligned. `neg_003` and `neg_014` contained an adjacent real shot;
  `neg_006` was misaligned. All three were removed from the clean negatives.

At the default `MIN_RISE_PX=120`, event-level performance on the audited set:

| Tolerance | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| ±1 s | 3 | 11 | 16 | 0.21 | 0.16 | 0.18 |
| ±2 s | 5 | 9 | 14 | 0.36 | 0.26 | 0.30 |
| ±3 s | 7 | 7 | 12 | 0.50 | 0.37 | 0.42 |

Clip-level: 10/19 positive clips contain any shot, and **0/7 clean negatives**
contain a shot. The previous three negative-clip detections were all on misaligned
or adjacent-shot clips, not clean negatives.

A replay sweep over the cached frames shows the recall/precision trade-off:

| MIN_RISE_PX | ±2 s P/R/F1 | ±3 s P/R/F1 | Positive clips with any shot | Negative clips with any shot |
|---:|---|---|---:|---:|
| 60 | 0.26/0.58/0.36 | 0.31/0.68/0.43 | 18/19 | 5/7 |
| **80** | **0.36/0.68/0.47** | **0.36/0.68/0.47** | **19/19** | **4/7** |
| 100 | 0.26/0.26/0.26 | 0.37/0.37/0.37 | 14/19 | 0/7 |
| 120 (default) | 0.36/0.26/0.30 | 0.50/0.37/0.42 | 10/19 | 0/7 |

**Finding (pre-candidate-audit):** `MIN_RISE_PX=120` is too strict. At `80`, every
positive clip has a candidate, so the candidate generator is adequate. The original
4/7 clean-negative rate is not a valid release metric after the candidate audit
below: one negative contains a real shot and two are unverified.

### Candidate-level audit (2026-10-22)

All 36 rise80 candidates were reviewed on `candidate_audit.html`:

| Priority | Real shot | Turnover | Not a shot | Unsure |
|---|---:|---:|---:|---:|
| Critical (clean-negative candidates) | 1 | 1 | 0 | 3 |
| Ambiguous (extra candidates in positive clips) | 14 | 0 | 0 | 4 |
| Positive (within ±2 s of selected PBP event) | 5 | 0 | 4 | 4 |
| **Total** | **20** | **1** | **4** | **11** |

Among the 25 non-unsure candidates, candidate precision is **20/25 = 0.80**
(95% CI 0.61–0.91). Duplicate candidates are still counted separately.

**The timing finding is the important one.** For the 20 candidates labeled real
shot, the candidate frame relative to the PBP event was median **−3.22 s** (mean
−3.00 s; range −5.53 to −0.37 s). The 4 not-shot candidates clustered at median
**+1.15 s** (range −1.30 to +1.67 s). The detector’s real candidates are usually
2–5 s **before** the PBP event; the candidates near the PBP event are often the
shot result, rebound, or another non-shot action. The current proximity labels are
close to inverted for this detector.

This explains why the verifier trained on proximity labels could fall below
chance. It also means the symmetric ±2/3 s event scorecard is not a fair
shot-release metric. A provisional asymmetric window of about −5.5 to +1 s around
the PBP event fits these 20 shots, but it needs validation on a second game.

Re-scoring the same cached predictions with that asymmetric window gives
P 0.71 / R 0.53 / F1 0.61, versus P 0.50 / R 0.37 / F1 0.42 at symmetric ±3 s.
This is an in-sample sensitivity result; it is not a new detector run and it must
be validated on a second game.

The working interpretation, verified on the clips, is **two clocks**: the orange
candidate is the release/attempt anchor; the blue selected PBP event is the
made/missed result anchor. The measured offset is therefore **release-to-result**
(plus a small trigger-to-release lead), not detector error. Release tracking must
use candidate labels; made-shot detection uses the PBP outcome label and an
associated release candidate. The made/missed split above is about the size of the
result lag (made n=8, median −2.09 s; missed n=11, median −3.47 s), not whether
blue is release. Release-frame labels are still needed before encoding a
made/missed-specific shift. See `eval/pbp_shots/two_clock.md`.

The negative set also changed:

- `neg_001` contains a real shot at −4.53 s; remove it from clean negatives.
- `neg_007` contains a confirmed turnover candidate at −4.13 s — a true false
  positive on a clean negative; its second candidate at +1.57 s is unsure.
- `neg_002` and `neg_013` have only unsure candidates; their negative status is
  unverified.
- The old “4/7 clean negatives fire” is therefore not a clean-negative FP rate.

### Event-detection pilot (2026-10-22)

A 46-clip pilot tested the event-detection manifest before the 186-event run:
29 PBP positives (19 audited + 10 new) and 17 negatives (4 strict existing +
13 new non-shot clips). The new positives aligned 10/10; the new negatives
aligned 13/16 (three unaligned clips were excluded).

At `MIN_RISE_PX=80`, the detector found a candidate in **28/29 positive clips**
and **6/17 negative clips**. The negative taxonomy is the important result:

- **4/6 free-throw negatives** fired.
- 1/8 turnover negatives fired.
- 1/3 foul negatives fired.

At the default `MIN_RISE_PX=120`, only **1/17** negatives fired (a free throw).
The negative construction works; free throws are the dominant candidate-level
false positive. A free-throw/scorebug state filter is the next event-detection
improvement. The 186-event run should follow that filter, not precede it.

See `eval/pbp_shots/pilot_results.md` and `pilot_threshold_sweep.md`.

### Full 162-clip event detection (2026-10-22)

The full-game event set has **162 aligned PBP positives** and **28 negatives**
(7 free throws, 12 turnovers, 7 fouls, 2 dead-ball clips). Twenty-four positives
were excluded because the scorebug clock did not align, mostly quarter-end or
period-boundary plays. The negative set is not balanced; 10 s windows with no PBP
field goal are scarce.

Clip-level event detection:

| `MIN_RISE_PX` | Positive clips with any shot | Negative clips with any shot |
|---:|---:|---:|
| 60 | 144/162 (89%) | 15/28 (54%) |
| **80** | **135/162 (83%)** | **12/28 (43%)** |
| 100 | 109/162 (67%) | 9/28 (32%) |
| 120 (default) | 81/162 (50%) | 4/28 (14%) |

At rise80, event-level performance is P 0.47 / R 0.65 / F1 0.54 at ±3 s. At the
default it is P 0.55 / R 0.38 / F1 0.45. The pilot’s 28/29 recall did not fully
generalize; the full set contains harder blocks, closeouts, and end-of-quarter
actions.

At rise80, the 12 negative hits are 5/7 free throws, 4/12 turnovers, 3/7 fouls,
and 0/2 dead-ball clips. Free throws have the highest rate, but turnovers and
fouls also fire; a general non-shot filter is needed, not only a free-throw rule.
Positive coverage by shot type is 66/75 threes, 16/19 mid-range, 9/11 dunks,
20/25 layups, and 24/32 other; the detector is not strongly type-biased.

The full release-window sensitivity is P 0.62 / R 0.43 / F1 0.50 for −5.5/+1 s,
versus P 0.55 / R 0.38 / F1 0.45 at symmetric ±3 s. The two-clock effect is real,
but smaller than the pilot suggested.

A 226-candidate logistic-regression filter on the existing features scored
AUC 0.566 with PBP-proximity labels, 0.601 with clip-level shot/no-shot labels,
and 0.686 with release-window labels. The release label likely leaks timing
(ball height/velocity correlate with phase), and the event F1 gain was small.

Adding 60 balanced six-second no-FG negatives changes the result: **246
candidates, LOO AUC 0.728, cross-quarter holdout AUC 0.717**, and event F1
**0.67 vs raw 0.52** at ±3 s on held-out quarters. The filter is still trained
and tested on one game, so it is **not adopted**; a second broadcast is required.
See `eval/pbp_shots/non_shot_filter.md`.

A second broadcast (ESPN `401766463`, Mavericks–Grizzlies) failed the transfer
test. On 36 aligned positives and 30 balanced negatives, rise80 found a candidate
in only **14/36** positive clips (39%) and fired on **9/30** negatives. The game-1
filter at threshold 0.3 scored **AUC 0.659** and event F1 **0.35 vs raw 0.33** at
±3 s. The detector itself, not just the filter, has a second-broadcast domain
shift. Do not claim cross-broadcast generalization; see
`eval/pbp_shots/second_game_validation.md`.

#### Verifier v1 — null result (2026-10-22)

A leave-one-clip-out logistic regression on the 36 proximity-labeled rise80
candidates (13 positive, 23 negative) used the existing 12 features plus court-view
fraction, scene-cut count, and predicted shot-type flags. **OOF AUC = 0.528**,
which is chance. The historical 42-candidate set scored 0.512. The event-level
threshold table is noise; do not adopt or tune it.

On the 25 human-labeled candidates (20 real shots, 5 non-shots, 11 unsure
excluded), the same verifier scored **AUC 0.270** — below chance. With only five
negatives the estimate is unstable and not a usable model. The next step is a
balanced candidate label set, not another fit on these 25 rows.

The prerequisite is more data and better features, not another model on 26 clips:
expand to all 186 PBP events and a second game, add release/pose/trajectory
evidence, and match candidates against all PBP field goals in the clip window.

## Repo mapping

- `analytics/shot_events.py` — PBP shot and rebound records.
- `analytics/scorebug.py` — scorebug text parser.
- `scripts/nba_shot_event_manifest.py` — event manifest and validation clips.
- `scripts/align_shots_by_scorebug.py` — OCR clock search and corrected clips.
- `scripts/eval_shot_attempts.py` — clip-level attempt evaluation.
- `scripts/nba_clean_negatives.py` — scorebug-aligned turnover negatives and FP report.
- `analytics/shot_features.py` — candidate feature extraction.
- `scripts/build_shot_candidate_dataset.py` — candidate-level dataset.
- `scripts/eval_shot_candidate_filter.py` — leave-one-clip-out candidate filter.
- `scripts/broadcast_demo.py` — one-command broadcast run + demo page.
- `scripts/build_demo_page.py` — static page, summary, shot chart, report.
- `scripts/label_broadcast_events.py` — local shot-labeling page.
- `scripts/score_broadcast_events.py` — event-level P/R/F1 from labels.
- `eval/broadcast_demo/manifest.json`, `labels.json`, `scorecard.md`, `event_scorecard.md`
  — broadcast demo benchmark, labels, and results.
- `inference/run_video.py` — the shot attempt detector being evaluated.

## Failure modes

- **Treating the PBP frame as the shot release.** It is the event clock, not the release.
- **Skipping the alignment audit.** A closeup/replay clip will make a detector look worse
  than it is, or better if it finds a different shot.
- **Using one scorebug ROI for every broadcast.** The crop is MSG/bottom-left specific.
- **Counting any shot in a 10 s clip as a correct event.** A clip can contain a putback
  or the next possession; use the event frame where possible.
- **Calling a 50% clip-level result production-ready.** It is a baseline on one game.

## Open questions

1. How much does scorebug alignment improve the full 170-event set, not just 20?
2. Can a rim-proximity + ball-trajectory detector recover layups and dunks?
3. What is the right negative class for shot attempts: random half-court windows,
   passes, or rebounds?
4. Can the scorebug score field produce made/miss labels and validate the PBP?
5. How does the detector transfer to a second NBA broadcast with a different scorebug?

## Proposed experiments

- **EX-260** Scorebug clock alignment — **done**. 18/20 events matched; offsets up to 13 s
  corrected.
- **EX-261** Shot attempt detector baseline — **done**. 10/20 clips detected; layups and
  mid-range are the gap.
- **EX-262** Layup/rim attempt detector: ball-near-rim + trajectory, evaluated on 20
  layup/dunk clips.
- **EX-263** Rebound detector: ball off rim + player proximity + possession change,
  evaluated on 20 missed-shot clips.
- **EX-264** Made/miss classifier: rim/net/ball trajectory features, evaluated on 50
  scorebug-aligned clips.
- **EX-265** Clean negative set: scorebug-aligned turnover possessions with no shot
  attempt — **done**. 10 clips; baseline 1/10 FP, rise80 3/10, rise60 6/10.
- **EX-266** Pose/geometry candidate filter: wrist-above-elbow and ball-player-rim
  relation to reject drives/passes admitted by the lower rise threshold.
