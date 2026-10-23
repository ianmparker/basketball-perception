---
id: perception.events.detection
title: Event detection and temporal localization
layer: perception
track: perception.events
status: active
confidence: medium
updated: 2026-10-21
sources: [arxiv-2607.21267, arxiv-2608.19646, arxiv-2105.07404, arxiv-2505.03991, arxiv-2404.05392, arxiv-2503.00147]
code: [events/, eval/metrics/events.py, eval/score.py]
prereqs: [perception.ball.tracking, perception.tracking.mot]
tags: [events, shots, screens, rebounds, temporal, labels]
---

# Event detection and temporal localization

**Claim.** An event is a **defined change in game state**, not a visual pattern. If the
definition is ambiguous, no model will fix it: two annotators will disagree, the metric
will be noise, and the coach will not trust the output. Define first, detect second.

## Two paradigms

1. **Rule-based on tracking.** Derive events from ball/player trajectories and court
   geometry: a shot is a release followed by a ballistic arc toward the rim; a screen is a
   stationary offensive player whose defender is impeded; a rebound is the first player to
   control the ball after a missed shot. Interpretable, cheap, brittle when tracking is
   sparse.
2. **Learned spotting.** A temporal model (TCN, transformer, or 3D CNN) consumes a clip
   and emits event classes with start/end times. Handles visual cues that rules miss;
   needs dense labels and inherits annotation noise.

The practical answer is hybrid: rules for geometry-verifiable events (shot location,
possession change), learned models for appearance/pose-dependent events (screen, foul,
contest), and a shared ontology so the two can be compared.

## Event definitions that survive contact with video

| Event | Operational definition | Needed signal | Hard part |
|---|---|---|---|
| Shot attempt | Ball leaves a player's hand with an upward/forward velocity and a trajectory consistent with the rim | ball track, pose, court | release vs pass vs dribble |
| Made shot | Ball passes downward through the rim plane inside the cylinder | ball track, rim location | rim occlusion, camera angle |
| Rebound | First player to gain control after a shot that did not score | ball track, players, shot outcome | tip/deflection vs control |
| Pass | Ball travels from one teammate to another with possession retained | ball + track IDs | deflections, bounce passes |
| Screen | Offensive player stationary while a defender is impeded within a threshold distance | poses, positions, velocities | moving screens, slip screens |
| Pick-and-roll | Screen followed by the screener rolling and the handler attacking | screen + trajectories | delay between events |
| Steal | Defensive player causes a live-ball turnover | ball + possession change | out-of-bounds vs steal |
| Foul | Illegal contact or illegal action per rulebook | pose, contact, context | judgment; officiating domain |
| Violation | Traveling, double dribble, 3 seconds, etc. | ball, feet, paint, clock | rule/league differences |

`ShotEventModel`, `ScreenEventModel`, `PnREventModel`, `DunkEventModel`,
`PassEventModel`, and `InterceptionEventModel` in `inference/schema.py` cover the shot,
pass, screen, and PnR rows only partially. **Shot outcome (make/miss), rebound, foul,
turnover, free throw, substitution, and violation have no field** — and they are exactly
what a coach and a referee care about.

## Temporal localization is its own problem

Event *classification* ("this clip contains a screen") and event *spotting* ("the screen
starts at frame 412 and ends at 447") are different tasks. Coaching needs the timestamp;
officiating needs the frame. Evaluate with:

- **temporal tolerance**: a detection is correct if its start is within `δ` frames of
  ground truth (the repo already sweeps 5/15/30);
- **event-level P/R/F1**, not frame-level accuracy;
- **mAP@δ** for spotting, if we adopt a spotting model;
- **duplicate rate**: the same event fired twice is a precision failure, not a recall win.

`eval/score.py` already computes event P/R/F1 with tolerance and a noise floor. Keep that
as the primary gate. Add a per-class table so a good shot F1 cannot hide a broken screen
detector.

## The ontology problem

BasketEvent defines ten player-grounded categories (made/missed shot, free throw, foul,
turnover, rebound, assist, block, steal, jump ball), plus a background class, from NBA
play-by-play.
PL-NBA adds possession-level clips and captions. Both are closer to a coach's vocabulary
than our current six classes. Adopt a shared event ontology before adding detectors, and
map every detector output to it.

Key distinctions the ontology must make:

- **attempt vs make** (shot), **assisted vs unassisted**;
- **offensive vs defensive rebound**;
- **live-ball vs dead-ball** turnover;
- **personal vs team vs technical foul**;
- **on-ball vs off-ball screen**;
- **possession-preserving vs possession-ending** action.

## Label noise

Event labels are noisy even from play-by-play: timestamps are coarse, the responsible
player is sometimes ambiguous, and broadcasts omit off-ball events. Consequences:

- report inter-annotator agreement (Cohen's κ) per class before trusting a model;
- keep a adjudicated gold set separate from the training labels;
- never tune a threshold on the test clips (`perception-researcher` rule);
- if a class has κ < 0.6, the definition is the bug, not the model.

## Repo mapping

- `events/detect_shots.py`, `shot_detector_v2.py`, `detect_shots_pose.py` — shot rules.
- `events/detect_screens.py`, `screen_detector_v2.py` — screen rules.
- `events/pnr_detector.py` — pick-and-roll.
- `events/dunk_detector.py` — dunk.
- `analytics/passes.py` — pass/interception.
- `eval/metrics/events.py` — tolerance matching and P/R/F1.
- `eval/labels/` — the labeled clips; `schema.md` defines the label format.

## Failure modes

- **Event definitions drift between code, labels, and docs.** Put the definition in the
  ontology note and cite it from the detector.
- **Duplicate firing.** The screen detector has a de-dup test; generalize the pattern.
- **Tolerance too wide.** A 30-frame tolerance at 30 fps is 1 s — enough to call a pass a
  shot. Report the strict number as primary.
- **Class imbalance.** A model that never predicts "foul" can still look accurate. Use
  per-class P/R/F1 and macro averages.
- **Ignoring the clock and score.** A shot in garbage time is not the same event as a
  game-winner; context belongs in the event record.

## Open questions

1. Which event classes can be defined purely geometrically, and which require judgment?
2. What is the inter-annotator κ for screen and foul on our own clips?
3. Should events be represented as intervals (start, end) rather than instants? A screen
   has duration; a shot has a release instant.
4. How do we evaluate off-ball events when the broadcast camera does not show them?

## Proposed experiments

- **EX-030** Write the event ontology v1 and map every existing detector output to it;
  measure coverage and gaps.
- **EX-031** Add rebound detection as a rule on top of shot outcome + ball tracking;
  label 50 rebounds and score it.
- **EX-032** Measure inter-annotator agreement on 30 screens and 30 fouls before training
  anything.
- **EX-033** Compare rule-based vs a T-DEED-style spotter on the same shot clips.
