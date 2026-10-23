---
id: basketball.strategy.playbooks
title: Strategy, playbooks, and play recognition
layer: basketball
track: basketball.strategy
status: active
confidence: medium
updated: 2026-10-21
sources: [winter-triple-post-offense, taylor2016, arxiv-2607.21267, arxiv-2608.19646, nba-stats-api, synergy-sportradar]
code: [events/, analytics/, docs/screen-pnr-ontology.md]
prereqs: [basketball.ontology.schema, basketball.tactics.spacing]
tags: [strategy, playbook, triangle, play recognition, counters, matchup]
---

# Strategy, playbooks, and play recognition

**Claim.** A play is a **sequence with a purpose**, not a frame label. A playbook is a
**graph of formations, actions, reads, and counters**, not a list of set plays. Play
recognition is therefore a possession-level sequence problem with ambiguity, context, and
overlapping actions — and it is the bridge from "what happened" to "what should we do."

## The unit hierarchy

```
game
└── possession        control → shot / turnover / period end
    └── play         a sequence with a purpose: PnR, ISO, post-up, transition
        └── action   screen, cut, handoff, roll, pop, drive, closeout
            └── event shot, pass, rebound, foul, turnover
```

The current `ClipResult` is strong on frames and events but has no `possession_id` or
`play_id`. Play recognition cannot be evaluated without them. That is the first schema
gap to close.

## The triangle as a case study

Tex Winter's triple-post offense is the canonical example of a **system**, not a set
play. Its value for this project is representational:

- **Spacing.** A sideline triangle (corner, wing, post) with a weak-side two-man game.
  The geometry creates passing angles and cut windows.
- **Reads.** The same formation produces different actions depending on how the defense
  plays the cutters and the post. The offense is read-and-react, not a memorized route.
- **Counters.** Overplay the wing → backdoor cut. Help from the weak side → skip pass.
  Switch the two-man game → slip or seal the mismatch. The counter is a conditional edge
  in the playbook graph, not a separate play.

Why this matters: a classifier that outputs "triangle" has learned almost nothing. A
useful system recognizes the **action and the read** ("they overplayed the wing, the
backdoor was open") and connects it to a counter. That is the coaching layer.

Modern NBA teams rarely run the pure triangle, but the representation generalizes. The
dominant actions are pick-and-roll, dribble handoff, zoom, horns, Spain PnR, off-ball
screens, isolation, and transition. Build the representation for those; keep the triangle
as the strategy case study.

## A working taxonomy

Use the NBA/Synergy play types as the first level:

| Play type | Core signal |
|---|---|
| Isolation | one handler, four teammates spaced, no screen |
| PnR ball handler | screen + handler attacks; decision is pass/drive/shot |
| PnR roll man | screener dives or pops; catch in the pocket |
| Post-up | player catches on the block with back to basket |
| Spot-up | catch-and-shoot or closeout attack from a stationary position |
| Handoff | dribble handoff between two players |
| Cut | off-ball movement to the rim or open space |
| Off-screen | player runs off a screen away from the ball |
| Putback | offensive rebound follow |
| Transition | possession changes and attacks before the defense is set |
| Miscellaneous | everything else; a real category, not a failure |

Add two orthogonal labels:

- **Coverage:** man, drop, switch, hedge/show, ice/down, trap/blitz, zone (2-3, 3-2,
  1-3-1), full-court press.
- **Play family:** horns, Spain PnR, zoom, floppy, elevator, stagger, etc.

A possession can have multiple labels (a horns set can become a PnR; a broken play is
still a possession). Design for **multi-label + unknown**, not a single hard class.

The machine-readable version is `research/ontology/play_taxonomy.json` — 11 play types,
10 coverages, and 7 families, each with observable signals. Validate with
`python research/ontology/validate.py`.

## Recognition

**Inputs:** possession boundary, tracking IDs, ball track, team assignment, player roles,
court coordinates, clock, score, and the opponent's defensive alignment.

**Outputs:** a distribution over play types, a coverage label, a confidence, and evidence
(frames/clips + the key action). The coach should be able to see *why* the model said
"drop coverage."

**Methods, in order of cost:**

1. **Rules on tracking** — screen + handler + screener's defender depth → drop/hedge/
   switch. Interpretable, brittle when tracking is sparse.
2. **Sequence model on tracking features** — TCN/transformer over 1–3 s windows of
   positions, velocities, and assignments. Handles overlap and timing.
3. **Video model** — 3D CNN/video transformer over the possession clip. Highest ceiling,
   needs the most labels.

The practical answer is hybrid: geometry for possession, screens, and spacing; a sequence
model for play type and coverage; video for the actions geometry misses.

**Labels:** Synergy play types are commercial ground truth (not shippable without a
license); PL-NBA and BasketEvent provide research labels; NBA Stats API gives play-by-play
and shot coordinates. Manual labels on the coach's own footage are the product truth.

**Evaluation:** per-possession top-1/top-3 accuracy, per-class confusion, calibration,
and a multi-label F1. The unit is the **possession**, grouped by game — never the frame.

## Counters: the coaching layer

A counter is a conditional edge:

```
if coverage == drop and screener's defender is below the screen:
    counter = short roll OR pull-up
if coverage == switch and mismatch is small-on-big:
    counter = seal the switch OR slip to the rim
if coverage == ice and handler is forced baseline:
    counter = baseline drive OR kick to the weak-side corner
if coverage == zone:
    counter = high-post flash OR skip to the short corner
if coverage == press:
    counter = press break with a middle receiver
```

These are **examples**, not universal truths. Each needs a precondition (personnel, shot
clock, score, fouls) and a coach's confirmation. The output should be a ranked set of
counters with evidence, not a single command.

Rotation minutes, stamina, and matchup adjustments sit above this layer. They require
lineup data, substitution logs, minutes, and fatigue proxies; they are a later
optimization layer, not a Phase A feature.

## Repo mapping

- `events/` — screen/PnR/action detectors; the candidate generator feeds the play layer.
- `analytics/` — possession, lineup, and value models.
- `docs/screen-pnr-ontology.md` — the event definitions and pair/interval semantics.
- `research/ontology/play_taxonomy.json` — the play/coverage/family vocabulary.
- `inference/schema.py` — add `possession_id`, `play_id`, `game_state` before play
  recognition can be evaluated.
- `research/experiments/backlog.json` — EX-180/181/182 below.

## Failure modes

- **Taxonomy ambiguity.** "PnR" can include the handler's action, the screener's action,
  and the coverage. Label the dimension, not just the word.
- **Overlapping plays.** A possession can be horns → PnR → kick-out. Multi-label or
  hierarchical labels; do not force one class.
- **Broken plays.** The most common NBA possession is not a clean set. Keep an `unknown`
  or `broken` class and report it.
- **Small samples.** A team runs a play 5 times in a game. Report counts and uncertainty;
  do not rank tendencies on 5 possessions.
- **Commercial labels.** Synergy is ground truth for evaluation, not a data source you can
  ship.
- **Triangle romanticism.** The triangle is a great representation case study; it is not
  the modal NBA offense. Build the taxonomy for modern actions.

## Open questions

1. What is the minimum possession segmentation quality for play recognition to beat a
   naive "PnR vs ISO" baseline?
2. Does a sequence model on tracking features beat rules on screens and coverage?
3. How many labeled possessions per play type are needed for a stable per-class F1?
4. Can we learn counters from coach decisions, or must they be authored?
5. What is the first coach-facing output: a play-type frequency sheet, a coverage report,
   or a counter card?

## Proposed experiments

- **EX-180** Freeze play taxonomy v1 (Synergy play types + coverage + play families) as
  JSON; map BasketEvent/PL-NBA labels to it; report unmapped classes.
- **EX-181** Play-recognition baseline on 50 possessions: rules vs a sequence model on
  tracking features; per-possession top-1/top-3 accuracy.
- **EX-182** Counter-mapping table from coverage to action, coach-reviewed; measure
  agreement and usefulness.
- **EX-183** Lineup/rotation data model (blocked until game-state and substitution data
  exist).
