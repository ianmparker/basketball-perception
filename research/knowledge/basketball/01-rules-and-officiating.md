---
id: basketball.rules.officiating
title: Rules, dimensions, and officiating for a vision system
layer: basketball
track: basketball.rules
status: active
confidence: high
updated: 2026-10-21
sources: [fiba2026, nba2025rules, fiba_documents, price_wolfers2010]
code: [events/, docs/]
prereqs: []
tags: [rules, court, fouls, violations, refereeing, NBA, FIBA, NCAA]
---

# Rules, dimensions, and officiating for a vision system

**Claim.** A basketball event detector is a rule engine with a camera attached. If the
code does not know the difference between an NBA and a FIBA court, it will produce
confident, wrong events. Rule knowledge belongs in the ontology, not in a comment.

## Court and equipment

| Quantity | NBA | FIBA | NCAA |
|---|---|---|---|
| Court length × width | 94 × 50 ft (28.65 × 15.24 m) | 28 × 15 m (91.9 × 49.2 ft) | 94 × 50 ft |
| 3-point arc | 23 ft 9 in (7.24 m) | 6.75 m (22.15 ft) | 22 ft 1.75 in (6.75 m) |
| 3-point corner | 22 ft (6.71 m) | 6.60 m (derived: 7.50 − 0.90) | check current rule |
| Free-throw line | 15 ft from backboard | 5.80 m from endline | 15 ft from backboard |
| Restricted / no-charge | 4 ft radius | 1.30 m radius | 4 ft arc |
| Hoop height | 10 ft (3.05 m) | 3.05 m | 10 ft |
| Rim diameter | 18 in (45.7 cm) | 45 cm | 18 in |
| Ball circumference | 29.5 in (size 7) | 75–78 cm (size 7) | 29.5 in |
| Game length | 4 × 12 min | 4 × 10 min | 2 × 20 min |
| Shot clock | 24 s (14 s reset) | 24 s (14 s reset) | 30 s (check current rule) |
| Backcourt | 8 s | 8 s | 10 s (check men's rule) |
| Personal fouls | 6 | 5 | 5 |

Verified against the downloaded PDFs on 2026-10-21: FIBA 2026 Art. 2.1, 2.5, 2.5.7,
8.1, 26, 28, 29; NBA 2025-26 Rule 1 Section I, Rule 5, Rule 7, Rule 12. The NCAA column
is not fully verified — check the current NCAA rulebook before encoding it.

**These values change.** The source of truth is the current rulebook PDF, not this
table. Encode dimensions in a versioned config (`ruleset: nba_2025_26`) and cite the
document in the event metadata. A court template is a ruleset artifact.

## Violations a camera can adjudicate

| Violation | Geometric signal | Difficulty |
|---|---|---|
| Out of bounds | ball/player vs boundary line | low (line occlusion) |
| Shot clock / game clock | clock OCR or scorebug feed | low–medium |
| 3-second (offensive) | player in paint while ball controlled in frontcourt | medium |
| 8-second backcourt | ball + timer + halfcourt line | medium |
| 5-second inbound | ball + boundary + timer | medium |
| Backcourt violation | possession + halfcourt crossing | medium |
| Traveling | gather step, pivot foot, dribble timing | high (judgment + feet) |
| Double dribble / carry | hand/ball contact pattern | high |
| Goaltending / basket interference | ball descent + rim cylinder | medium (occlusion) |

## Fouls: the hard boundary

A foul is **illegal contact or an illegal action** as judged by a referee. Video can show
contact; it cannot show intent, the official's angle, or the rule's judgment thresholds.
Blocks vs charges depend on legal guarding position, foot placement, and timing. Flagrant
vs common depends on wind-up, follow-through, and impact.

Therefore a foul model should output **evidence + rule citation + confidence**, never a
binary "referee was wrong". That is a product and liability decision, not a modeling
preference.

## Officiating mechanics (useful for camera priors)

- Three-person crew: lead, trail, center. Each has primary coverage areas and rotates on
  transitions. The referee closest to the play is usually the one responsible for the
  call.
- Referees are trained to get the best angle, which means they are often *not* in the
  broadcast frame. A camera-only system sees what the director chose, not what the
  official saw.
- Replay triggers and coach's challenges are league-specific. Check the current rulebook
  before encoding a trigger.

## Bias: what the evidence says

Price & Wolfers (2010) found evidence consistent with own-race bias in NBA foul calls.
The finding has been debated and revisited, but the methodological lesson is durable:
officiating data are observational, referees are assigned non-randomly, and player/team
composition is confounded. Any "referee analytics" product must state its identification
assumption and report uncertainty. Do not ship a bias claim from a raw foul-rate table.

## Repo mapping

- `events/` — every detector should carry `ruleset` and a rule citation.
- `docs/` — this note is the bridge between the rulebook and the code.
- `eval/labels/schema.md` — event definitions should reference ontology IDs.
- `inference/schema.py` — add `ruleset`, `period`, `clock`, `score` to `ClipMetadata`
  before encoding clock-dependent events.

## Failure modes

- **Using an NBA court template on FIBA footage.** The 3-point arc and lane are different;
  shot classification and spacing metrics shift.
- **Ignoring the clock.** A "shot" at the buzzer has different semantics; a 3-second
  violation depends on the clock.
- **Treating a referee's call as ground truth.** It is an expert label with noise and
  bias, not a measurement.
- **Encoding a rule from memory.** Cite the rulebook page and version.
- **Claiming a missed call.** Without the official's angle and judgment framework, the
  claim is not supported.

## Open questions

1. Which ruleset is the product's primary target: NBA, FIBA, NCAA, or a configurable
   ruleset per game?
2. Can we obtain clock/score data from the broadcast scorebug reliably, or should we
   require a manual game-state input?
3. Which violations have high enough inter-annotator agreement to be worth automating?
4. What is the review workflow that keeps a human referee in the loop?

## Proposed experiments

- **EX-050** Add a `ruleset` config with court dimensions and shot-clock rules; make the
  court template a parameter, not a constant.
- **EX-051** Build a rule-constrained event validator (e.g. a 3-second violation cannot
  fire without possession and a paint occupancy interval) and measure false positives.
- **EX-052** Label 30 block/charge plays and measure inter-annotator agreement; decide
  whether this is in scope.
