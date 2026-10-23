---
id: basketball.ontology.schema
title: Basketball ontology and event vocabulary
layer: basketball
track: basketball.ontology
status: active
confidence: medium
updated: 2026-10-21
sources: [nba_stats_glossary, bref_glossary, arxiv-2607.21267, fiba2026]
code: [inference/schema.py, eval/labels/schema.md, events/]
prereqs: [basketball.rules.officiating]
tags: [ontology, events, taxonomy, stats, labels]
---

# Basketball ontology and event vocabulary

**Claim.** The ontology is the contract between perception, annotation, analytics, and
the coach. If the same word means different things in the detector and the label file,
the model is learning noise. One name, one definition, one parent, one citation.

## Entities

```
game
├── period (Q1..Q4, OT1..)
├── team (home, away)
│   └── lineup (5 players on court)
├── player (roster identity, jersey number)
├── official (crew of 3)
└── possession (offense/defense, start/end, points)
    └── play (a sequence with a purpose: PnR, ISO, transition, post-up)
        └── event (atomic, timestamped)
```

Every record should carry: `game_id`, `possession_id`, `play_id` (nullable), `event_id`,
`period`, `clock`, `team_id`, `player_id`, `ruleset`, and `schema_version`. The repo's
`ClipResult` has `metadata` and per-frame data but no possession/play/event IDs yet; that
is the main schema gap.

## Event taxonomy v1

`repo_field` is where the event lands today. `—` means no field exists.

| ID | Parent | Definition | repo_field |
|---|---|---|---|
| `shot.attempt` | shot | A player propels the ball toward the basket with intent to score | `shots` |
| `shot.make` | shot | Ball enters the basket from above through the rim | — |
| `shot.miss` | shot | Attempt does not enter; includes blocked attempts | — |
| `shot.free_throw` | shot | Uncontested attempt from the free-throw line | — |
| `shot.three` | shot | Attempt with both feet beyond the 3-point line at release | `shots.shot_type` |
| `shot.mid_range` | shot | Attempt inside the arc, outside the paint | `shots.shot_type` |
| `shot.layup` | shot | Attempt at the rim with one hand, close to the basket | `shots.shot_type` |
| `shot.dunk` | shot | Attempt where the player drives the ball through the rim | `dunks` |
| `pass.completed` | pass | Ball travels from one teammate to another, possession retained | `passes` |
| `pass.intercepted` | pass | Opponent gains possession from the pass | `interceptions` |
| `screen.on_ball` | screen | Legal stationary screen on the ball handler's defender | `screens` |
| `screen.off_ball` | screen | Legal stationary screen away from the ball | `screens` |
| `pnr.roll` | play | Screener rolls to the rim after the screen | `pnrs` |
| `pnr.pop` | play | Screener pops to the perimeter after the screen | `pnrs` |
| `rebound.offensive` | rebound | Offense recovers a missed shot | — |
| `rebound.defensive` | rebound | Defense recovers a missed shot | — |
| `turnover.live` | turnover | Possession lost while ball is live (steal, etc.) | — |
| `turnover.dead` | turnover | Possession lost on a dead ball (travel, out of bounds) | — |
| `foul.personal` | foul | Illegal contact or action against an opponent | — |
| `foul.shooting` | foul | Personal foul on a shooter; free throws awarded | — |
| `foul.technical` | foul | Non-contact technical | — |
| `violation.travel` | violation | Illegal movement of the pivot foot | — |
| `violation.three_seconds` | violation | Offensive player remains in the paint too long | — |
| `violation.backcourt` | violation | Offense returns the ball to the backcourt | — |
| `violation.out_of_bounds` | violation | Ball or player breaks the boundary | — |
| `substitution` | game | A lineup change | — |
| `timeout` | game | A team-requested stoppage | — |

The ten BasketEvent categories (made/missed shot, free throw, foul, turnover, rebound,
assist, block, steal, jump ball) plus a background class are a good cross-check. If our
taxonomy cannot express a BasketEvent label, we cannot evaluate against that dataset.

## Stats vocabulary

Use the standard formulas so numbers are comparable:

```
eFG%   = (FGM + 0.5 · 3PM) / FGA
TS%    = PTS / (2 · (FGA + 0.44 · FTA))
TOV%   = TOV / (FGA + 0.44 · FTA + TOV)
ORB%   = ORB / (ORB + opp. DRB)
FTR    = FTA / FGA
ORtg   = 100 · PTS / POSS
DRtg   = 100 · opp. PTS / POSS
Pace   = POSS per 48 min (NBA) or 40 min (FIBA)
POSS   ≈ FGA + 0.44 · FTA - ORB + TOV   (with team-rebound corrections)
```

Do not invent a "basketball IQ" score before these are correct. The glossary sources
(NBA Stats, Basketball-Reference) are the tie-breakers.

## Roles and actions

Roles (not positions): ball handler, screener, roller, popper, corner spacer, weak-side
cutter, rim protector, help defender, on-ball defender. Positions (PG/SG/SF/PF/C) are
roster metadata, not tracking labels — a modern center can be a hub.

Actions: cut (backdoor, curl, flare), closeout, hedge, drop, switch, ice, trap, rotate,
box out, outlet. Each action needs a definition and a minimal observable signal; see
`basketball.tactics.spacing`.

## Governance

Add a class only when all five exist:

1. a one-sentence definition;
2. two positive and two negative video examples;
3. a rulebook or glossary citation;
4. an inter-annotator agreement target (κ ≥ 0.6 to start);
5. a consumer (which report or model uses it).

Deprecate a class by keeping the ID and marking it `deprecated`; never reuse an ID for a
different meaning.

## Repo mapping

- `inference/schema.py` — `ClipResult`, `ShotEventModel`, `ScreenEventModel`,
  `PnREventModel`, `DunkEventModel`, `PassEventModel`, `InterceptionEventModel`.
- `eval/labels/schema.md` — label fields (`passes`, `shots`, `screens`, `pnrs`, `dunks`,
  `team_assignments`, `tracking_checks`).
- `events/` — detector outputs; map each to a taxonomy ID.
- `analytics/passes.py` — pass/interception events.

## Failure modes

- **Multiple names for one event.** "Pick", "screen", "block" (not the shot block) are
  used interchangeably in coaching speech. Pick one ID and list synonyms.
- **Instant vs interval.** A screen is an interval; a shot release is an instant. The
  schema must support both (`frame_start`, `frame_end`).
- **Parent/child confusion.** A dunk is a shot; a block is not a rebound. Get the
  hierarchy right before counting.
- **League-specific rules.** A FIBA 3-second violation is not an NBA defensive 3-second.
  Put `ruleset` on the event.

## Open questions

1. Should `play` be a first-class entity with its own ID, or a tag on a possession?
2. Do we need separate IDs for "shot attempt" and "field goal attempt" (free throws are
   attempts but not field goals)?
3. Which stats are computed from tracking vs play-by-play, and how do we reconcile them?
4. How do we represent an event that is later overturned by replay?

## Proposed experiments

- **EX-060** Freeze ontology v1 as JSON (`research/ontology/events.json`), generate the
  Pydantic enums from it, and fail tests on unknown IDs.
- **EX-061** Map the 10 BasketEvent classes to our taxonomy; report unmapped classes.
- **EX-062** Add `possession_id` and `play_id` to `ClipResult` (additive, schema minor
  bump) and measure downstream report quality.
