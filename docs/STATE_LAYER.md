# State Layer - Design and v0 Contract

**Status:** v0 state models and alignment implemented 2026-10-21; PBP-derived
possession/lineup extraction v0.1 implemented 2026-10-21. Video possession
detection, scorebug OCR, and play-recognition integration are still future work.

## v0 implementation

- `inference/schema.py`: `GameState`, `Possession`, `Play`, and
  `LineupInterval`; optional `game_state` / `possessions` / `plays` / `lineups`
  on `ClipResult`; optional `possession_id` / `play_id` on events. Schema bumped
  to **1.3.0** for lineup intervals.
- `state_layer/manual.py`: manual game-state JSON validation and conversion to
  `GameState`.
- `state_layer/espn.py`: public ESPN PBP adapter, period/clock/score
  normalization, and final-score reconciliation.
- `state_layer/alignment.py`: explicit `(frame, period, clock)` anchors; linear
  interpolation only within an anchored clock range. No invented frames.
- `state_layer/possession.py`: heuristic PBP possession ledger with end reasons
  and point reconciliation.
- `state_layer/lineups.py`: five-player intervals from box-score starters and
  PBP substitutions.
- `state_layer/extract.py`: composes the ledger, intervals, and validation
  checks.
- `scripts/state_layer.py`: `fetch-pbp`, `validate-manual`, `attach`, `align`,
  and `extract`.
- Example extraction:

  ```powershell
  python scripts/state_layer.py extract `
      --timeline data/state/401810030/timeline.json `
      --manual state_layer/examples/manual_game.json `
      --out data/state/401810030/state.json `
      --scorecard data/state/401810030/scorecard.md
  ```
- One-game alignment artifact: Knicks 137-Timberwolves 114, ESPN `401810030`,
  466 PBP plays, 48 manual anchors, 466/466 assigned frames. This is coverage,
  not a frame-accuracy claim.
- One-game extraction artifact: 201 possessions (100/101 by offense), final
  points reconciled exactly (114/137), 74 substitutions applied, 47 lineup
  intervals, 13/14 players used per team, and all eleven extraction checks
  pass. POSS estimate deltas are +1.04 and +2.52 against a per-team tolerance of
  +/-5; PBP ORtg/DRtg proxies are within +/-5 points per 100 possessions of the
  box-score proxy.

The one-game scorecards are generated under `data/state/401810030/`
(gitignored). The video is a research-only YouTube asset outside Git.

## Why

`ClipResult` knows about frames, players, the ball, and events. It does not know:

- who has possession,
- who is on the floor,
- what period/clock/score it is,
- which possession or play an event belongs to.

Without that, events are atoms with no game context. Play recognition, Four
Factors, EPV, matchups, and coach reports all need this layer.

## Entities

```
game
  period (Q1..Q4, OT1..)
    possession (offense, defense, start/end, points)
      play (play_type, coverage, family, confidence)
        event (shot, pass, screen, rebound, foul, turnover)
    lineup interval (team, five player IDs, start/end frame)
```

## Models (additive to `inference/schema.py`)

```python
class GameState(_Model):
    period: int                         # 1..4, 5+ = OT
    clock_s: float | None               # seconds remaining in the period
    score_home: int
    score_away: int
    home_team_id: int | None
    away_team_id: int | None
    possession_team_id: int | None
    lineup_home: list[int] = []         # roster IDs in v0
    lineup_away: list[int] = []
    ruleset: str = "unknown"            # nba_2025_26 | fiba_2026 | nfhs | ncaa
    footage_type: str = "unknown"       # nba_broadcast | fiba_broadcast | fixed_gym | handheld
    camera_spec: dict | None = None     # height_ft, position, zoom, source


class Possession(_Model):
    possession_id: int
    offense_team_id: int
    defense_team_id: int
    start_frame: int
    end_frame: int | None = None
    points: int = 0
    end_reason: str | None = None       # made_shot | defensive_rebound | turnover | period_end | other
    confidence: float = 0.0


class Play(_Model):
    play_id: int
    possession_id: int
    play_type: str                      # ID from research/ontology/play_taxonomy.json
    coverage: str | None = None         # ID from the same taxonomy
    family: str | None = None
    confidence: float = 0.0
    evidence_frames: list[int] = []


class LineupInterval(_Model):
    team_id: int
    start_frame: int
    end_frame: int | None = None
    player_ids: list[int] = []          # ESPN roster IDs in v0
    source: str = "unknown"
    confidence: float = 0.0
```

Add to `ClipResult`:

```python
game_state: GameState | None = None
possessions: list[Possession] = []
plays: list[Play] = []
lineups: list[LineupInterval] = []
```

Add to each event model:

```python
possession_id: int | None = None
play_id: int | None = None
```

Schema v1.2.0 added the state objects; v1.3.0 adds `LineupInterval` and
`ClipResult.lineups`. Both are additive; existing clips still load.

## Sources of truth in v0.1

| Layer | v0.1 source | Status |
|---|---|---|
| Game clock / score | Manual anchors for video; ESPN PBP for events | Implemented |
| Possession | ESPN scoring, rebound, turnover, and period events | Heuristic; points reconcile |
| Lineup composition | ESPN box-score starters and substitution events | Deterministic from PBP |
| Lineup video identity | Not yet; no jersey/roster matching | Future |
| Video possession | Not yet; no tracking-based detector | Future |
| Scorebug OCR | Not yet; manual anchors | Future |

The v0.1 possession ledger is not an official possession feed. It uses a
deterministic state machine; every possession carries `confidence=0.5` to signal
that the boundary is inferred rather than observed.

## Invariants

1. Exactly five players per lineup when known; empty means unknown, not zero.
2. Score is non-decreasing; clock is non-increasing within a period.
3. A possession has one offense and one defense; they differ.
4. Possession changes only on a made shot, defensive rebound, turnover, or
   period start.
5. An event's `possession_id` must resolve; `play_id` is optional.
6. A play belongs to exactly one possession; a possession can have zero or more
   plays.
7. Period is 1-4 or an overtime value; OT is not period 0.
8. `ruleset` and `footage_type` travel with every clip; no implicit defaults in
   reports.
9. A known lineup interval has five distinct player IDs and a start frame no
   later than its end frame.
10. Lineup intervals cover the anchored game span without gaps; substitutions
    at the same aligned frame are recorded as changes, not zero-length
    intervals.

## Evaluation

| Layer | Metric | Gate | v0.1 status |
|---|---|---|---|
| Game state | score/clock accuracy vs box score/scorebug | exact score; clock within 1 s | PBP/score exact; video clock not independently scored |
| Possession | boundary F1 at +/-1 event | usable on 3 games | Not measured; one-game count reconciliation only |
| Lineup | 5-player accuracy per stoppage | 100% at manually entered stoppages | Five-player and player-set checks pass one game; not video-verified |
| Accounting | POSS/ORtg/DRtg vs box score | reconcile within a stated tolerance | Points exact; POSS within +/-5 per team; rating proxies within +/-5 points/100 |
| Play | per-possession top-1/top-3 | beat a PnR-vs-ISO baseline | Not started |

The unit is the possession, grouped by game. A frame-level accuracy is not
acceptable.

## Migration order

1. Add the models and optional fields; round-trip tests; schema minor bump.
2. Manual game-state form and JSON import; no model dependency.
3. NBA play-by-play adapter for the broadcast lab.
4. PBP possession ledger and lineup intervals; reconciliation against box score.
5. Video possession detector from tracking + events; compare to PBP.
6. Scorebug OCR for broadcast; compare to PBP.
7. Play recognition consumes `possession_id` and `play_id`.

## Open questions

1. Is the clock source the scoreboard, the broadcast scorebug, or manual input?
2. Do we use track IDs or roster IDs for lineups? v0.1 uses ESPN roster IDs;
   track IDs remain ephemeral until jersey/roster matching exists.
3. How do we represent and-1s, technical free throws, and team rebounds in
   `points`? The v0.1 ledger assigns scoring to the active offense and
   reconciles the final score; it does not model technical free throws as a
   separate possession.
4. How do we handle a possession that spans a period end?
5. Do we need a separate `team_possession` for offensive rebounds (same
   possession) vs a new possession after a made shot?
6. How do we mark a possession as "unknown" when the camera cuts away?
7. Should zero-length lineup intervals remain in the artifact for auditability,
   or are substitution-change events sufficient? v0.1 keeps changes in
   diagnostics and emits only positive-length intervals.

## Non-goals

- A full play-by-play replacement for the official scorer.
- A referee clock/score system.
- A lineup optimizer; that is a later value layer.

## Implementation status

- [x] Models in `inference/schema.py`
- [x] Round-trip tests
- [x] Manual game-state JSON + validator
- [x] PBP adapter (ESPN public endpoint)
- [x] PBP possession ledger
- [x] Lineup intervals from starters and substitutions
- [x] One-game extraction scorecard
- [x] Independent NBA Stats API cross-check (`analytics/possessions.py`; see
      `research/knowledge/perception/09-possession-ledger.md`)
- [ ] Video possession detector
- [ ] Scorebug OCR
- [ ] Play-recognition integration
- [ ] Three-game possession-boundary F1

This document is the contract. Change it before changing the code.
