---
id: perception.state.nba-stats-pbp-crosscheck
title: NBA Stats API play-by-play cross-check
layer: perception
track: basketball.analytics
status: active
confidence: high
updated: 2026-10-21
sources: [nba-stats-api, oliver2004, arxiv-2608.19646]
code: [analytics/possessions.py, scripts/nba_pbp_ingest.py, state_layer/possession.py, docs/STATE_LAYER.md]
prereqs: [basketball.ontology.schema]
tags: [possession, play-by-play, state, NBA, cross-check]
---

# NBA Stats API play-by-play cross-check

**Claim.** The official NBA Stats API v3 is an independent **second source** for validating
the canonical state layer. On game `0022300001` it reproduces the final score exactly
(IND 121, CLE 116) and lands within **2.7%** of the box-score possession estimate.

The canonical ledger is `state_layer/possession.py` (ESPN PBP, pydantic `Possession`,
clock-to-frame alignment, lineup intervals). This note and `analytics/possessions.py` are
the cross-check, not a replacement.

## What was built

- `scripts/nba_pbp_ingest.py` fetches `playbyplayv3` and caches normalized actions.
- `analytics/possessions.py` runs a conservative state machine: possession changes on
  made shots, defensive rebounds, turnovers, and period boundaries; offensive rebounds
  keep it alive; the last made free throw closes it.
- `report()` prints the ledger and compares it to the box-score possession estimate.

## Result: game `0022300001` (IND 121, CLE 116)

| Metric | NBA Stats cross-check | Box score |
|---|---:|---:|
| Actions | 504 | — |
| Possessions | 205 | 210.6 (107.6 + 103.0) |
| IND points | 121 | 121 |
| CLE points | 116 | 116 |
| IND possessions | 101 | 107.6 |
| CLE possessions | 104 | 103.0 |

Possession count is **2.7% low** overall, a reasonable v0 against the
`FGA + 0.44·FTA − ORB + TOV` estimate (itself an approximation).

End-reason mix: 89 made shots, 63 defensive rebounds, 31 turnovers, 15 made free throws,
4 period ends, 3 team changes, 2 technical free throws, 2 free-throw state mismatches.

## The data quirks that mattered

1. **`playbyplayv2` returns `{}` for this game.** Use `playbyplayv3`.
2. **Made free throws have blank `shotResult`.** A miss is marked by `MISS` at the start of
   the description; otherwise the FT is made.
3. **Free throws have `shotValue = 0`.** Count one point per made FT.
4. **Technical free throws are standalone points.** If you add them to the current
   possession, they get credited to whichever team happens to have the ball.
5. **`teamId` can be 0** for some non-possession events. Skip those for possession state.
6. **Two non-technical FTs had a shooting team different from the open possession**
   (`free_throw_state_mismatch`). Points are correct; the surrounding possession state is
   worth a manual look (likely an and-1 or a delayed foul sequence).

## Why this is useful

- **Cross-source validation.** The canonical ledger uses ESPN; this uses NBA Stats. Two
  independent feeds agreeing on points and staying within a few possessions is a real
  integrity check.
- **Data-quirk checklist.** Any PBP adapter, including the canonical one, needs the FT and
  technical-FT rules above.
- **Possession-boundary second opinion.** When the ESPN ledger and the NBA Stats ledger
  disagree, the disagreement is a concrete debugging target.
- **No replacement.** The canonical ledger has pydantic models, frame alignment, lineups,
  and a scorecard; this one has none of those.

## Repo mapping

- `state_layer/possession.py` — canonical possession ledger.
- `scripts/state_layer.py` — canonical fetch/align/extract CLI.
- `analytics/possessions.py` — NBA Stats cross-check ledger.
- `scripts/nba_pbp_ingest.py` — NBA Stats fetch/cache/report.
- `analytics/four_factors.py` — box-score POSS estimate used for comparison.

## Failure modes

- **Treating this as the product ledger.** It has no frame alignment, lineups, or pydantic
  contract. The canonical layer is `state_layer/`.
- **Treating box-score POSS as truth.** A 2.7% gap is not automatically a ledger bug.
- **Crediting technical FTs to the current possession.** Points land on the wrong team.
- **Counting an offensive rebound as a new possession.** It is the same possession.
- **Ignoring and-1 / delayed foul sequences.** These produce the FT state mismatches.

## Open questions

1. Which of the 2 FT state mismatches are and-1s, and which are ledger bugs?
2. Can the NBA game ID be mapped to the ESPN event ID for a direct ledger comparison?
3. Per quarter, where do the two ledgers diverge?
4. Which play types can be inferred from the event sequence alone, and which need video?

## Proposed experiments

- **EX-220** NBA Stats API PBP cross-check — **done**. 205 possessions, points exact,
  POSS within 2.7%.
- **EX-221** Cross-source ledger comparison (NBA Stats vs ESPN) on one game; report
  possession-boundary disagreements and points per team.
- **EX-222** Per-quarter ledger reconciliation against the box score; expose the
  and-1/technical/team-rebound edge cases.
