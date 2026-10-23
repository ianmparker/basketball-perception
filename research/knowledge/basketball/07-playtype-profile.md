---
id: basketball.strategy.playtype-profile
title: NBA play-type profiles from Synergy data
layer: basketball
track: basketball.strategy
status: active
confidence: high
updated: 2026-10-21
sources: [synergy-sportradar, nba-stats-api, taylor2016]
code: [analytics/playtypes.py, scripts/nba_playtype_ingest.py, scripts/playtype_report.py, scripts/nba_scout_report.py]
prereqs: [basketball.strategy.playbooks]
tags: [play types, synergy, scouting, team profile, play recognition]
---

# NBA play-type profiles from Synergy data

**Claim.** The NBA Stats Synergy play-type endpoint gives the first quantitative answer
to "what does this team run?" It is **aggregate** frequency and efficiency, not
per-possession video recognition — but it is the right prior for scouting and the right
benchmark for the video layer.

## What was built

- `scripts/nba_playtype_ingest.py` fetches the 11 team or player play types for a season
  and caches them.
- `analytics/playtypes.py` normalizes the rows, computes league averages, and prints team
  and player profiles.
- `scripts/playtype_report.py` reports from a cached file.

Data fields: frequency (`POSS_PCT`), points per possession (`PPP`), eFG%, turnover rate,
free-throw rate, percentile, and per-game possessions/points.

## Example: Indiana Pacers, 2023-24

| Play type | Freq | PPP | eFG% | TOV% | FT% | Percentile |
|---|---:|---:|---:|---:|---:|---:|
| Spot-up | 23.8% | 1.067 | 0.550 | 5.9% | 5.4% | 69 |
| Transition | 20.0% | 1.160 | 0.625 | 11.1% | 11.0% | 66 |
| PnR ball handler | 18.1% | 0.959 | 0.520 | 12.5% | 7.2% | 83 |
| PnR roll man | 7.0% | 1.336 | 0.667 | 3.9% | 14.8% | 100 |
| Cut | 6.9% | 1.400 | 0.712 | 4.6% | 16.6% | 100 |
| Misc | 5.1% | 0.547 | 0.441 | 53.0% | 19.7% | 59 |
| Putback | 4.8% | 1.230 | 0.639 | 6.8% | 11.7% | 100 |
| Isolation | 4.4% | 0.988 | 0.498 | 9.0% | 12.2% | 76 |
| Handoff | 3.9% | 0.961 | 0.519 | 9.7% | 5.3% | 69 |
| Post-up | 3.5% | 1.112 | 0.573 | 9.9% | 18.0% | 97 |
| Off-screen | 2.6% | 0.991 | 0.505 | 8.1% | 7.2% | 48 |

League comparison:

| Play type | Team freq | League freq | Team PPP | League PPP |
|---|---:|---:|---:|---:|
| Spot-up | 23.8% | 24.0% | 1.067 | 1.048 |
| Transition | 20.0% | 17.2% | 1.160 | 1.136 |
| PnR ball handler | 18.1% | 16.4% | 0.959 | 0.904 |
| PnR roll man | 7.0% | 5.8% | 1.336 | 1.143 |
| Cut | 6.9% | 6.9% | 1.400 | 1.311 |
| Isolation | 4.4% | 6.7% | 0.988 | 0.935 |
| Post-up | 3.5% | 4.2% | 1.112 | 0.986 |

The profile reads as a high-tempo, spot-up-and-transition team that uses ball screens
more than the league and isolation less. Its roll-man and cut efficiency are elite
(100th percentile); its off-screen game is below league.

## Player profiles (EX-231)

The same endpoint supports `PlayerOrTeam=P`. The Pacers rotation, by total possessions:

| Player | Poss/g | Top play | Freq | PPP | Pctile |
|---|---:|---|---:|---:|---:|
| Pascal Siakam | 18.8 | Transition | 20.5% | 1.206 | 66 |
| Tyrese Haliburton | 18.4 | PnR ball handler | 43.4% | 1.055 | 91 |
| Myles Turner | 12.9 | PnR roll man | 27.7% | 1.397 | 87 |
| Bennedict Mathurin | 11.3 | PnR ball handler | 22.6% | 0.858 | 52 |
| Bruce Brown | 10.9 | Spot-up | 28.5% | 0.915 | 28 |
| T.J. McConnell | 9.7 | PnR ball handler | 52.7% | 1.066 | 93 |
| Andrew Nembhard | 9.3 | PnR ball handler | 34.5% | 0.833 | 46 |
| Buddy Hield | 8.6 | Spot-up | 38.4% | 1.014 | 45 |

Tyrese Haliburton's full profile:

| Play type | Freq | PPP | eFG% | TOV% | FT% | Pctile |
|---|---:|---:|---:|---:|---:|---:|
| PnR ball handler | 43.4% | 1.055 | 0.573 | 13.0% | 8.4% | 91 |
| Transition | 17.6% | 1.121 | 0.637 | 16.6% | 9.9% | 47 |
| Spot-up | 14.1% | 1.101 | 0.559 | 4.5% | 7.9% | 68 |
| Isolation | 10.3% | 1.031 | 0.500 | 6.9% | 13.1% | 72 |
| Handoff | 5.9% | 1.000 | 0.545 | 9.3% | 1.3% | 67 |
| Misc | 4.7% | 0.583 | 0.304 | 38.3% | 16.7% | 66 |
| Off-screen | 1.5% | 1.316 | 0.781 | 15.8% | 5.3% | 92 |
| Putback | 1.0% | 1.000 | 0.500 | 7.7% | 7.7% | 29 |

The scouting read: Haliburton is a high-volume PnR handler at the 91st percentile;
Turner is an efficient roll man; McConnell is a PnR-heavy backup. Player rows only
include play types where the player had possessions, so the table is not a full
taxonomy for every player.

## Scout report (EX-240)

`python scripts/nba_scout_report.py --team-id 1610612754 --season 2023-24` combines the
team Advanced/Four Factors table, the team and player play-type profiles, and the team
shot chart into one Markdown page under `outputs/nba-reports/`.

The Indiana 2023-24 read:

- 2nd in ORtg, 24th in DRtg, 2nd in pace.
- 1st in eFG%, 6th in TOV%, 16th in ORB%, 26th in FT rate.
- Opponents: 22nd in eFG% allowed, 18th in turnover creation, 26th in offensive
  rebounds allowed, 30th in FT rate allowed.
- Attack: paint/3s, the offensive glass, and rim pressure.
- Most-used action: spot-up (23.8%); most efficient: cut (1.400 PPP).

This is a pre-game **prior**, not a possession plan. It says what to expect and where the
data says the matchup is soft; it does not say which play to call.

## What this does not tell you

- **Which possession** was a PnR, an ISO, or a post-up. Synergy is aggregate.
- **Who** ran it, who screened, who was covered, or what the defense did.
- **Whether the video agrees.** That requires per-possession labeling and recognition.
- **Why** the efficiency is high: personnel, shot quality, or opponent mix.

## How to use it

1. **Scouting prior.** Pull the opponent's play-type mix before a game; identify the
   actions to take away.
2. **Benchmark.** A video play classifier should be compared with Synergy frequencies on
   the same team/season, not against a generic "PnR vs ISO" guess.
3. **Player profiles.** The same endpoint supports `PlayerOrTeam=P`; player-level play-type
   profiles are the next cut.
4. **Report context.** A team play-type table belongs in the pre-game scouting report, not
   the post-game shot chart.

## Data quirks

- **`Putback` returns HTTP 400.** Fetch it as `OffRebound`; the response label is
  `OffRebound`.
- **`PRRollMan` casing.** The response uses `PRRollMan`, not `PRRollman`.
- **Taxonomy mapping.** `research/ontology/play_taxonomy.json` uses the endpoint labels
  (`PRRollMan`, `OffRebound`); `research/ontology/validate.py` enforces a one-to-one map.
- **`Misc` is turnover-heavy.** Its 53% TOV rate is a category artifact, not a coaching
  signal.
- **Synergy is commercial ground truth.** Use it for research and evaluation; do not ship
  it as our own data.

## Repo mapping

- `analytics/playtypes.py` — team/player normalize, league averages, and reports.
- `scripts/nba_playtype_ingest.py` — fetch/cache.
- `scripts/playtype_report.py` — report from cache.
- `scripts/nba_scout_report.py` — one-page team scout report combining Four Factors,
  play types, player roles, and shot profile.
- `research/ontology/play_taxonomy.json` — our play-type IDs; map Synergy labels to them.

## Failure modes

- **Treating aggregate frequencies as per-possession labels.** They are priors.
- **Reading `Misc` as a real play type.** It is a residual bucket.
- **Comparing across seasons without checking the label definitions.** Synergy changes
  categories occasionally.
- **Using the data commercially without a license.** It is not ours to ship.
- **Ignoring opponent context.** A team's play mix depends on the defense it faces.

## Open questions

1. How stable are team play-type frequencies across a season and across opponents?
2. Can player-level profiles identify the actions a team runs for a specific player?
3. How do we map Synergy's 11 labels to our taxonomy without losing "Misc"?
4. What is the minimum per-possession label set for a video play classifier?
5. Can we learn a play classifier from tracking + lineup data without Synergy labels?

## Proposed experiments

- **EX-230** NBA Synergy team play-type profile — **done**. 30 teams × 11 types; IND
  2023-24 example above.
- **EX-231** Player-level play-type profiles (`PlayerOrTeam=P`) — **done**. Pacers
  rotation and Haliburton profile above.
- **EX-232** Compare a video play classifier to Synergy frequencies on the same team and
  season; gate on per-possession top-1/top-3.
- **EX-240** Pre-game scout report v0 — **done**. One-page Markdown combining Four
  Factors, play types, player roles, and shot profile; Indiana 2023-24 example above.
- **EX-241** Opponent-specific scout report: combine both teams' profiles into a matchup
  page with explicit strengths, weaknesses, and attack/defend priorities.
