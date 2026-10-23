---
id: measurement.basketball.four-factors
title: Four Factors, pace, and team evaluation
layer: measurement
track: basketball.analytics
status: active
confidence: high
updated: 2026-10-21
sources: [oliver2004, bref_glossary, arxiv-2002.04148, nba-stats-api]
code: [analytics/, eval/]
prereqs: [basketball.ontology.schema]
tags: [four factors, ORtg, pace, lineups, RAPM]
---

# Four Factors, pace, and team evaluation

**Claim.** Before any tracking-derived metric, the box-score accounting has to be right.
Possessions, pace, and the Four Factors are the ledger; tracking metrics are annotations
on top. A coach will forgive a missed screen before they forgive a wrong point total.

## The ledger

```
POSS  ≈ FGA + 0.44 · FTA - ORB + TOV        (team rebounds and technicals need care)
ORtg  = 100 · PTS / POSS
DRtg  = 100 · opp. PTS / POSS
Pace  = POSS per 48 min (NBA) or 40 min (FIBA)
```

The `0.44` coefficient estimates the share of free throws that end a possession. It is an
approximation; play-by-play and-1s and technical free throws break it. State the formula
and the exceptions in the code.

## The Four Factors

| Factor | Formula | What it answers |
|---|---|---|
| Shooting | `eFG% = (FGM + 0.5·3PM) / FGA` | Did we make shots efficiently? |
| Turnovers | `TOV% = TOV / (FGA + 0.44·FTA + TOV)` | Did we waste possessions? |
| Offensive rebounding | `ORB% = ORB / (ORB + opp. DRB)` | Did we buy extra possessions? |
| Free throws | `FTR = FTA / FGA` | Did we get to the line? |

These are the four levers a coach can actually influence. They are not independent
(e.g. more 3s can raise eFG% and lower ORB%), so report them together and with pace.

## v0 check: NBA box score

`analytics/four_factors.py` computes the ledger from an NBA box score. For
`0022300001` (IND 121, CLE 116):

| Team | PTS | POSS | ORtg | DRtg | eFG% | TOV% | ORB% | FTR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| IND | 121 | 107.6 | 112.5 | 112.6 | 0.610 | 0.157 | 0.189 | 0.279 |
| CLE | 116 | 103.0 | 112.6 | 112.5 | 0.571 | 0.120 | 0.132 | 0.298 |

Pace 105.3. The numbers match a hand calculation; the `TO` vs `TOV` field difference was
the first real bug the check caught.

This validates the accounting layer on a real box score. It is not a video-derived
ledger yet; the next step is to compute the same table from `ClipResult` and reconcile.

## Sample size and noise

- One NBA game is ~100 possessions per team. One lineup can have 20 possessions in a
  week. The noise is large.
- Do not rank lineups on raw plus-minus. Use **adjusted plus-minus** (ridge regression of
  possession margin on player indicators) or a Bayesian hierarchical model with
  shrinkage.
- Report uncertainty. A 5-point ORtg difference over 10 games is not a signal.
- Regression to the mean is real: extreme shooting percentages early in a season revert.

The intrinsic-dimension paper is a useful warning: high-resolution tracking data has a
low effective dimension, so adding hundreds of correlated features buys nothing and
inflates variance.

## Player valuation

| Metric | What it is | Caveat |
|---|---|---|
| PER | Box-score composite, pace-adjusted | Weights are arbitrary; ignores defense |
| BPM | Box-score plus-minus regression | Team/lineup confounding |
| Win Shares | Box-score contribution to wins | Depends on team performance |
| RAPM | Ridge-adjusted plus-minus from possessions | Collinearity; needs many possessions |
| EPV | Tracking-based expected points | Model-dependent; selection bias |

Use them as **priors and sanity checks**, not as ground truth. The tracking-based EPV
layer is the differentiator; the box-score metrics are the baseline it must beat.

## Coaching metrics that matter more than a composite

- shot profile (rim / mid-range / 3 distribution) vs league;
- turnover types (live vs dead, forced vs unforced);
- offensive rebounding by lineup;
- free-throw rate by player and situation;
- transition frequency and efficiency;
- half-court vs transition ORtg split;
- defensive coverage mix (man/zone/drop/switch) and points per possession against each.

These are actionable. "Player X has a PER of 18" is not.

## Repo mapping

- `analytics/` — possession accounting and team metrics.
- `eval/` — scorecards, not analytics; keep the distinction.
- `inference/schema.py` — `ClipResult` has events but no score/possession ledger yet;
  add a `game_state` block before computing ORtg/DRtg from video.

## Failure modes

- **Wrong possession count.** The most common analytics bug; audit against play-by-play.
- **Garbage time.** Exclude or flag; rotations change the meaning of the number.
- **Opponent adjustment.** Raw ORtg vs a weak defense is not the same as vs a good one.
- **Lineup collinearity.** Players who always play together cannot be separated by RAPM;
  report the uncertainty, not a point estimate.
- **Composite scores as truth.** They encode assumptions; show the components.

## Open questions

1. Can we recover possession boundaries from video alone, or do we need play-by-play?
2. What is our ground truth for points scored — scorebug OCR, play-by-play, or manual?
3. Which Four Factors are stable enough at the clip level to use in a scorecard?
4. How do we present uncertainty to a coach without overwhelming them?

## Proposed experiments

- **EX-090** Compute POSS, ORtg, DRtg, and the Four Factors from one labeled game;
  reconcile with the public box score.
- **EX-091** Build a possession ledger from `ClipResult` and compare to play-by-play on
  10 games.
- **EX-092** Estimate the noise floor for ORtg over 1, 5, and 20 games using bootstrap.
- **EX-190** NBA box-score Four Factors baseline — **done**; `analytics/four_factors.py`
  matches a hand calculation on `0022300001`.
- **EX-200** NBA game report v0 — **done**; combines Four Factors, shot profile, and a
  short coach read from the box score + shot chart.
