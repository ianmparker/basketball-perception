# Second-Game Validation

**Game:** ESPN `401766463`, Mavericks–Grizzlies  
**Purpose:** test whether the game-1 event detector/non-shot filter transfers to a
different broadcast. No tuning on game 2.

## Dataset

- 172 PBP field goals ingested from ESPN.
- 40-clip validation subset selected: 10 per quarter, 20 made / 20 missed.
- 36/40 aligned; 4 skipped because no scorebug clock sample matched.
- 30 balanced six-second no-FG negatives.
- Source is 60 fps; clips resampled to 30 fps for the pipeline.
- Center-bottom ESPN scorebug, not the MSG bottom-left layout.

This is a validation subset, not a full second-game benchmark.

## Event detection

| `MIN_RISE_PX` | Positive clips with any shot | Negative clips with any shot |
|---:|---:|---:|
| 60 | 20/36 (56%) | 10/30 (33%) |
| 80 | **14/36 (39%)** | **9/30 (30%)** |
| 100 | 9/36 (25%) | 7/30 (23%) |
| 120 (default) | 5/36 (14%) | 4/30 (13%) |

Event-level at rise80, ±3 s: **P 0.42 / R 0.28 / F1 0.33**.  
At default, ±3 s: P 0.33 / R 0.08 / F1 0.13.

The game-1 rise80 result was 135/162 positive clips (83%). The second game is
**14/36 (39%)** — a large domain shift. This is the main failure, before the filter.

## Filter validation

Train on game 1 (162 positives + 60 balanced negatives), then apply to game 2 at
the frozen threshold 0.3:

- Game-2 candidates: **24** (15 positive, 9 negative).
- AUC: **0.659**.
- Event F1 at ±3 s: **0.35** vs raw rise80 **0.33**.
- Positive recall: **0.28**.

**Verdict: fail / weak transfer.** The filter does not meaningfully improve game 2
and should not be adopted. The small candidate count makes the estimate unstable,
but there is no evidence of a useful transfer.

## What this means

1. **The detector, not just the filter, has a second-broadcast problem.** Candidate
   generation drops from 83% to 39% of positive clips at rise80.
2. The filter AUC falls from 0.728 (game 1) to 0.659 (game 2); the F1 gain is
   negligible.
3. Possible causes: ESPN camera/lighting/scorebug domain shift, ball detector
   transfer, clip alignment, or the 60→30 fps conversion. These need to be
   separated before another model attempt.
4. Game 1 remains the only validated event benchmark. Do not claim cross-broadcast
   generalization.

## Next

1. Audit a sample of second-game clips: is the ball tracked? Is the shot in the
   clip? Is the clip alignment wrong?
2. If alignment is wrong, fix the center scorebug/clock mapping.
3. If ball detection is the issue, evaluate/fine-tune the ball detector on the
   second broadcast before touching the filter.
4. Keep the filter frozen and not adopted.

Artifacts: `second_game_scorecard.md`, `second_game_threshold_sweep.md`,
`second_game_filter_validation.md`.
