# Broadcast Demo Scorecard — v1

**Date:** 2026-10-22
**Benchmark:** `eval/broadcast_demo/manifest.json`
**Status:** demo sanity check, not an accuracy claim.
**Label-driven event metrics:** `eval/broadcast_demo/event_scorecard.md`.

> **Event benchmark:** the full-game PBP set in [`../pbp_shots/`](../pbp_shots/) is
> the accuracy benchmark. The Kyrie runs here are the visual demo and a small
> sanity check; do not use their 4/5 number as the event claim.

## Runs

| Run | Regime | Frames | Ball detected | Shots | Screens | PnRs | Passes | Interceptions | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `kyrie_1200_current` | NBA broadcast | 1200 | 73.4% | 2 | 4 | 0 | 5 | 5 | Current defaults; 1 shot has no valid location |
| `kyrie_1200_legacy` | NBA broadcast | 1200 | 52.6% | 3 | 4 | 0 | 10 | 6 | 2026-10-20 run; 3-shot comparison |
| `lima_q2b_450` | HS top-down | 450 | 65.3% | 2 | 1 | 1 | 2 | 2 | Fixed/elevated HS camera, not an NBA broadcast; 35% court-gate rejections; team ID imbalanced |
| `broadcast_74_20` | NBA broadcast | 20 | 80.0% | 0 | 1 | 0 | 0 | 0 | Smoke test only |

## Labeled result (2026-10-22)

Seven candidates were labeled by the owner with the criterion “a real shot attempt
with the release visible in the clip.” **The two regimes must be read separately.**

| Regime | Runs | Real | False positive | Candidate precision (Wilson 95% CI) |
|---|---|---:|---:|---|
| **NBA broadcast** | Kyrie current + legacy | **4** | **1** | **0.80** (0.38–0.96) |
| **HS top-down probe** | Lima | 0 | 2 | 0.00 (0.00–0.66) |
| Mixed (not the headline) | All | 4 | 3 | 0.57 (0.25–0.84) |

Recall is **not measured**: the label set contains only predicted candidates, so a
real shot the detector missed is not represented. Add missed shots with
`predicted_frame: null` to measure it.

### False-positive taxonomy

| ID | Regime | Run | What the detector fired on | Evidence | Fix direction |
|---|---|---|---|---|---|
| FP-1 | NBA broadcast | `kyrie_1200_legacy` frame 439 | A **free throw**, not a field-goal attempt | 15.3 ft distance, scorebug reads `FREE THROWS 2/3` | Exclude free-throw sequences (scorebug text or stationary-shooter rule) |
| FP-2 | HS top-down | `lima_q2b` frame 288 | A **pass/drive**, no release | Court location `[0, 10]` invalid; contact sheet shows ball movement, no shot | Phase B: release verifier + HS-specific tuning |
| FP-3 | HS top-down | `lima_q2b` frame 356 | A **rim-area pass/rebound**, plus a camera cut | Ball near the rim but no shot; clip includes a cut | Phase B: court-view continuity + release verifier |

### What the labeled result says

- **The NBA broadcast path is a usable candidate generator:** 4 of 5 candidates are
  real shots (2/2 current, 2/3 legacy). The one false positive is a free throw.
- **The HS top-down view does not transfer:** 0 of 2 candidates are real. This is a
  camera-regime gap, not a broadcast failure. The top-down view changes ball motion,
  player scale, and rim geometry; NBA-tuned thresholds should not be expected to
  carry over.
- **Do not tune NBA thresholds on Lima.** The right next step for HS is a separate
  Phase B benchmark, calibration, and verifier.
- **The broadcast next step is a verifier, not another threshold sweep:** free-throw
  exclusion, then more broadcast labels to measure recall.

## What this says

1. **The one-command demo path works.** A 1200-frame 1080p broadcast clip was
   processed end-to-end: ball, players, court, team/possession, events, tactical
   view, JSON, MP4, page, and event clips.
2. **The current ball detector is better on this clip.** Raw ball detection rose
   from 52.6% (legacy 7-class, conf 0.50) to 73.4% (replay-v3, conf 0.25) while
   the smoothed track still covers every frame.
3. **The event counts are not stable across runs.** The current run finds 2 shots
   and 5 passes; the legacy run finds 3 shots and 10 passes. Without labels, the
   difference is not a quality ranking.
4. **Shot location and shooter attribution are the weak link.** One current shot
   projects to the court corner `[94, 50]`, which means the ball was off-camera or
   the projection failed. The page drops invalid locations and labels them in the
   timeline. The shooter track can also be wrong; frame inspection shows a rim
   attack attributed to a player at the wing.
5. **Team assignment is usable but incomplete.** About 7.7k player-frames are
   unassigned on the 1200-frame clip; the assigned frames split roughly 50/50
   between the two teams. On the HS top-down clip the split collapses to
   9.9% / 90.1%, which is a team-ID failure, not a real possession signal.
6. **The HS top-down probe is a transfer check, not a broadcast win.** It found a
   “layup” and a PnR, but the owner labeled both shots as false positives. The
   lesson is that the broadcast detector does not transfer to the top-down HS
   camera; Phase B needs its own calibration and verifier.
7. **Event clips work.** Every shot/screen/PnR/dunk has a 4–4.5 second H.264 clip
   on the demo page, so a reviewer can judge whether the event is real.

## What this does not say

- It does not measure recall. The labels contain only predicted candidates; missed
  shots are not in the file.
- It does not compare camera regimes. The NBA broadcast and HS top-down numbers are
  separate; the mixed 0.57 is not a product number.
- It does not prove the court coordinates are correct for the shots; the chart is
  a candidate view, not a shot chart a coach should trust yet.
- It does not cover made/miss, rebounds, blocks, fouls, or play recognition.

## Known failures to fix next

| Priority | Failure | Evidence | Next step |
|---|---|---|---|
| 1 | Free throw counted as a shot | Kyrie legacy frame 439, 15.3 ft, scorebug `FREE THROWS 2/3` | Add a free-throw guard |
| 2 | Shot location invalid or approximate | `[94, 50]` corner projection on the current run | Use the shooter/rim geometry at the release, not the gather ball position |
| 3 | Shooter attribution wrong | Frame 595: rim attack attributed to a wing player | Match the shot to the player nearest the rim/ball at release |
| 4 | Recall unmeasured | Labels contain only candidates | Add missed shots with `predicted_frame: null` |
| 5 | HS top-down does not transfer | 0/2 real on Lima | Phase B: separate HS benchmark, calibration, and verifier |
| 6 | Court-view gate rejects too much on the HS clip | 35% of the Lima 450-frame segment is crowd/close-up | Pick continuous-court segments; add a pre-roll trim step |
| 7 | Team ID collapses on the HS clip | 9.9% / 90.1% split, implausible | Phase B: tune or require a manual team override |
| 8 | Only one full NBA broadcast source | Kyrie current + legacy are the same source; the longer second segment is running | Finish the second Kyrie segment, then a third broadcast |

## Reproduce

```powershell
# Current run (replace with your own clip)
python inference/run_video.py --video "/path/to/clip.mp4" --frames 1200 `
  --out outputs/broadcast_demo/demo_pipeline.mp4 `
  --json-out outputs/broadcast_demo/demo_clip.json `
  --stats-out outputs/broadcast_demo/demo_stats.csv `
  --events --pose --team-id --possession --court-view-gate --offline-events

python scripts/broadcast_demo.py `
  --run-json outputs/broadcast_demo/demo_clip.json `
  --render outputs/broadcast_demo/demo_pipeline.mp4 `
  --out-dir outputs/broadcast_demo/demo_current `
  --title "Broadcast Demo (current pipeline)" --event-clips
```

The outputs are local; the source footage is research-only and is not redistributed.
