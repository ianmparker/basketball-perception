# Evaluation harness

Ground truth and scoring for per-clip event, possession, and tracking metrics. The
harness is deliberately small: metrics are implemented directly so the numbers can be
audited without a black-box dependency.

## Layout

```
eval/
├── README.md            this file
├── schema.md            label JSON format
├── labels/              one JSON per labeled clip
├── diagnostics.py       noise floor, shuffled-label check, grouped scorecards
├── metrics/             metric implementations
│   ├── events.py        event P/R/F1 at a frame or second tolerance
│   ├── detection.py     event + team-assignment scores
│   ├── tracking.py      MOTA / IDF1 / ID switches + player-count checks
│   └── possession.py    team/jersey possession accuracy
├── score.py             driver: ClipResult JSON + labels → metric table
└── results/             per-run output (gitignored)
```

## Running it

```bash
# Score every labeled clip (runs the pipeline if a cached ClipResult is missing)
python eval/score.py

# Score one clip against its cached ClipResult
python eval/score.py --clip <clip> --skip-pipeline

# Re-run the pipeline even if a cached ClipResult exists
python eval/score.py --clip <clip> --refresh
```

Cached artifacts are re-scored by default. The driver warns when an artifact is older
than `--max-artifact-age` days (default 7) and records `artifact_ages_days` in the
scorecard JSON. Use `--refresh` after changing detection, tracking, or homography.

The pipeline is invoked with the flags the labels were scored against, held in
`EVAL_PIPELINE_ARGS`. These are benchmark flags, not generic detector defaults:

```
--ball-conf 0.25  --shot-no-court-required  --kmeans-teams  --offline-events
```

`--offline-events` is load-bearing: online detection runs before the ball track is
smoothed, so it fires nothing on these clips. Replaying the detectors over the
smoothed frames is what produces the benchmark's shot events. Override with
`--pipeline-arg` when comparing settings deliberately.

## What the driver prints

1. The per-clip event table.
2. **Localization sweep** — F1 at each `--tolerances` value (default `5,15,30`
   frames). The primary threshold stays strict; the extra columns show how close a
   detector is without loosening the pass mark.
3. **Noise floor** — with n labeled events, differences under X F1 points are not
   measurable.
4. **Person-count spot checks** — observed vs expected on labeled frames. The observed
   count includes referees and bench players on broadcast footage, so an over-count is
   expected and is not by itself a tracking failure.
5. **Track-ID churn** — unique tracks, long tracks, and fragments. Many fragments in a
   sparse scene means ID switches or ghost detections.
6. **Possession distance mode** — frames scored in court vs pixel units.
7. **Shuffled-label check** — re-scores against scrambled labels. A warning means the
   metric is finding signal in noise; inspect the matcher before trusting the run.
8. **Grouped scorecards** — mean F1 per game and per footage type, with a bootstrap
   95% CI over clips.

## MultiSports broadcast benchmark

`eval/multisports_v1/` builds a screen/PnR/pass/shot/block/rebound benchmark from the
public MultiSports validation split (CC-BY-NC-4.0). Official action tubes are mapped
to start frames. See `eval/multisports_v1/README.md`.

## Tracking metrics

`eval/metrics/tracking.py` implements MOTA, IDF1, ID switches, and IoU matching
directly (no TrackEval dependency). `scripts/export_mot.py` writes
MOTChallenge-format rows:

```bash
python scripts/export_mot.py --clip-json eval/results/<clip>.clip.json \
    --out data/mot --sequence <clip>
```

Add `gt/gt.txt` for the sequence and point TrackEval (MIT) at the directory if you want
their exact implementation; the built-in scorer is for CI and quick comparisons.

## Statistical rules

- **The unit of independence is the clip, not the frame.** 18,420 frames from 12 clips
  is 12 samples. Error bars are bootstrapped over clips.
- **Group by game.** `eval/labels/<clip>.json` may include
  `"game_id"` and `"footage_type"`. Grouped scorecards use those keys.
- **Event count sets the noise floor.** A 2-point F1 difference is not measurable on
  120 labeled events, no matter how clean the table looks.
- **Shuffled labels must crater.** If they do not, the tolerance window or the split is
  leaking.

## Adding a labeled clip

1. Copy an existing `labels/<clip>.json` as a template.
2. Watch the clip frame-by-frame in VLC (`j`/`l` to step).
3. Fill in `passes`, `shots`, `screens`, `pnrs`, `dunks`, and `team_assignments`.
4. Add `game_id` and `footage_type` so grouped scorecards stay meaningful.
5. Run `python eval/score.py --clip <your-clip>` to confirm it parses.
