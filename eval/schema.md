# Label Schema v1 (Phase 15)

One JSON file per labeled clip, `eval/labels/<video_stem>.json`.

## Grouping fields (B-03)

Add these so frame-level noise cannot masquerade as a real improvement:

| Field          | Type   | Notes |
|----------------|--------|-------|
| `game_id`      | string | Groups clips from the same game. Falls back to the clip stem when omitted. |
| `footage_type` | string | e.g. `nba_broadcast`, `college_broadcast`, `phone_gym`. |
| `status`       | string | `reviewed`, `draft`, or `unlabeled`. Only `reviewed` labels are scored. |

`eval/score.py` reports mean F1 per group with a bootstrap 95% confidence
interval over clips (never over frames).

## Top-level fields

| Field          | Type       | Notes |
|----------------|------------|-------|
| `clip`         | string     | filename of the source video (must exist under `data/sample/videos/` or `eval/clips/`) |
| `fps`          | number     | source fps (usually 30) |
| `duration_frames` | int     | total labeled frames; events must be `< duration_frames` |
| `possession`   | list[Possession] | time-indexed segments, one per ball-carrier |
| `passes`       | list[Pass] | |
| `interceptions`| list[Interception] | |
| `shots`        | list[Shot] | |
| `screens`      | list[Screen] | optional |
| `pnr`          | list[PnR] | optional; a screen that led to a shot by the screener or handler |

## Item schemas

### Possession
```json
{"start_f": 0, "end_f": 42, "team": 1, "player_jersey": 7}
```
`team` ∈ {1, 2}. `player_jersey` is the number printed on the jersey (not the
pipeline's tracker ID). Use `null` if jersey not visible.

### Pass
```json
{"frame": 43, "from_jersey": 7, "to_jersey": 23}
```
`frame` is the frame where the pass is *received* (ball reaches receiver).

### Interception
```json
{"frame": 89, "by_jersey": 4}
```

### Shot
```json
{"frame": 132, "shooter_jersey": 4, "made": true, "type": "layup"}
```
`type` ∈ {`layup`, `dunk`, `jump_shot`, `three_pointer`, `free_throw`}.
`frame` is the frame where the ball leaves the shooter's hand.

### Screen
```json
{"frame": 21, "screener_jersey": 23, "cutter_jersey": 7, "on_ball": false}
```
`frame` = frame of closest approach (screener + cutter within ~3 ft).

### PnR (pick-and-roll)
```json
{"frame": 21, "shot_frame": 45, "outcome": "pull_up"}
```
`outcome` ∈ {`pull_up`, `roll_shot`, `kick_out`, `turnover`, `unknown`}.

## Tolerances (matched against pipeline output)

| Event          | Frame tolerance |
|----------------|-----------------|
| Shot           | ±15 frames (0.5 s @ 30 fps) |
| Pass           | ±15 frames |
| Interception   | ±15 frames |
| Screen         | ±30 frames (1.0 s) |
| PnR            | ±60 frames (2.0 s) |

Within tolerance = true positive. Outside = false positive / false negative.

## Jersey → tracker-ID mapping

Optional `jersey_map` section:
```json
"jersey_map": {
  "per_track": {"7": [12, 18], "23": [3], "4": [7]}
}
```
Lists pipeline tracker IDs that correspond to each jersey number over the clip
(tracker may fragment; one jersey → list of IDs). Used by possession metric to
resolve carrier identity. Author by eyeballing the pipeline's annotated MP4.
If absent, possession is scored at the **team** level only.

## Example

See `eval/labels/broadcast_74.json`.
