# Play labeling protocol v1

Status: active. This is the labeling contract for the first per-possession play set.

## Goal

Produce a frozen set of **50 offensive possessions** with one primary play-type label
each, from broadcast video. This is the ground truth for the play-recognition baseline
(`EX-181`) and the video-vs-Synergy comparison (`EX-232`).

The labels are our taxonomy, not Synergy's. The taxonomy already carries the Synergy
mapping in `research/ontology/play_taxonomy.json`.

## Unit

One offensive possession. The clip starts at the possession start (capped at 14 s before
the ending event) and runs 3 s past the shot, rebound, or turnover. Possessions whose
start and end collapse to the same second are skipped by the clip builder; do not label
those.

Label the **offense** named in the card, not the team that happens to end the clip with
the ball. A clip can spill a few seconds into the next possession, especially after a
missed shot turns into a transition. Label only the card's possession; note the spill.
The next possession belongs on its own card.

## One primary label

Label the action that created the **first advantage**, not the final shot type. A PnR that
produces a spot-up 3 is `pnr_ball_handler`, not `spot_up`. A post-up that draws a double
and produces a corner 3 is `post_up`, not `spot_up`.

A set can contain several screens. Label the screen that forces the defense to make the
decision that produces the shot. An off-ball screen that is only an entry or decoy is not
`off_screen`.

Use this order:

1. **Transition** — the possession starts before the defense is set (roughly within 6 s
   of the change) and the offense attacks. Transition takes precedence over the half-court
   action that follows.
2. **Putback** — an offensive rebound is immediately followed by a shot attempt. Use this
   even if the previous action was a PnR.
3. **PnR ball handler** — the handler uses an on-ball screen and creates the advantage
   (shot, drive, or pass). Label this when the screen is the main action.
4. **PnR roll man** — the screener receives the pass and finishes or creates; the roll man
   is the terminal advantage. If the handler creates and the roll man only screens, label
   the handler.
5. **Dribble handoff** — the handoff is the main action, not a decoy.
6. **Post-up** — a catch with the back to the basket inside roughly 12 ft, followed by a
   post move. A switch into a post-up after a PnR is still a PnR if the screen created it.
7. **Off-ball screen** — a player runs off a screen away from the ball and the screen
   creates the catch/shot.
8. **Cut** — an off-ball cut creates the catch/shot; no screen is the main action.
9. **Spot-up** — a stationary catch outside roughly 15 ft creates the shot or closeout
   attack. Do not use this for the final catch of a PnR or post-up advantage.
10. **Isolation** — one ball handler attacks a set defender with no screen or off-ball
    action creating the advantage.
11. **Miscellaneous** — a broken play, loose ball, or action that does not fit. This is a
    real category, not a dump for hard clips.

## Ambiguity

If two labels fit, choose the action that created the first advantage and mark the clip
**ambiguous** with a note naming the alternative. Do not invent a new label. If the clip
is unlabelable (replay, cut, scoreboard, wrong possession), leave it unlabeled and note
why.

### Example: off-ball screen into PnR

The clip starts with an off-ball screen, then the offense flows into a ball screen near the
end. If the late PnR is what forces the switch/hedge and produces the shot:

- label `pnr_ball_handler` if the handler shoots or passes out of it;
- label `pnr_roll_man` if the screener rolls and finishes;
- mark **ambiguous** and note `off-ball screen entry into PnR`.

Do not label `off_screen` unless the off-ball screen itself created the open catch or shot.
If the late ball screen is a decoy and the off-ball action produced the advantage, reverse
the choice.

## What not to use

- Do not read the play-by-play text. Label from the video.
- Do not use the final shot type as the play type.
- Do not label defense/coverage in this pass. Coverage is a second stage.
- Do not force a label when the clip does not show the possession.

## Workflow

1. Run the label page from `outputs/play-labels/<event>/`:

   ```powershell
   python -m http.server 8765 --directory outputs/play-labels/401810030
   ```

2. Open <http://localhost:8765/labels.html>.
3. Label the first 10 possessions. Stop and review the decision rules before continuing.
4. Label the remaining 40. Export/download `labels.json` into the same directory.
5. Validate:

   ```powershell
   python scripts/validate_play_labels.py `
     --labels outputs/play-labels/401810030/labels.json `
     --possessions outputs/play-labels/401810030/possessions.json `
     --require-all
   ```

A valid first set has 50 labels, no unknown IDs, and an ambiguity rate we can explain.
Class balance is descriptive, not a target; the sample is random, so rare actions may be
absent. Add targeted clips later rather than distorting the first set.

## Files

- `scripts/make_play_label_page.py` — cut clips and build the page.
- `analytics/play_labels.py` — clip-window and label-validation logic.
- `scripts/validate_play_labels.py` — label-file contract.
- `outputs/play-labels/<event>/possessions.json` — clip manifest.
- `outputs/play-labels/<event>/labels.json` — exported labels (local, not committed).
