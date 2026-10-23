# Two clocks: release vs result

**Status:** working interpretation, 2026-10-22  
**Owner observation:** on the confirmed real field-goal attempts, the orange
candidate is consistently at the release, while the blue selected PBP event is at
the made shot/result.

## Definitions

| Anchor | Source | Meaning |
|---|---|---|
| **Orange candidate** | detector `ShotEvent.frame_start` | shot release / attempt anchor (human-confirmed on the audited shots; the trigger can be slightly before release) |
| **Blue selected event** | PBP event frame | made/missed result / scorekeeping anchor, not release |

The measured gap is therefore **release-to-result**, not detector error. It should
not be used to score release tracking.

## Two separate tasks

### 1. Release tracking

- Positive label: a human-confirmed real field-goal release.
- Negative label: pass, drive, rebound, turnover, free throw, or no release.
- Metric: candidate precision/recall after dedupe.
- Do **not** match against the blue PBP result timestamp.

### 2. Made-shot detection

- Outcome label: PBP `made` / `missed` at the blue event.
- Attempt anchor: the associated orange release candidate.
- Features: release pose, ball trajectory, rim/net geometry, scorebug/result
  context, shot type.
- Metric: made/missed accuracy, AUC, and calibration, conditioned on a detected
  attempt. This is separate from release precision.

### 3. Association / reinforcement

For each PBP field goal at blue time `t_blue`, search for an orange candidate in a
release window before it, approximately:

`[t_blue - 5.5 s, t_blue - 0.3 s]`

If a candidate is found, associate it with the PBP outcome. This gives a supervised
made/missed dataset without hand-labeling the outcome. The association rule must be
validated against human release labels and checked for adjacent-shot contamination.

## Experiment order

1. **Event detection first (shot vs no-shot).**
   - Positives: all 186 PBP field-goal attempts.
   - Negatives: PBP possessions/events with no field goal in the clip window.
   - Metric: clip-level and event-level precision/recall/F1 after dedupe.
   - This validates that the detector finds an attempt before we attach an outcome.

2. **Made-shot classification second.**
   - Inputs: confirmed shot attempts associated with PBP outcomes.
   - Labels: PBP `made` / `missed`.
   - Metrics: accuracy, AUC, calibration, split by shot type.
   - Mask or exclude the scorebug/score region so the model cannot read the result
     from the broadcast graphics.

3. **Release tracking third.**
   - Inputs: human release frames on a subset.
   - Metrics: trigger-to-release timing and release-frame error.
   - Do not use the blue PBP result timestamp as release ground truth.

Event detection comes first because a made/missed label is only meaningful once the
attempt is localized. Made-shot data is also easy to leak: the scorebug score,
crowd reaction, and replay package can reveal the result without the shot motion.
The PBP splice gives both label sets, but the event detector is the gate.

## Pilot association (20-minute audit)

On the 19 positive PBP clips:

- **15/19** have at least one human-confirmed real-shot candidate.
- Made: **7/9** have a confirmed candidate.
- Missed: **8/10** have a confirmed candidate.
- Several clips have two confirmed shot candidates; these are likely duplicate
  phases or adjacent attempts and must be deduped before unique-shot recall.

Candidate-level outcome counts from the audit:

- 8 made, 11 missed, 1 unknown (the real shot in the `neg_001` negative window).

This is too small to train a made-shot model. The association rate is the useful
pilot result; the 186-event expansion is the next data source.

## What this changes

- The previous symmetric event scorecard mixed the two clocks. It is a misaligned
  release metric, not a made-shot metric.
- The shifted −5.5/+1 s window is an association/sensitivity tool, not proof of a
  better release detector.
- A made-shot model should use the blue PBP outcome labels, but its input window
  should be anchored at the orange release and include the post-release trajectory.
- Do not report one “shot accuracy” number. Report release detection, made-shot
  classification, and association rate separately.

## Next experiment

1. Dedupe candidates per clip and associate each PBP field goal with at most one
   orange release candidate.
2. Compute association rate, release precision, and made/missed counts on the
   current 19 clips.
3. Expand to all 186 PBP events with the two-clock association.
4. Train a made-shot baseline only after the 186-event set and a second game exist.
5. Validate the release-to-result gap on the second broadcast; do not assume the
   scorekeeping lag is stable.
