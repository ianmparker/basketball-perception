# Known failures

This list is intentionally blunt. A system that hides its failure modes is harder to
improve than one that names them.

## Event detection

- **Recall is the main gap.** On the full-game PBP benchmark, recall at ±3 s is 0.38
  (95% CI 0.31–0.46). Many made shots produce no candidate at all.
- **Free throws are a false-positive source.** A free throw looks like a shot attempt
  to the motion gate; the scorebug/state filter is not yet applied everywhere.
- **Passes and rim-area action can fire.** A pass near the rim or a rebound can look
  like a release. The shot verifier is trained on too few balanced negatives.
- **The verifier is not a win yet.** Verifier v1 was at chance on the audited
  candidate set (AUC 0.528); it is documented as a null result, not shipped as a
  solution.

## Timing

- **Two clocks.** Human labels place real-shot candidates a median 3.2 s before the
  PBP event. The symmetric scorecard is therefore a misaligned view, not the final
  release metric.
- **Release-frame calibration is open.** Trigger-to-release and release-to-PBP are not
  yet separated by made/missed outcome.

## Camera regimes

- **Broadcast does not transfer to top-down high-school footage.** Two labeled
  top-down candidates were both false positives (0/2). Ball motion, player scale, and
  rim geometry all change.
- **A single homography assumes a fixed camera.** Broadcast cuts and zooms invalidate
  it; per-segment calibration is required for moving cameras.

## Identity and attribution

- **Team ID drifts on top-down footage.** The assigned split collapsed to 9.9% / 90.1%
  on one high-school clip, which is a team-ID failure rather than a possession signal.
- **Shooter attribution can be wrong.** Frame inspection found a rim attack attributed
  to a player on the wing.
- **Unassigned player-frames remain.** About 7.7k player-frames were unassigned on a
  1,200-frame broadcast clip.

## Product and runtime

- **Not real-time.** The demo is an offline, 1,200-frame path; live delivery is a later
  phase.
- **Not commercial-clean.** Training data includes research-only footage; the task
  weights are not cleared for redistribution.
- **Play recognition is out of scope.** Screens and pick-and-rolls are experimental
  event detectors, not product claims.
