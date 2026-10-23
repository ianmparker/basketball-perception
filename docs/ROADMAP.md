# Roadmap

The public roadmap follows the project's research phase boundaries.
Dates are targets, not commitments.

## Phase A — Broadcast demo (current)

A one-command broadcast pipeline, a broadcast event benchmark with a scorecard, a
shot chart/report, and a documented set of known failures.

Exit gate: the demo runs on a broadcast clip; the event scorecard reports clip- and
event-level counts with tolerances; the failure list is current.

## Phase B — Handheld/tripod high-school adapter

A fixed/elevated high-school benchmark, three handheld/tripod games calibrated, a shot
chart a coach can use, and the broadcast demo still passing.

Exit gate: the high-school path has its own calibration and verifier; broadcast metrics
do not regress.

## Phase C — Live capture, stoppage-time delivery

Event-to-card latency under 30 seconds in a three-game pilot with no event spam.

Exit gate: a live path that is fast enough to be useful and quiet enough to trust.

## Phase D — Phone-only and strategy layer

Local inference with acceptable latency, battery, and accuracy parity on a phone, plus
a strategy/reporting layer built on the possession ledger.

Exit gate: a phone demo that a coach can run without a laptop.

## How work is chosen

One bet per six-week cycle, a WIP limit, and a circuit breaker. Scope changes require
a written decision; the plan and the scorecards are the source of truth. See
`docs/EVALUATION.md` for the evidence bar.
