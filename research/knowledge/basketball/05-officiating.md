---
id: basketball.officiating.review
title: Officiating science and referee decision support
layer: basketball
track: basketball.officiating
status: active
confidence: medium
updated: 2026-10-21
sources: [price_wolfers2010, nba2025rules, fiba_documents]
code: [events/, docs/]
prereqs: [basketball.rules.officiating]
tags: [officiating, bias, replay, review, fouls]
---

# Officiating science and referee decision support

**Claim.** A referee product is not a classifier with a whistle. It is an **evidence
system**: show the frame, cite the rule, state the confidence, and let a human decide.
Anything else is a liability and a credibility risk.

## The product boundary

| Layer | What we can provide | What we must not claim |
|---|---|---|
| Measurement | ball/player positions, clock, line crossing, contact distance | that the official had the same view |
| Rule | the relevant rule and its version | that the rule is unambiguous |
| Evidence | synchronized frames, angles, slow motion | intent, force, or "the ref missed it" |
| Decision | a ranked set of reviewable calls | an automated verdict |

The NBA and FIBA both publish rulebooks and interpretation documents; the review workflow
is league-specific. Encode the ruleset and the review trigger, not a universal "correct
call" label.

## What is objectively checkable

- out of bounds (ball/player vs line);
- clock and shot clock (if visible or available);
- 3-point line (feet vs arc at release);
- backcourt / 8-second;
- 5-second inbound;
- offensive 3-second;
- goaltending / basket interference (partially; cylinder and descent);
- basket counts (ball through the rim).

These are geometric and temporal. They are the right first products because the evidence
is visible and the rule is precise.

## What is judgment

- block vs charge (legal guarding position, timing, foot placement);
- common vs flagrant (wind-up, follow-through, impact);
- continuation / gather / travel (hand contact, pivot, dribble timing);
- incidental vs illegal contact;
- defensive 3-second (NBA) and off-ball holds.

For these, output **evidence and a rule citation**, not a binary. A coach may disagree; a
referee may have a better angle; the system should make the review faster, not replace it.

## Bias and identification

Price & Wolfers (2010) found evidence consistent with own-race bias in NBA foul calls.
The result has been debated, but the methodological lesson is durable:

- referees are assigned non-randomly to games;
- player/team composition is confounded with referee crew;
- foul rates are an outcome of both behavior and officiating;
- a raw correlation is not a bias estimate.

Any officiating analytics must state its identification assumption, control for game
context, and report uncertainty. Do not ship a bias claim from a foul-rate table.

## Referee mechanics as a camera prior

- Three-person crew: lead, trail, center; each has primary coverage areas and rotates.
- The responsible official is usually the one with the best angle, which is often outside
  the broadcast frame.
- Broadcast direction chooses the camera; the system sees what the director showed.
- Therefore: a "missed call" can be a **camera coverage** failure, not an officiating
  failure. Report visibility.

## Data sources

- official play-by-play (foul type, player, time);
- league last-two-minute reports (where published);
- referee assignments and crew;
- video with clock/score overlay;
- rulebook and interpretation documents (`fiba_documents`, `nba2025rules`).

## Repo mapping

- `events/` — event detectors produce the candidate calls; add a `reviewable` flag and a
  rule citation.
- `docs/` — this note and the rules note are the product spec.
- `inference/schema.py` — add `ruleset`, `period`, `clock`, `score`, and `visibility` to
  the clip metadata before any officiating feature.

## Failure modes

- **Claiming a missed call from one angle.** The official may have had a better view.
- **Using the referee's call as ground truth.** It is an expert label with noise and bias.
- **Ignoring the ruleset.** A FIBA travel is not an NBA travel in every detail.
- **No confidence.** A binary output invites over-trust.
- **No audit trail.** The evidence frames and rule version must be stored with the call.

## Open questions

1. Which reviewable calls have high enough inter-annotator agreement to automate?
2. Can we obtain clock/score from the broadcast scorebug reliably enough for review?
3. What is the legal/liability posture for a referee-assist product?
4. How do we present uncertainty to a referee without slowing them down?

## Proposed experiments

- **EX-160** Build a reviewable-call prototype for out-of-bounds and 3-point line
  violations; measure precision and evidence quality.
- **EX-161** Label 30 block/charge plays and measure agreement; decide go/no-go.
- **EX-162** Add a `visibility` field (official angle vs camera angle) to the review
  output and user-test with a referee.
