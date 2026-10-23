# Screen and PnR-defense ontology

Status: **draft for Basketball Domain Expert review**. This document defines the
event semantics for the screen/PnR branch. It does not change the inference
schema in this branch.

Scope: broadcast/full-court basketball, **on-ball pick-and-roll only**. The
MultiSports labels used for the benchmark are single-player action tubes, not
pair annotations. The benchmark is a proxy, not product ground truth.

## 1. Rulebook basis

A **screen** is an action whose legality is a separate judgment. The camera
detects the action; it does not decide whether the screen was legal.

- FIBA, *Official Basketball Rules 2026*, Art. 33.7 (Screening: Legal and
  illegal), p. 40: "Screening is an attempt to delay or prevent an opponent
  without the ball from reaching a desired position on the court." The same
  article lists the legal and illegal screening cases. Source: `fiba2026`.
- NBA, *Official 2025-26 NBA Playing Rules*, Rule 4, Section X (Screen), p. 19:
  "A screen is the legal action of a player who, without causing undue contact,
  delays or prevents an opponent from reaching a desired position." Source:
  `nba2025rules`.
- Illegal screening in the NBA is covered separately in Rule 12B, Section III,
  p. 46. Do not infer it from the Rule 4 definition.

"Pick-and-roll defense" is not an official rule term. It is a tactical and
ontology label, defined for this project from the observable action and the
MultiSports annotation protocol (`arxiv-2105.07404`, Appendix E.4, pp. 18â€“19).

The MultiSports definitions we map from are:

> **Screen:** "In pick-and-roll, the offensive player uses his body to set a
> pick for his ball-controlling teammate. Start: Both feet of the offensive
> player touches the ground. End: Any foot of the offensive player is ready to
> leave the ground completely. Small range movement is not considered the end."

> **Pick-and-roll Defense:** "In pick-and-roll, the defender of the offensive
> ball-controlling player is blocked by the teammate of this offensive player.
> Start: The defender has physical contact with the offensive screening player.
> End: The defender does not have physical contact with the offensive screening
> player."

The MultiSports screen class is an **on-ball screen**. Off-ball screens are out
of scope for this branch and need their own examples, labels, and agreement
study before they can share the same event ID.

## 2. Operational definitions

### `screen.on_ball` (action)

A screen is an offensive player's attempt to delay or prevent the screened
defender from reaching a desired position while the screener's team has
possession.

Observable test:

- the screener places his body in the screened defender's path;
- the screener's team is in an on-ball pick-and-roll;
- the screened defender is the primary defender responsible for the ball
  handler at the start of the action.

Event presence does **not** require contact, and it does **not** require the
screen to be legal. Stationary feet, cylinder, distance, and moving-screen
criteria belong to the separate `legality` field. A no-contact screen is still a
screen event; a moving screen is still a screen event and may be an illegal one.

### `pnr.defense.contact` (draft tactical event)

A PnR-defense contact interval is the period in which the screened defender is
in physical contact with the screener after being obstructed by the on-ball
screen.

Observable test:

- an on-ball screen is present;
- the defender of the ball handler contacts the screener;
- the interval starts at the first contact frame and ends at the first frame
  with no physical contact.

This is the MultiSports `basketball pick-and-roll defensive` class. It is **not**
the same as the offensive screen event, and it is **not** the same as the
pipeline's `PnREvent`, which combines an offensive on-ball screen with a later
shot.

Post-contact recovery, drop, hedge, switch, or roll is a separate
`post_contact_outcome` field; do not fold it into the contact interval.

## 3. Roles

| Role | Definition | Source in MultiSports |
|---|---|---|
| Screener | Offensive player setting the screen | `basketball screen` tube box |
| Ball handler / screen user | Offensive player controlling the ball and using the screen | Not a target tube; assign from possession and ball tracking |
| Screened defender | Defender primarily responsible for the ball handler at screen start | Not a target tube; assign from tracking and matchup |
| On-ball defender | Same as the screened defender for this on-ball event | `basketball pick-and-roll defensive` tube box |

**Cutter** is reserved for off-ball screens (the player receiving the screen).
It is not a synonym for the on-ball ball handler. The inference schema's
`cutter_id` is a legacy name; do not use it for the on-ball screen-user role.

A `screen.on_ball` event requires the triple **screener + ball handler +
screened defender**. A frame-level detection without that triple is a
`proposal`, not a scored event. Roles are assigned at the start of the event;
a switch may change the assignment after the event starts.

## 4. Interval semantics

The current JSON stores start frames for most events. The source tubes contain
both starts and ends, and the benchmark now keeps the defensive interval in a
separate `pnr_defenses` field. The benchmark still scores **starts only**.

| Event | Start | End |
|---|---|---|
| `screen.on_ball` | First frame the screener is positioned to affect the screened defender | First frame the screener clearly releases the screen or abandons it |
| `pnr.defense.contact` | First frame of physical contact between the screened defender and screener | First frame with no physical contact |

A possession-level `pnr_possession` interval (screen start through the ball
handler's decision/shot/pass) is a different event and must not be conflated
with the contact interval.

## 5. Ontology IDs and mapping

These are proposals for the branch; they are not added to the shared research
taxonomy, and no inference-schema change is made here.

The shared research note currently defines `screen.on_ball` as a "legal
stationary screen." This draft deliberately separates the action from legality;
that is a proposed ontology change, not an edit to `research/`. The Basketball
Domain Expert must resolve the shared definition before annotation.

| Ontology ID | Parent | Meaning | Repo field today |
|---|---|---|---|
| `screen.on_ball` | screen | On-ball screen action; legality separate | `screens` |
| `pnr.defense.contact` | pnr | Screened defender's contact interval | `pnr_defenses` (benchmark only) |
| `screen.legality` | screen attribute | `unknown` / `legal` / `illegal` | not implemented |
| `pnr.post_contact_outcome` | pnr attribute | recover, drop, hedge, switch, roll | not implemented |

MultiSports mapping:

| MultiSports label | Ontology mapping | Limitation |
|---|---|---|
| `basketball screen` | `screen.on_ball`; tube box is the screener | No cutter/ball-handler/screened-defender IDs |
| `basketball pick-and-roll defensive` | `pnr.defense.contact`; tube box is the screened defender | Contact interval only; no full possession |
| `basketball sag` | Separate defensive action / confounder | Not part of `pnr.defense.contact` |
| `basketball pass`, shots, etc. | Not this branch | Do not add other event classes here |

Field mapping and gaps:

- `ScreenEventModel.frame` corresponds to label `screens[].frame_idx` /
  `frame_start`; `ScreenEventModel` has no `frame_end`.
- `ScreenEventModel.cutter_id` is a legacy field; it must not be read as the
  screen user unless the role mapping is explicit.
- `ScreenEventModel` has no `screened_defender_id`, `legality`, `ruleset`, or
  `status` field.
- `PnREventModel` is an offensive screen + shot model. It must not be used to
  score MultiSports `pnr.defense.contact`.
- The benchmark's `pnr_defenses` is an evaluation sidecar; `pnrs` is empty.

## 6. Positive and negative examples

All examples are from the MultiSports `test_videos` held-out split, which this
branch uses for evaluation only. Frame numbers are 1-based in the GT pickle.

### Positive examples

**Screen**

1. `v_4LXTUim5anY_c012`, screen frames 54â€“71; PnR-defense contact frames 54â€“72.
   On-ball screen with immediate defender contact.
2. `v_4LXTUim5anY_c012`, screen frames 150â€“167; PnR-defense contact frames
   152â€“183. The earlier pair also exists in the same clip; both are on-ball.

**PnR-defense contact**

1. `v_4LXTUim5anY_c012`, defensive contact frames 152â€“183 (screen 150â€“167).
   The defender contact starts after the screen starts.
2. `v_82WN8C8qJmI_c015`, defensive contact frames 60â€“69 (screen 58â€“67).
   The contact starts after the screen starts and ends after the screen ends.

### Negative examples

**Screen**

1. `v_4LXTUim5anY_c006`, frame 100. The clip has no screen or PnR-defense tubes;
   a normal half-court frame is not a screen.
2. `v_2j7kLB-vEEk_c001`, frame 100. Same: no screen target, despite close
   bodies in a normal possession.

These are valid **class negatives**, but they are not hard visual negatives. A
hard-negative set must be built separately.

**PnR-defense contact**

1. `v_2j7kLB-vEEk_c006`, frame 87. Screen tube 81â€“95; no PnR-defense tube. The
   screen is present, but there is no annotated defender contact. The clip also
   contains `basketball sag` (95â€“132), which is a separate coverage action; do
   not count it as PnR-defense contact.
2. `v_-6Os86HzwCs_c001`, frame 111. Screen tube 108â€“115; no PnR-defense tube.
   Again, screen without annotated defender contact.

### Known invalid/ambiguous examples

- `v_4LXTUim5anY_c008` (PnR defense 82â€“103, no screen tube) is **not** a valid
  screen negative. The MultiSports PnR-defense definition entails a screen; the
  missing screen tube is an annotation omission or a labeling inconsistency.
- `v_2j7kLB-vEEk_c005` PnR defense starts at frame 801, five frames before the
  screen tube (806â€“823). This is either a moving-screen contact or an annotation
  ordering inconsistency. It is not a clean canonical positive.
- The earlier version of this document marked every screen `off_ball`; that was
  a mapping error. MultiSports `basketball screen` is on-ball by definition.

## 7. Disagreement criteria

Two annotators review the same 30-clip pilot sample, grouped by source video.
The unit of analysis is one **event interval within one clip and possession**.

Matching rule before comparison: two annotations match if temporal IoU â‰¥ 0.5
with the same clip, possession, and event type. Do **not** require role agreement
to establish a match; roles are compared after matching. Unmatched annotations are
presence disagreements. A third adjudicator resolves disagreements; annotators
are blinded to each other and to model outputs.

Call a disagreement when:

- **Presence:** one annotator marks an event and the other does not;
- **Roles:** matched events assign a different screener, ball handler, or
  screened defender;
- **Start:** matched events have start frames differing by more than 5 frames
  (200 ms at 25 fps);
- **End:** matched events have end frames differing by more than 10 frames.

`screen_type` is not a disagreement axis in this branch: only on-ball screens
are in scope. Off-ball screens require a separate study.

## 8. Inter-annotator agreement target

Pilot, not product-grade:

- event-presence Cohen's Îº â‰¥ **0.70**;
- joint exact-match agreement for the screener + ball handler + screened
  defender triple â‰¥ **0.60** (per-role agreement alone is not enough);
- median absolute start-boundary difference â‰¤ **5 frames**;
- 90th-percentile absolute start-boundary difference â‰¤ **10 frames** for events
  at least 15 frames long;
- median absolute end-boundary difference â‰¤ **10 frames**;
- 90th-percentile absolute end-boundary difference â‰¤ **15 frames**.

Report prevalence, PABAK in addition to Îº, and cluster-robust confidence
intervals by source video. Thirty clips is a pilot, not a stable estimate; the
target applies to a larger adjudicated set before release.

## 9. Decision consequences

- The YOLO frame detector is evaluated as a **proposal stage**.
- A proposal cannot be promoted to a screen event without the required triple.
- The benchmark is **start-frame only**; no full-boundary or pair claim is
  allowed from it.
- The rule-based `PnREvent` is offensive screen + shot; it is not a defensive
  contact baseline.
- No pipeline integration or inference-schema change happens in this branch
  until the domain review and the 60-clip go/no-go are complete.

## Sources

- `fiba2026` â€” FIBA, Official Basketball Rules 2026, Art. 33.7, p. 40.
- `nba2025rules` â€” NBA, Official 2025-26 NBA Playing Rules, Rule 4 Â§X, p. 19;
  Rule 12B Â§III, p. 46.
- `arxiv-2105.07404` â€” MultiSports, ICCV 2021, Appendix E.4, pp. 18â€“19.
- `basketball.ontology` and `basketball.rules` research notes for the shared
  taxonomy; this branch does not edit them.

License note: MultiSports is CC-BY-NC-4.0, research-only. The benchmark and
weights are not for commercial redistribution.
