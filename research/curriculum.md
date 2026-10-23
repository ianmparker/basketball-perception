# 12-Week Research Curriculum

Three lanes run in parallel: **CV/ML**, **sports vision**, and **basketball**. Each week
has one reading spine, one exercise, and one written artifact. The artifacts accumulate
into the knowledge base.

Use `python research/query.py track <id>` to pull the current reading list for a lane.

Legend: `[book]`, `[paper]`, `[rule]`, `[web]`. Keys are library keys.

---

## Week 1 — Orientation and shared vocabulary

**CV/ML:** `szeliski2022` Ch. 1–2; `prince2023` Ch. 1–2.
**Sports:** `arxiv-2505.03991` (survey, skim the task taxonomy).
**Basketball:** `fiba2026` (rules 1–3); `nba2025rules` (court and equipment);
`research/knowledge/basketball/02-ontology-and-terminology.md`.

**Exercise:** build a 25-term glossary card (term, definition, source, repo field). Mark
every term that has no repo field.
**Write:** a one-page "what the pipeline currently knows" map:
perception → state → event → tactic → value.
**Check:** `python research/query.py gaps`.

---

## Week 2 — Geometry and calibration

**CV/ML:** `szeliski2022` Ch. 2 and 8; `hartley2004` Ch. 2–4 (projective geometry, DLT).
**Sports:** `arxiv-2207.11709` (TVCalib), `arxiv-2404.08401` (PnLCalib),
`arxiv-2404.09807` (calibration benchmark protocol).
**Basketball:** court dimensions table in `basketball/01-rules-and-officiating.md`.

**Exercise:** calibrate a fixed-gym frame with 4, 6, and 8 points. Report median, RMS,
p95, and temporal jitter.
**Write:** update `perception/01-camera-calibration.md` with your numbers.
**Decision:** what p95 threshold should gate a calibration?

---

## Week 3 — Detection

**CV/ML:** `szeliski2022` Ch. 6; `arxiv-1512.03385` (ResNet); `arxiv-1506.02640` (YOLO);
`arxiv-1612.03144` (FPN); `arxiv-1708.02002` (focal loss); `arxiv-2005.12872` (DETR).
**Sports:** `arxiv-2208.08190` (DeepSportRadar).
**Basketball:** shot/rim/player classes in `inference/schema.py`.

**Exercise:** reproduce the ball detector's precision/recall at conf 0.10/0.15/0.25 on a
labeled clip. Plot the curve.
**Write:** a short note on why mAP@50 and operating-point F1 disagree for the ball.
**Decision:** which operating point do we ship, and for which footage regime?

---

## Week 4 — Training and generalization

**CV/ML:** `prince2023` Ch. 6–10; `goodfellow2016` Ch. 7–8; `arxiv-1502.03167`;
`arxiv-1412.6980`; `arxiv-1207.0580`.
**Sports:** `d2l2023` detection chapters (run the notebooks).
**Basketball:** review `docs/COMPREHENSIVE_PLAN.md` and the dataset merge notes.

**Exercise:** train a small detector on 80% of one arena, test on another. Report the
domain gap.
**Write:** a leakage and augmentation checklist for basketball detection.
**Decision:** what augmentation actually helps, and what only memorizes arenas?

---

## Week 5 — Tracking and identity

**CV/ML:** `arxiv-1602.00763` (SORT), `arxiv-1703.07402` (DeepSORT),
`arxiv-2110.06864` (ByteTrack), `arxiv-2206.14651` (BoT-SORT), `arxiv-2203.14360`
(OC-SORT).
**Sports:** `arxiv-2304.05170` (SportsMOT), `arxiv-2406.19655` (Basketball-SORT),
`arxiv-2009.07736` (HOTA).
**Basketball:** `basketball/03-tactics-and-coverage.md` (matchups depend on IDs).

**Exercise:** export MOT rows and compute MOTA/IDF1/HOTA on one clip. Identify the top 10
ID switches and classify their cause (occlusion, cut, similar jersey).
**Write:** an ID-switch taxonomy with examples.
**Decision:** is GMC or ReID the next tracker investment?

---

## Week 6 — Pose and action

**CV/ML:** `arxiv-1812.08008` (OpenPose), `arxiv-2204.12484` (ViTPose),
`arxiv-2303.07399` (RTMPose).
**Sports:** `arxiv-2503.20781` (BASKET, fine-grained skills).
**Basketball:** `krause2000` (skills and drills) — pick 10 skills and define their
observable signal.

**Exercise:** measure pose coverage and keypoint confidence by distance from camera.
**Write:** a table mapping 10 basketball skills to pose/tracking signals and failure modes.
**Decision:** which skills are worth automating first?

---

## Week 7 — Events and temporal localization

**CV/ML:** `arxiv-2404.05392` (T-DEED), `arxiv-2503.00147` (long-range spotting).
**Sports:** `arxiv-2607.21267` (BasketEvent), `arxiv-2608.19646` (PL-NBA),
`arxiv-2105.07404` (MultiSports).
**Basketball:** `basketball/02-ontology-and-terminology.md`.

**Exercise:** map the 10 BasketEvent classes to our taxonomy; list unmapped classes.
**Write:** event ontology v1 as JSON + a label guide with 2 positive/2 negative examples
per class.
**Decision:** which event class gets the next detector?

---

## Week 8 — Representations and foundation models

**CV/ML:** `arxiv-2010.11929` (ViT), `arxiv-2103.00020` (CLIP), `arxiv-2304.07193`
(DINOv2), `arxiv-2111.06377` (MAE), `arxiv-2304.02643` (SAM).
**Sports:** `arxiv-2404.09807` (domain shift in sports calibration).
**Basketball:** team/role labels in `eval/labels/`.

**Exercise:** freeze DINOv2, cluster teams on 3 clips, compare with HSV K-means.
**Write:** a transfer protocol (linear probe → kNN → fine-tune) with metrics.
**Decision:** do we adopt a frozen foundation feature for ReID or team ID?

---

## Week 9 — Basketball analytics I: possessions and value

**CV/ML:** `murphy2022` Ch. 1–4 (probability, calibration).
**Sports:** `arxiv-1408.0777` (EPV), `arxiv-1107.5793` (shot selection).
**Basketball:** `oliver2004` (Four Factors), `bref_glossary`, `nba_stats_glossary`.

**Exercise:** compute POSS, ORtg, DRtg, and the Four Factors for one game; reconcile with
the public box score.
**Write:** a possession-ledger spec with edge cases (and-1, technicals, team rebounds).
**Decision:** what is the v0 xPTS baseline?

---

## Week 10 — Basketball analytics II: shots, lineups, defense

**CV/ML:** `arxiv-1908.05745` (Bayesian shot charts), `arxiv-2006.07513` (hierarchical
shot selection), `arxiv-1609.04849` (shot making from trajectories),
`arxiv-2405.01182` (model-based shot charts).
**Sports:** `arxiv-1405.0231` (defensive skill), `arxiv-1612.05502` (defensive roles),
`arxiv-2002.04148` (intrinsic dimension).
**Basketball:** `goldsberry2019` (spatial history).

**Exercise:** fit a logistic xPTS model with distance/angle/contest; report log loss and
a calibration curve.
**Write:** a lineup evaluation memo: why raw plus-minus is not enough, and what RAPM
assumes.
**Decision:** which value model ships to the coach first?

---

## Week 11 — Tactics and officiating

**CV/ML:** `arxiv-2009.07736` (HOTA, re-read for evaluation discipline).
**Sports:** `arxiv-2304.05170` (SportsMOT, re-read for camera motion).
**Basketball:** `taylor2016`, `price_wolfers2010`, `fiba_documents` (interpretations),
`basketball/01-rules-and-officiating.md`, `basketball/03-tactics-and-coverage.md`.

**Exercise:** label 50 pick-and-roll possessions (drop/hedge/switch/ice/blitz) and
measure agreement.
**Write:** a referee-review product spec: evidence, rule citation, confidence, human
decision.
**Decision:** which calls are objectively checkable and which are out of scope?

---

## Week 12 — Synthesis

**Read:** everything you marked `open` in the notes; skim the `core` entries you missed.
**Exercise:** pick one end-to-end question ("why did our half-court offense underperform
in Q3?") and trace it from pixels to value.
**Write:** a note in `research/knowledge/` with:
- the question;
- the evidence (source keys);
- what the pipeline can and cannot answer today;
- the three experiments that would close the gap.
**Decision:** the next quarter's research agenda.

---

## Standing exercises

- **Every week:** add one verified source and one note update.
- **Every experiment:** hypothesis → baseline → change → scorecard → keep/revert.
- **Every month:** run `validate.py --check-links` and `query.py gaps`; fix or retire
  what is stale.
