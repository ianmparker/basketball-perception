# Knowledge Base

The distilled "brain": one note per question, not one note per paper. Notes are written
for the next agent and the next decision.

## Structure

```
knowledge/
├── perception/   calibration, ball, MOT, events, representation
├── basketball/   rules, ontology, tactics, shot quality/EPV
├── measurement/  metrics, four factors, experiment design
├── systems/      footage regimes, runtime
├── index.json    generated; do not edit
└── INDEX.md      generated; do not edit
```

The folder is the coarse layer; the front-matter `track` is the precise taxonomy ID.

## Note contract

A note is `active` when it has:

1. a **claim** you could disagree with;
2. the **model/definition** (equations, tables, or a decision rule);
3. **evidence** as library keys (not bare names);
4. **repo mapping** to real files;
5. **failure modes** observed or expected;
6. **open questions**;
7. **proposed experiments** with IDs that appear in `experiments/backlog.json`.

If any of 1–5 is missing, the note is a `stub`. Do not mark a summary as `active`.

## Status and confidence

| Field | Values | Meaning |
|---|---|---|
| `status` | `stub` | outline only; no evidence |
| | `active` | usable; claims and evidence present |
| | `mature` | reproduced on our data; stable |
| | `retired` | superseded; kept for history |
| `confidence` | `low` | single source, untested |
| | `medium` | multiple sources or partial reproduction |
| | `high` | reproduced and measured on our data |

`confidence: high` is reserved for claims with a scorecard. Most notes should be
`medium`.

## Front matter

```yaml
---
id: perception.calibration.homography     # stable; never reuse for a different idea
title: Camera calibration and court homography
layer: perception                          # folder-level layer
track: perception.calibration              # must exist in research/taxonomy.json
status: active
confidence: high
updated: 2026-10-21
sources: [szeliski2022, hartley2004, arxiv-2207.11709]
code: [homography/, scripts/calibrate_from_points.py]
prereqs: []
tags: [homography, DLT, RANSAC]
---
```

- `sources` must resolve in `bibliography/library.json`; the index build fails otherwise.
- `code` paths should exist; the query tool uses them for `query.py code <path>`.
- `prereqs` are other note IDs, used to derive a reading order.
- `updated` changes when the claim or evidence changes, not on a typo fix.

## Editing workflow

```powershell
# 1. Write or edit the note
# 2. Rebuild the index (also audits citations and tracks)
python research/knowledge/build_index.py

# 3. Check coverage
python research/query.py gaps
python research/query.py track perception.calibration

# 4. CI check
python research/knowledge/build_index.py --check
```

## What not to do

- Do not copy a paper abstract into a note. Write the claim and the consequence.
- Do not cite a source you did not open. Use the library key and verify the link.
- Do not add a metric without saying how it is computed and on what split.
- Do not rewrite a claim silently. If it changes, note the change in the digest.
- Do not put a paywalled PDF in the repo.
