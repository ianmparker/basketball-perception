# Research Department

The research arm of the AI Basketball Coach project. Its job is to turn the literature
into **decisions**: a curated bibliography, a distilled knowledge base, and a backlog of
measurable experiments that feed `eval/`.

This is not a reading club. Every entry says what it changes for this repo, every note
maps to code, and every proposed experiment names a metric.

## Map

```
research/
├── taxonomy.json              stable track IDs shared by all artifacts
├── bibliography/
│   ├── library.json           papers, books, rulebooks, datasets (source of truth)
│   ├── schema.json            JSON Schema for an entry
│   └── validate.py            stdlib validator + optional link check
├── knowledge/
│   ├── INDEX.md               generated table of notes
│   ├── index.json             generated machine index
│   ├── build_index.py         parse front matter, audit citations
│   ├── perception/            calibration, ball, MOT, events, representations
│   ├── basketball/            rules, ontology, tactics, shot quality/EPV
│   ├── measurement/           metrics, four factors, experiment design
│   ├── ml/                    training, optimization, regularization
│   └── systems/               footage regimes and runtime
├── ontology/
│   ├── play_taxonomy.json     play types, coverages, and play families
│   └── validate.py            taxonomy contract validator
├── experiments/
│   └── backlog.json           paper-derived hypotheses with owners and metrics
├── curriculum.md              12-week, three-lane reading program
└── query.py                   search, track view, code view, gaps, citations
```

## Quick start

```powershell
# Validate the catalog and check every link (advisory)
python research/bibliography/validate.py --check-links

# Rebuild the knowledge index after editing a note
python research/knowledge/build_index.py

# What do we have?
python research/query.py stats
python research/query.py gaps

# What should I read for the tracker?
python research/query.py track perception.tracking
python research/query.py code tracking/

# Search across titles, authors, topics, and "why"
python research/query.py search occlusion
```

No third-party packages required.

## How an item enters the library

1. **Find** the work. Prefer the publisher, arXiv, an official league/FIBA site, or an
   author's own page.
2. **Verify** the identifier: arXiv ID against the arXiv API, DOI against the publisher,
   URL with a HEAD request. Never trust a search snippet.
3. **Record access.** `open_access`, `preprint`, `publisher`, `purchase`, `subscription`,
   `official_web`, or `dataset_page`.
4. **Record the license** when known. When unknown, say `unknown`; do not guess.
5. **Write `why`.** One sentence on what this changes for *this* repo. "Interesting" is
   not a reason.
6. **Attach tracks and repo paths.** `python research/query.py gaps` will tell you what is
   missing.
7. **Validate.** `python research/bibliography/validate.py`.

### Access policy

- Download only `open_access`, `preprint`, or `official_web` items that permit it.
- `purchase`, `subscription`, and `publisher` items are **citation-only**. Do not fetch,
  mirror, or commit a paywalled PDF.
- Downloaded PDFs go to `research/library/pdf/` (gitignored). The manifest records the
  URL, size, SHA-256, and fetch date so the artifact is reproducible.
- The bibliography is a map, not a redistribution. Do not upload the PDF cache anywhere.

## How a note is written

Use `knowledge/_TEMPLATE.md`. A note is `active` only when it has:

- a **claim** you could disagree with;
- the **model/definition** (equations or a table, not prose only);
- **evidence** with library keys;
- **repo mapping** (which files implement or should implement it);
- **failure modes** (what breaks in practice);
- **open questions** and **proposed experiments**.

Notes are living documents. Update `updated` when the claim or evidence changes; do not
silently rewrite history. If a claim is overturned, say so in the note and keep the old
reference as a rejected hypothesis.

## Cadence

- **Weekly:** one digest. What was read, what changed in the notes, which experiment
  moved, what is blocked.
- **Per experiment:** hypothesis → baseline → change → scorecard → keep/revert.
- **Monthly:** bibliography audit; dead links; retired sources; gap report.
- **Per release:** re-check the claims the release depends on (calibration, metrics,
  ruleset).

## Relationship to the rest of the repo

- `registry/` is for **data** provenance. `research/bibliography/` is for **literature**
  provenance. They share the principle, not the schema.
- `eval/` is the judge. A research note may propose a metric; only `eval/` decides
  whether a change ships.
- `.github/agents/` owns the roles. The research agents write artifacts here; they do not
  edit production thresholds without an eval scorecard.

## Current snapshot

- **100 entries**: 68 papers, 16 books, 2 rulebooks/standards, 12 official/web references,
  2 datasets.
- All 66 arXiv IDs verified on 2026-10-21; two additional papers checked by DOI/URL.
- **22 knowledge notes** across perception, basketball, measurement, systems, and ML.
- The open gaps are visible with `python research/query.py gaps`.

See `curriculum.md` for the reading order and `docs/ARCHITECTURE.md` for how the
research maps to code.
