# Source Registry

The registry is the front door for every external video, dataset, image set, or annotation file. Nothing enters `data/` without a record here.

Records exist for **provenance**, not permission. This project is research-only: licenses are recorded when known but do not gate intake, and nothing in the registry blocks training. The only hard stops are an explicit `DO-NOT-USE` tier or a `removed` record.

## Files

| File | Purpose |
|---|---|
| `sources.schema.json` | JSON Schema for a source record |
| `sources.json` | Current candidate/approved sources |
| `../.github/agents/data-scout.agent.md` | Finds candidates |
| `../.github/agents/licensing-auditor.agent.md` | Records license/provenance when it is cheap to do so (advisory) |
| `../.github/agents/dataset-curator.agent.md` | Downloads assets and versions them |

## Status flow

```
staged → under-review → approved → (ingested)
approved → removed   (takedown)
```

A record is **trainable** when:

- it exists, so the clip can be traced to a source;
- `status` is not `removed`;
- `risk_tier` is not `DO-NOT-USE`.

`license_id`, `license_url`, `provenance_confidence`, and the permission flags are optional metadata. Leave them at their defaults (`unrecorded`, `unknown`, `RESEARCH-ONLY`) rather than guessing. If a batch is ever promoted to a release, fill them in first.

## Risk tiers

| Tier | Meaning |
|---|---|
| `SAFE` | Commercial use, modification, redistribution-as-derivative, and ML training permitted (e.g. Pexels, Pixabay, CC0/PD, CC-BY with attribution). |
| `CAUTION` | Usable only after per-item verification (Wikimedia Commons, Internet Archive, per-project Roboflow, broadcast-derived datasets with a permissive label). |
| `RESEARCH-ONLY` | Research and evaluation use. Do not release the raw footage or redistribute a model trained on it. This is the default tier. |
| `DO-NOT-USE` | The owner has explicitly excluded this source. Never ingest it. |

## Adding a source

1. Create a record with `asset_id`, `name`, `source_url`, `source_platform`.
2. Leave license fields at their defaults unless the information was already at hand.
3. Set `status: "approved"` if you intend to ingest it. `RESEARCH-ONLY` records need no human signature; `SAFE`/`CAUTION` records do, because those tiers claim commercial usability.
4. The **dataset-curator** downloads and registers the asset with DVC.

## Intake commands

```powershell
python registry/validate.py
python scripts/data_ops.py eligible
python scripts/ingest_wikimedia.py --search "basketball game" --limit 5  # discovery only, writes a manifest
python scripts/data_ops.py download --source <asset_id> --manifest files.json --dry-run
python scripts/data_ops.py download --source <asset_id> --manifest files.json
```

`data_ops.py` requires a registry record, refuses `DO-NOT-USE`/`removed` sources,
verifies SHA-256 checksums, and writes an `ingest_manifest.json` beside the
downloaded files.

## Current snapshot

24 candidates: 4 `SAFE`, 8 `CAUTION`, 11 `RESEARCH-ONLY`, 1 `DO-NOT-USE`.

`roboflow-basketball-bs0zc` is a 9,445-image, 5-class (`ball`, `person`, `made`,
`rim`, `shoot`) object-detection set labeled CC BY 4.0. It is `CAUTION` because
the uploader account ("test datset") has no description and the images look
broadcast-derived, so the license tag is self-asserted rather than a chain of
title. It contains no video, track IDs, or events.

The highest-value research-only sources are SportsMOT (tracking), MultiSports (action localization), BasketEvent (shot/rebound/block events), BASKET (scale/skill), PL-NBA (possession events), APIDIS/Spiroudome (multi-view calibration and tracking), and DeepSportRadar (ball/pose/calibration).
