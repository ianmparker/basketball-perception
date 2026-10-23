# Data policy

This repository is a **code and evaluation** release. It does not distribute the
footage used to develop or train the models.

## What is included

- Source code for detection, tracking, pose, calibration, analytics, rendering, and
  evaluation.
- Aggregate scorecards and benchmark notes in `eval/`.
- Labels for footage from Wikimedia Commons (see `eval/labels/wikimedia_*.json`).
- Documentation of the data sources, licenses, and known failures.

## What is not included

- Broadcast video, extracted frames, screenshots of footage, or event clips.
- Trained task-specific weights (ball, player, court keypoint models).
- Raw label JSON, per-clip manifests, or frames derived from research-only broadcast
  footage.
- Credentials, bucket names, or local filesystem paths from the development
  environment.

## Bring your own video

To run the pipeline, supply a clip you have the right to use. The demo and tests do
not require a specific source. If you publish results, record the source, license,
and retrieval date — the same discipline used in `registry/`.

## Model weights

The public backbones (`yolov8n.pt`, `yolov8x-pose.pt`) can be downloaded with
`scripts/download_models.py`. The task-specific weights were trained on research-only
footage and are not redistributed. `models/README.md` documents the expected paths.

## Scorecards and labels

Aggregate metrics are facts about the system and are included. Labels derived from
Wikimedia Commons footage are included because that footage is openly licensed.
Labels derived from broadcast footage are not included; the aggregate scorecards
remain so the results can be audited without redistributing the source.

## Reporting a problem

If you believe content in this repository infringes your rights, contact
**ianmparker@outlook.com** with the file path and a description of the issue. Content
will be removed or replaced while the claim is reviewed.
