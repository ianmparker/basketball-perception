---
id: perception.representation.foundation-models
title: Representation learning and foundation models for basketball
layer: perception
track: perception.representation
status: active
confidence: medium
updated: 2026-10-21
sources: [arxiv-2103.00020, arxiv-2304.07193, arxiv-2111.06377, arxiv-2304.02643, arxiv-2010.11929]
code: [inference/, tracking/, annotation/]
prereqs: [ml.foundations]
tags: [CLIP, DINOv2, MAE, SAM, transfer, ReID]
---

# Representation learning and foundation models for basketball

**Claim.** With ~8k labeled images and no player-level identity labels, the highest-value
use of modern foundation models is **frozen features plus a small head**, not end-to-end
fine-tuning. Fine-tuning a large ViT on this dataset will memorize arenas and jerseys.

## The inductive-bias trade

- A CNN has locality and translation equivariance built in. It learns well from small
  datasets and is the right default for detection at this scale.
- A ViT has global attention from layer one. It needs more data or strong pretraining,
  but its features transfer better and its attention is interpretable.
- Hybrids (Swin, ConvNeXt) try to get both. ConvNeXt is a strong reminder that a modern
  CNN can match a transformer on vision benchmarks.

For this repo: keep YOLO for detection, and use foundation features for the tasks where
labels are scarce — team assignment, ReID, retrieval, annotation triage.

## What each model is actually good for

| Model | Objective | Best use here | Do not use it for |
|---|---|---|---|
| CLIP | image–text contrastive | zero-shot prompts, semantic search, team/action prototypes | fine-grained jersey numbers, small ball |
| DINOv2 | self-distillation, no labels | frozen dense features, ReID, clustering, retrieval | real-time detection on CPU |
| MAE | masked reconstruction | pretraining when we have unlabeled footage | few-shot classification without a head |
| SAM | promptable segmentation | mask bootstrapping, player/referee silhouettes, annotation | identity, action, or cheap real-time |
| ViT | supervised/self-supervised | backbone for pose or event models | small-data training from scratch |

## Transfer protocol

1. **Linear probe** first: freeze the backbone, train a logistic regression on pooled
   features. This tells you whether the representation already separates the classes.
2. **kNN / retrieval** second: if nearest neighbors are wrong, a linear head will not
   save you.
3. **Fine-tune** only after the probe shows signal, and only the last block(s).
4. **Report calibration**, not just accuracy. A confident wrong team assignment is worse
   than an abstention.

For clustering (team ID), evaluate with adjusted Rand index or cluster purity against
labeled team assignments, not with silhouette alone. Silhouette can be high on the wrong
two clusters.

## Team identification: the concrete case

Current default: K-means on HSV color, zero-shot, no prompts. Alternative: CLIP ViT-B/32
with `--openai-clip`.

What the literature suggests:

- Color histograms are strong when jerseys are distinct and lighting is stable.
- CLIP features encode semantics and context; they can separate "white jersey" from
  "dark jersey" but may also latch onto court, crowd, or skin tone. Prompt wording
  matters, and the model was not trained on basketball jerseys.
- DINOv2 features are likely stronger for instance discrimination (same player vs
  different player) because the objective is self-distillation, not text matching.

The honest test is on labeled `team_assignments` in `eval/labels/`: per-frame accuracy,
confusion matrix, and accuracy by lighting/jersey combination. The repo already reports
team accuracy in `eval/metrics/detection.py`.

## Domain shift is the real adversary

Broadcast NBA and a high-school gym differ in: camera height and lens, lighting color
temperature, jersey design, court paint, crowd density, graphics overlays, and motion
blur. A feature that separates teams in one arena can separate *arenas* in another.

Mitigations, in order of cost:

1. Evaluate per arena / footage type (DeepSportRadar's per-arena split is the model).
2. Color-normalize (gray-world or white balance) before clustering.
3. Use frozen features with a per-game adaptation step (fit the two team centroids on
   the first N frames of that game). This is unsupervised domain adaptation with a very
   small budget.
4. Fine-tune on labeled target footage only when you have enough of it.

## Repo mapping

- `inference/` — `--openai-clip` team path; CLIP feature extraction.
- `tracking/` — no ReID embedding yet; DINOv2 is the natural first candidate.
- `annotation/` — SAM-assisted masks could cut labeling time.
- `eval/metrics/detection.py` — team accuracy; extend with per-game breakdown.

## Failure modes

- **Background leakage.** Features can identify the arena, not the team. Test on a
  held-out arena.
- **Prompt sensitivity.** "gold jersey" vs "yellow jersey" changes CLIP output. Freeze
  the prompt set and report it.
- **Frozen features on tiny crops.** A 40×80 player crop at broadcast resolution may be
  too small for a 14×14 patch ViT; upscale or use a multi-scale crop.
- **Assuming SSL features are label-free.** They are label-free to train, not to
  evaluate; you still need labeled pairs for retrieval metrics.
- **Fine-tuning on the test arena.** Split by game/arena, never by frame.

## Open questions

1. Do DINOv2 features beat HSV K-means on team assignment when both teams wear similar
   colors?
2. Can DINOv2 embeddings support ReID across a broadcast cut without any fine-tuning?
3. What crop resolution is the break-even point for ViT features on players?
4. Does SAM-assisted annotation actually reduce label time, or does it just move the
   effort to mask correction?

## Proposed experiments

- **EX-040** Freeze DINOv2, cluster team assignments on 3 labeled clips; compare with
  HSV K-means and CLIP.
- **EX-041** Build a retrieval benchmark: same player / different player pairs across
  frames; report mAP@10.
- **EX-042** Linear-probe player vs referee vs bench on frozen features; report per-arena
  accuracy.
- **EX-043** Measure annotation time for 50 SAM-assisted masks vs 50 manual masks.
