---
id: ml.foundations.training
title: ML foundations for a small, messy, domain-shifted dataset
layer: ml
track: ml.foundations
status: active
confidence: high
updated: 2026-10-21
sources: [prince2023, goodfellow2016, arxiv-1512.03385, arxiv-1706.03762, arxiv-1412.6980, arxiv-1502.03167]
code: [models/, src/]
prereqs: []
tags: [training, optimization, regularization, architectures, reproducibility]
---

# ML foundations for a small, messy, domain-shifted dataset

**Claim.** With ~8k images, one sport, and a large domain shift between arenas, the
binding constraints are **data quality, evaluation discipline, and regularization** — not
architecture novelty. Every hour spent on a new backbone before the split is clean is an
hour wasted.

## The bias–variance view

```
E[error] = bias² + variance + irreducible noise
```

- A large model on 8k images has low bias and high variance; it memorizes arenas.
- A small model has high bias and underfits the ball.
- The fix for variance is data, augmentation, regularization, and early stopping — not a
  bigger model.
- The fix for bias is capacity and better features — but only after the variance side is
  under control.

## Optimization choices that matter

| Choice | Default here | Why |
|---|---|---|
| Optimizer | SGD momentum or AdamW | Adam converges fast; SGD can generalize better |
| LR schedule | warmup + cosine | stable for small batches |
| Weight decay | 0.0005 (AdamW) | regularization, not just optimization |
| Batch norm | yes for CNNs | stabilizes training; interacts with batch size |
| Mixed precision | FP16 | speed/memory; watch for NaNs |
| EMA of weights | optional | cheap stability gain |
| Seed + deterministic | required | reproducibility |

Do not tune the optimizer before the data pipeline and the eval split are correct. A
learning-rate sweep on a leaky split is wasted compute.

## Architectures, briefly

- **CNN**: locality and translation equivariance; strong inductive bias for small data.
  ResNet is the reference; ConvNeXt is the modern CNN.
- **Transformer**: global attention, weak inductive bias, needs data or pretraining. ViT
  is the reference; Swin adds hierarchy.
- **Hybrid**: CNN stem + transformer blocks; often the best accuracy/compute trade-off.

For detection, the YOLO family is the pragmatic choice. For pose, RTMPose/ViTPose. For
features, DINOv2. Do not fine-tune a ViT from scratch on 8k images.

## Losses

- **Cross-entropy** for classification; **focal loss** for extreme class imbalance (the
  ball).
- **L1/L2** for regression; **IoU/GIoU/CIoU** for boxes.
- **Contrastive/triplet** for ReID; needs careful mining.
- **Reprojection** for geometry; never MSE on homography entries.

A loss is a statement about what errors matter. Write it down before training.

## Transfer and domain shift

1. Pretrain on a large corpus (ImageNet, COCO, CLIP/DINOv2).
2. Fine-tune on basketball; freeze early layers if data is small.
3. Evaluate on a **held-out arena** to measure domain shift.
4. If the gap is large, prefer frozen features + a small head over full fine-tuning.

The repo's 7,860-image dataset is enough to fine-tune a detector, not enough to learn
basketball from scratch.

## Reproducibility

- Pin the config, seed, and git SHA in every run (`ClipResult.metadata` already does this
  for inference).
- Log the dataset hash and split definition.
- Store the scorecard, not just the checkpoint.
- A result that cannot be reproduced is a rumor.

## Repo mapping

- `models/` — training scripts, runs, checkpoints.
- `src/` — pipeline source.
- `docs/MATHEMATICAL_FOUNDATIONS.md` — the math reference.
- `eval/` — the judge; see `measurement.experiment.noise`.

## Failure modes

- **Tuning on the test set.** The most common and most fatal.
- **Batch-size/normalization interactions.** Changing batch size changes BN statistics and
  the result.
- **Class imbalance ignored.** The ball is rare and small; a global loss will ignore it.
- **Augmentation that changes semantics.** Horizontal flip is fine for detection, not for
  left/right-handed shot form or court orientation.
- **Checkpoint without config.** Unreproducible.

## Open questions

1. Does AdamW + cosine beat SGD momentum on the ball detector at this data scale?
2. How much does EMA improve the detector's stability across seeds?
3. What is the seed variance of mAP@50 on this dataset? If it is >1 point, most small
   "improvements" are noise.
4. Does mixed precision change the small-object recall?

## Proposed experiments

- **EX-150** Run the same detector with 3 seeds; report the seed variance of mAP@50 and
  ball recall.
- **EX-151** Compare AdamW vs SGD momentum on the grouped benchmark.
- **EX-152** Ablate horizontal flip and color jitter; measure per-arena accuracy.
