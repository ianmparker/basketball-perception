---
id: perception.pose.actions
title: Pose estimation and action recognition
layer: perception
track: perception.pose
status: active
confidence: medium
updated: 2026-10-21
sources: [arxiv-1812.08008, arxiv-2204.12484, arxiv-2303.07399, arxiv-2503.20781]
code: [inference/pipeline.py, scripts/check_pose.py, events/detect_shots_pose.py]
prereqs: [perception.detection.objects]
tags: [pose, keypoints, action recognition, skills, occlusion]
---

# Pose estimation and action recognition

**Claim.** Broadcast pose is a **low-resolution, occluded, multi-person** problem. The
current YOLOv8x-pose gives useful orientation and jump signals for the player nearest the
camera, but it is not a motion-capture system. Use pose as a **feature**, not as ground
truth.

## What pose gives you

COCO-17 keypoints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
From them:

- **orientation** (shoulder/hip vector) — offense vs defense, facing the basket;
- **jump state** (ankle/knee above a baseline) — rebound, block, shot;
- **release height** (wrist above shoulder + ball proximity) — shot detection;
- **contest** (defender's wrist near the shooter's release) — shot quality;
- **screen legality** (feet planted, body stationary) — screen detection;
- **fall/contact** (hip/knee trajectory) — foul evidence, not a foul call.

## The model families

| Model | Type | Use here |
|---|---|---|
| OpenPose | bottom-up, part affinity fields | reference; multi-person without a detector |
| HRNet | top-down, high-resolution | accuracy baseline; heavy |
| ViTPose | top-down ViT | strongest accuracy; compute-hungry |
| RTMPose | top-down, real-time | best candidate for edge/CPU |

The repo uses YOLOv8x-pose, a top-down detector + keypoint head. It is a reasonable
default. The first upgrade to test is RTMPose for speed, ViTPose for accuracy.

## The resolution problem

At full-court framing a player is ~100 px tall; the wrist is ~5 px. Keypoint error of
3–5 px is a large fraction of the signal. Consequences:

- do not compute joint angles from broadcast far-court players and call them precise;
- gate pose-derived events on player bounding-box height (e.g. > 80 px);
- report keypoint confidence and drop low-confidence joints rather than imputing them;
- prefer **relative** features (wrist above shoulder) over absolute angles.

## Action recognition vs pose

Pose is a frame-level signal. An action is a temporal pattern. Three approaches:

1. **Rule-based on pose sequences** — a shot is wrist above shoulder + upward velocity +
   ball release; a screen is a stationary lower body + a nearby cutter. Interpretable,
   brittle.
2. **Skeleton-based action recognition** — GCN/transformer over keypoint sequences.
   Robust to appearance, needs labeled sequences.
3. **Video action recognition** — 3D CNN / video transformer over RGB clips. Highest
   ceiling, most data.

BASKET (4,477 hours, 20 fine-grained skills) is the long-term benchmark. It is not a
near-term training set for this repo. Start with rules on pose for the events we already
detect.

## Repo mapping

- `inference/pipeline.py` — YOLOv8x-pose integration, keypoint confidence threshold 0.30.
- `scripts/check_pose.py` — pose sanity check.
- `events/detect_shots_pose.py` — pose-aware shot gate.
- `events/detect_screens.py` — uses proximity/stance, not full pose yet.

## Failure modes

- **Pose on a 40 px player.** The keypoints are noise; the rule fires on noise.
- **Missing keypoints treated as zero.** Impute or drop; never silently use 0,0.
- **Camera angle changes the apparent angle.** A shot form angle from a sideline camera is
  not comparable to a baseline camera.
- **Double-counting people.** Pose and detection must share the same track ID.
- **Action labels without temporal boundaries.** "Shot" is an instant; "screen" is an
  interval. Label accordingly.

## Open questions

1. What player bounding-box height is the break-even for usable wrist keypoints?
2. Does RTMPose improve keypoint accuracy enough at broadcast resolution to change an
   event decision?
3. Which events can be detected from pose alone, and which need the ball?
4. Can we learn a basketball-specific pose prior from the 134-image pose dataset, or is
   it too small?

## Proposed experiments

- **EX-140** Measure keypoint confidence vs player box height on 3 clips; find the gate.
- **EX-141** Swap YOLOv8x-pose for RTMPose on one clip; compare keypoint accuracy and
  speed.
- **EX-142** Add a pose-based release detector and compare with the ball-trajectory shot
  detector.
