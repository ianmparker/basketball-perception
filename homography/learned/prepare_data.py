"""
Data Preparation for Court Keypoint Heatmap Training

Loads the reloc2 dataset (COCO keypoint format) downloaded from Roboflow
and creates a PyTorch Dataset that yields:
  - image: (3, H, W) normalised RGB tensor
  - heatmaps: (N_kp, H/4, W/4) Gaussian ground-truth heatmaps
  - mask: (N_kp,) bool — which keypoints are visible/labelled

reloc2 dataset structure (after download):
  data/court_keypoints/
    train/   _annotations.coco.json  + images
    valid/   _annotations.coco.json  + images
    test/    _annotations.coco.json  + images

COCO keypoint format: each annotation has a 'keypoints' list of
  [x1, y1, v1,  x2, y2, v2, ...] where v=0 (absent), 1 (occluded), 2 (visible)
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# Target spatial size for training
INPUT_H, INPUT_W = 384, 640
HEATMAP_H, HEATMAP_W = INPUT_H // 4, INPUT_W // 4  # stride-4 feature map
HEATMAP_SIGMA = 4   # Gaussian width in heatmap pixels


def render_gaussian_heatmap(
    h: int, w: int, cx: float, cy: float, sigma: float = HEATMAP_SIGMA
) -> np.ndarray:
    """Render a 2D Gaussian centred at (cx, cy) on a (h, w) heatmap."""
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


class CourtKeypointDataset(Dataset):
    """
    PyTorch Dataset wrapping the reloc2 COCO keypoint annotations.

    Args:
        split_dir:      Path to train/ valid/ or test/ folder
        num_keypoints:  Number of court keypoints (default 18)
        augment:        Apply photometric augmentation (training only)
    """

    def __init__(
        self,
        split_dir: str,
        num_keypoints: int = 18,
        augment: bool = True,
    ):
        self.split_dir = Path(split_dir)
        self.num_keypoints = num_keypoints
        self.augment = augment

        ann_file = self.split_dir / "_annotations.coco.json"
        if not ann_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_file}\n"
                "Run: python scripts/download_court_keypoints.py --api-key YOUR_KEY"
            )

        with open(ann_file) as f:
            coco = json.load(f)

        # Build id → file-name map
        id_to_fname = {img["id"]: img["file_name"] for img in coco["images"]}

        # One annotation per image (COCO allows multiple but reloc2 has one court per image)
        self.samples: List[Dict] = []
        for ann in coco["annotations"]:
            kps = np.array(ann["keypoints"], dtype=np.float32).reshape(-1, 3)
            self.samples.append({
                "image_path": str(self.split_dir / id_to_fname[ann["image_id"]]),
                "keypoints_xy": kps[:, :2],   # (N_kp, 2) in original pixel coords
                "visibility":   kps[:, 2],    # (N_kp,)  0=absent 1=occluded 2=visible
            })

        print(f"Loaded {len(self.samples)} samples from {self.split_dir}")
        self.transform = self._build_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]

        img = cv2.imread(s["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        kps_xy = s["keypoints_xy"].copy()          # (N_kp, 2)
        vis    = s["visibility"].copy()             # (N_kp,)

        # --- Resize image and rescale keypoints ---
        img = cv2.resize(img, (INPUT_W, INPUT_H))
        scale_x = INPUT_W / orig_w
        scale_y = INPUT_H / orig_h
        kps_xy[:, 0] *= scale_x
        kps_xy[:, 1] *= scale_y

        # --- Augmentation (train: geometric + photometric via albumentations) ---
        # Geometric transforms (HorizontalFlip, Perspective) are handled by
        # albumentations with KeypointParams so keypoint coords are updated
        # automatically. OOB keypoints after perspective warp are caught by
        # the bounds check below when building heatmaps.
        if self.augment:
            vis_indices = [k for k in range(self.num_keypoints) if vis[k] > 0]
            kp_list = [(float(kps_xy[k, 0]), float(kps_xy[k, 1])) for k in vis_indices]
            aug = self.transform(image=img, keypoints=kp_list)
            img = aug["image"]
            for i, k in enumerate(vis_indices):
                kps_xy[k, 0], kps_xy[k, 1] = aug["keypoints"][i]
        else:
            img = self._to_tensor(img)

        # --- Build ground-truth heatmaps ---
        heatmaps = np.zeros((self.num_keypoints, HEATMAP_H, HEATMAP_W), dtype=np.float32)
        mask = np.zeros(self.num_keypoints, dtype=bool)

        for k in range(self.num_keypoints):
            if k >= len(vis) or vis[k] == 0:
                continue   # keypoint absent
            # Roboflow exports "deleted" keypoints with vis=1 at out-of-bounds coords.
            # Reject anything outside the resized image boundaries.
            x_k, y_k = float(kps_xy[k, 0]), float(kps_xy[k, 1])
            if x_k < 0 or x_k > INPUT_W or y_k < 0 or y_k > INPUT_H:
                continue   # OOB = deleted annotation — skip
            # Scale keypoint to heatmap resolution (stride 4)
            hx = x_k / INPUT_W * HEATMAP_W
            hy = y_k / INPUT_H * HEATMAP_H
            heatmaps[k] = render_gaussian_heatmap(HEATMAP_H, HEATMAP_W, hx, hy)
            mask[k] = True

        return {
            "image":     img,                                   # (3, H, W) float tensor
            "heatmaps":  torch.from_numpy(heatmaps),           # (N_kp, H/4, W/4)
            "mask":      torch.from_numpy(mask),               # (N_kp,) bool
            "kps_xy":    torch.from_numpy(kps_xy),             # (N_kp, 2) original px
            "image_path": s["image_path"],
        }

    # ------------------------------------------------------------------

    def _build_transform(self) -> A.Compose:
        """Geometric + photometric augmentation with albumentations keypoint tracking."""
        return A.Compose([
            # Geometric — albumentations updates keypoint coords automatically
            A.HorizontalFlip(p=0.5),
            A.Perspective(scale=(0.02, 0.06), p=0.4),
            # Photometric
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def _to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """Normalise and convert to tensor (validation path)."""
        pipeline = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
        return pipeline(image=img)["image"]


# ------------------------------------------------------------------
# Backward-compatible shim — old code imported HomographyDataset
# ------------------------------------------------------------------

class HomographyDataset(CourtKeypointDataset):
    """Alias for backward compatibility."""
    def __init__(self, data_file: str, transform=None, augment: bool = True):
        # data_file was a JSON path; assume split_dir is its parent
        super().__init__(split_dir=str(Path(data_file).parent), augment=augment)

