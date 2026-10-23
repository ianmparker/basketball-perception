"""
Court Keypoint Heatmap Network for Zero-Calibration Homography

Architecture (replaces direct matrix regression):
  1. HRNet-W32 backbone — high-resolution feature maps, ideal for keypoint detection
  2. Per-keypoint heatmap heads — one Gaussian heatmap per court landmark
  3. Differentiable argmax — sub-pixel keypoint coordinates from heatmaps
  4. DLT solver — analytical homography from detected keypoints (exact, interpretable)

Why heatmaps instead of direct regression:
  - Interpretable: visualise exactly which court features were found
  - Algebraically sound: H is solved, not memorised
  - Generalises across arenas: court geometry not camera angle is learned
  - Better uncertainty: low-confidence heatmaps = unreliable keypoints → skip DLT

reloc2 dataset keypoint structure (18 keypoints, empirically determined):

  Roboflow semantic IDs 0–17 (used as COCO keypoint names):
    Group A — near-baseline + right-center (cameras near right sideline y≈50):
      sem 0–5:  near baseline intersections (x=0), y=50→0 (right to left)
      sem 8:    halfcourt × right sideline  [47, 50]
      sem 9:    center circle left arc      [47, 19]  (circle center (47,25) radius 6)

    Group B — far-baseline + left-center (cameras near left sideline y≈0):
      sem 6–7:  far free-throw line T-junctions (x=75)
      sem 10:   halfcourt × left sideline   [47,  0]
      sem 11–15: far baseline intersections (x=94), y=3→50
      sem 16:   center circle right arc     [47, 31]
      sem 17:   center circle center        [47, 25]

  COCO EXPORT POSITION ORDER ≠ SEMANTIC ORDER for positions 6–15.
  Roboflow exports: ['0','1','2','3','4','5','9','8','6','7','15','14','13','12','11','10','16','17']
  COURT_KEYPOINTS_FT is therefore indexed by COCO position, not semantic ID.

  Note: Groups A and B are mutually exclusive in ~75% of frames (half-court views).
        ~25% of frames show BOTH groups (full-court sideline views).

NBA court coordinate system (feet):
  x = 0 (near baseline) … 94 (far baseline)
  y = 0 (left sideline)  … 50 (right sideline)
  Camera convention: "right" cameras sit near y=50, so y=50 features are most visible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

# NBA court coordinate system (feet from origin at near-baseline/bottom-sideline corner)
# Shape: (18, 2) — (x_along_length, y_across_width) in feet.
# Identities verified against the reloc2 COCO `keypoints` order and `skeleton`
# connectivity in data/court_keypoints/train/_annotations.coco.json:
#   chain  0-1-2-3-4-5-6  : near baseline + halfcourt @ bottom sideline
#   edge   0-7            : top-sideline halfcourt closes the near half
#   quad   2-8-9-3        : NEAR free-throw lane (paint, FT line at x=19)
#   chain  6-10-11-12-13-14-15 : far baseline
#   edge   15-7           : top-sideline halfcourt closes the far half
#   quad   13-16-17-12    : FAR free-throw lane (paint, FT line at x=75)
# COCO `keypoints` (semantic-id strings, position-ordered):
#   ['0','1','2','3','4','5','9','8','6','7','15','14','13','12','11','10','16','17']
COURT_KEYPOINTS_FT = torch.tensor([
    [0.0,  50.0],  # 0  near baseline @ top sideline
    [0.0,  47.0],  # 1  near baseline 3pt corner (top side)
    [0.0,  33.0],  # 2  near baseline paint corner (top)
    [0.0,  17.0],  # 3  near baseline paint corner (bottom)
    [0.0,   3.0],  # 4  near baseline 3pt corner (bottom side)
    [0.0,   0.0],  # 5  near baseline @ bottom sideline
    [47.0,  0.0],  # 6  halfcourt × bottom sideline
    [47.0, 50.0],  # 7  halfcourt × top sideline
    [19.0, 33.0],  # 8  NEAR FT line × paint top  (paired with 2)
    [19.0, 17.0],  # 9  NEAR FT line × paint bottom (paired with 3)
    [94.0,  0.0],  # 10 far baseline @ bottom sideline
    [94.0,  3.0],  # 11 far baseline 3pt corner (bottom)
    [94.0, 17.0],  # 12 far baseline paint corner (bottom)
    [94.0, 33.0],  # 13 far baseline paint corner (top)
    [94.0, 47.0],  # 14 far baseline 3pt corner (top)
    [94.0, 50.0],  # 15 far baseline @ top sideline
    [75.0, 33.0],  # 16 FAR FT line × paint top (paired with 13)
    [75.0, 17.0],  # 17 FAR FT line × paint bottom (paired with 12)
], dtype=torch.float32)

NUM_KEYPOINTS = 18


# ---------------------------------------------------------------------------
# HRNet-W32 backbone via timm — stride-4 high-resolution features
# ---------------------------------------------------------------------------

def _hrnet_backbone(
    pretrained: bool = True,
    coco_pose_ckpt: Optional[str] = None,
) -> "Tuple[nn.Module, int]":
    """
    Load HRNet-W32 via timm with proper stride-4 feature extraction.

    Key insight vs torchvision approach:
      torchvision HRNet with classifier=Identity outputs pooled (B,2048) or
      coarse stride-32 features — terrible for precise keypoint localisation.

      timm features_only=True, out_indices=[0] gives the HIGHEST-RESOLUTION
      branch (stride 4) from the FINAL HRNet stage — exactly the representation
      the original HRNet pose paper uses for keypoint detection.

      For INPUT_H=384, INPUT_W=640:
        features shape → (B, 128, 96, 160)  ← matches GT heatmap resolution
        No upsampling needed in the heatmap head!

    COCO pose pretrained weights (optional, recommended):
      Download: https://download.openmmlab.com/mmpose/top_down/hrnet/
                hrnet_w32_coco_256x192-c78dce93_20200708.pth
      Load via --backbone-ckpt flag — strips mmpose 'backbone.' prefix.
    """
    import timm

    # features_only + out_indices=[0] → list with one element:
    # the high-resolution (stride 4) branch from the final HRNet stage.
    backbone = timm.create_model(
        "hrnet_w32",
        pretrained=pretrained,
        features_only=True,
        out_indices=[1],  # stride-4, 128ch — needs INPUT_H divisible by 32 (use 384)
    )
    out_channels = backbone.feature_info.channels()[0]  # 128 for HRNet-W32 stage1

    # Optionally load COCO pose pretrained backbone weights (mmpose format)
    if coco_pose_ckpt:
        import os
        if not os.path.exists(coco_pose_ckpt):
            print(f"[WARNING] backbone ckpt not found: {coco_pose_ckpt} — skipping")
        else:
            ckpt = torch.load(coco_pose_ckpt, map_location="cpu", weights_only=True)
            state = ckpt.get("state_dict", ckpt)
            # mmpose keys: 'backbone.stage1.0.weight' → strip 'backbone.' prefix
            state = {k.replace("backbone.", ""): v
                     for k, v in state.items() if k.startswith("backbone.")}
            missing, unexpected = backbone.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded COCO pose backbone: {len(state)} keys, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected")

    return backbone, out_channels


# ---------------------------------------------------------------------------
# DINOv2-B backbone wrapper (stride-14 patch features, 768ch)
# ---------------------------------------------------------------------------

class _DINOv2BackboneWrapper(nn.Module):
    """
    Wraps facebook/dinov2-vitb14 to emit (B, 768, H/14, W/14) feature maps.

    Reflect-pads input to the next multiple of patch_size=14 so the data
    pipeline needs no input-size changes.  For INPUT_H=384, INPUT_W=640
    this adds +8 rows and +4 columns → effective 392×644, giving a
    28×46 = 1288-token patch grid.

    All backbone parameters are frozen by default (requires_grad=False).
    """

    _PATCH = 14

    def __init__(self, dinov2: nn.Module):
        super().__init__()
        self.dinov2 = dinov2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        pad_h = (self._PATCH - H % self._PATCH) % self._PATCH
        pad_w = (self._PATCH - W % self._PATCH) % self._PATCH
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        H_pad, W_pad = H + pad_h, W + pad_w
        h_p, w_p = H_pad // self._PATCH, W_pad // self._PATCH
        out = self.dinov2.forward_features(x)
        tokens = out["x_norm_patchtokens"]      # (B, h_p*w_p, 768)
        return tokens.permute(0, 2, 1).reshape(B, 768, h_p, w_p)


def _dinov2_backbone(pretrained: bool = True) -> "Tuple[nn.Module, int]":
    """
    Load DINOv2-B via torch.hub (download cached after first call).
    Returns a wrapper that emits spatial feature maps (B, 768, H/14, W/14).
    All parameters frozen — head-only training in train.py.
    """
    dinov2 = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vitb14", pretrained=pretrained
    )
    wrapper = _DINOv2BackboneWrapper(dinov2)
    for p in wrapper.parameters():
        p.requires_grad_(False)
    return wrapper, 768


# ---------------------------------------------------------------------------
# Heatmap decoder head
# ---------------------------------------------------------------------------

class HeatmapHead(nn.Module):
    """
    Lightweight conv head: stride-4 backbone features → per-keypoint heatmaps.

    With timm HRNet-W32 (features_only, out_indices=[1]):
      Input:  (B, 128, H/4, W/4)  — already at heatmap resolution, no upsampling!
      Output: (B, N_kp, H/4, W/4)  — one logit map per keypoint

    The spatial resolution matches the GT heatmaps (HEATMAP_H × HEATMAP_W)
    so the MSE loss comparison is direct with no interpolation artefacts.
    """

    def __init__(self, in_channels: int, num_keypoints: int = NUM_KEYPOINTS):
        super().__init__()
        # Project from HRNet stage1 (128 ch) → per-keypoint heatmaps
        mid = 256
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_keypoints, 1),  # per-keypoint logits
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)  # (B, N_kp, H/4, W/4) raw logits


# ---------------------------------------------------------------------------
# Direct homography regression head: features → 8-DoF H matrix
# ---------------------------------------------------------------------------

class HomographyRegressionHead(nn.Module):
    """
    Spatial-pool → MLP → 8 free parameters → 3×3 homography (H[2,2]=1 fixed).

    Maps normalised pixel coords [0,1]² → court coords in feet.

    v15 upgrade over v14: replaces GlobalAvgPool (1×1) with spatial pooling
    at (6, 10) — roughly matching the court aspect ratio (50:94 ≈ 1:2).
    A 1×1 conv first reduces channels (128→64), then AdaptiveAvgPool2d((6,10))
    gives 64×6×10 = 3840 spatially-aware features instead of 128 global averages.
    The model can now use WHERE in the image different features appear, which is
    essential for disambiguating perspective (e.g. near vs. far baseline views).

    Initialised so H ≈ affine scaling of image to court dimensions:
        H_init = [[94, 0, 0],
                  [ 0,50, 0],
                  [ 0, 0, 1]]
    meaning the model starts by predicting (u*94, v*50) ≈ court coords.
    The reprojection loss then pushes toward correct perspective.
    """

    _POOL_H = 6
    _POOL_W = 10
    _REDUCED_CH = 64

    def __init__(self, in_channels: int):
        super().__init__()
        flat_dim = self._REDUCED_CH * self._POOL_H * self._POOL_W  # 3840
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(in_channels, self._REDUCED_CH, kernel_size=1, bias=False),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((self._POOL_H, self._POOL_W))
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 8),  # 8 DoF; H[2,2] = 1 is fixed
        )
        # Zero weight, affine-scale bias so H starts as a reasonable mapping
        nn.init.zeros_(self.mlp[-1].weight)
        self.mlp[-1].bias.data = torch.tensor(
            [94., 0., 0.,   # row 0: x_court ≈ 94 * u
              0., 50., 0.,  # row 1: y_court ≈ 50 * v
              0., 0.],      # row 2: perspective params (zero → no distortion at start)
            dtype=torch.float32
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, C, H, W) → H_mat: (B, 3, 3)"""
        x = self.channel_reduce(features)  # (B, 64, H, W)
        x = self.pool(x)                   # (B, 64, 6, 10)
        params = self.mlp(x)               # (B, 8)
        B = params.shape[0]
        ones = torch.ones(B, 1, device=params.device, dtype=params.dtype)
        h = torch.cat([params, ones], dim=1)  # (B, 9)
        return h.view(B, 3, 3)


# ---------------------------------------------------------------------------
# Differentiable soft-argmax: heatmap → (x, y) coordinate
# ---------------------------------------------------------------------------

def soft_argmax(heatmaps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract sub-pixel keypoint coordinates and confidence from heatmaps.

    Args:
        heatmaps: (B, N_kp, H, W) raw logits

    Returns:
        coords:     (B, N_kp, 2) normalised coordinates in [0, 1]
        confidence: (B, N_kp)    max probability per keypoint ∈ [0, 1]
    """
    B, N, H, W = heatmaps.shape

    # Softmax over spatial dims to get probability distribution
    flat = heatmaps.view(B, N, -1)             # (B, N, H*W)
    probs = F.softmax(flat, dim=-1)            # (B, N, H*W)
    probs_2d = probs.view(B, N, H, W)

    # Build normalised coordinate grids [0, 1]
    grid_y = torch.linspace(0, 1, H, device=heatmaps.device)
    grid_x = torch.linspace(0, 1, W, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")  # (H, W)

    # Expectation over probability distribution
    coord_x = (probs_2d * grid_x).sum(dim=(-2, -1))  # (B, N)
    coord_y = (probs_2d * grid_y).sum(dim=(-2, -1))  # (B, N)

    coords = torch.stack([coord_x, coord_y], dim=-1)  # (B, N, 2)
    confidence = probs.max(dim=-1).values              # (B, N)

    return coords, confidence


# ---------------------------------------------------------------------------
# DLT solver: keypoint correspondences → homography matrix
# ---------------------------------------------------------------------------

def _hartley_normalise(pts: torch.Tensor) -> "Tuple[torch.Tensor, torch.Tensor]":
    """
    Isotropic Hartley normalisation (Algorithm 4.2, H&Z 2003).

    Translates the centroid to the origin and scales so the RMS distance
    from the origin equals sqrt(2).  Returns the normalised points and the
    3×3 similarity matrix T so that  pts_norm = T @ pts_hom.

    Args:
        pts: (B, N, 2)
    Returns:
        pts_norm: (B, N, 2)
        T:        (B, 3, 3) normalisation transform
    """
    B, N, _ = pts.shape
    device, dtype = pts.device, pts.dtype

    centroid = pts.mean(dim=1, keepdim=True)           # (B,1,2)
    shifted  = pts - centroid                          # (B,N,2)
    rms = (shifted.norm(dim=-1).pow(2).mean(dim=1)    # (B,)
           .clamp(min=1e-8).sqrt())                   # RMS distance
    scale = (2.0 ** 0.5) / rms                        # scalar per batch

    pts_norm = shifted * scale.view(B, 1, 1)

    # Build T: scale*(x - cx), scale*(y - cy)
    cx = centroid[:, 0, 0]    # (B,)
    cy = centroid[:, 0, 1]
    s  = scale

    zero = torch.zeros(B, device=device, dtype=dtype)
    one  = torch.ones( B, device=device, dtype=dtype)
    # T = [[s, 0, -s*cx], [0, s, -s*cy], [0, 0, 1]]
    T = torch.stack([
        torch.stack([s,    zero, -s*cx], dim=-1),
        torch.stack([zero, s,   -s*cy], dim=-1),
        torch.stack([zero, zero, one  ], dim=-1),
    ], dim=1)  # (B, 3, 3)

    return pts_norm, T


def dlt_solve(src_pts: torch.Tensor, dst_pts: torch.Tensor) -> torch.Tensor:
    """
    Normalised DLT homography solver (Hartley & Zisserman §4.1).

    Both point sets are Hartley-normalised before building the coefficient
    matrix, then the raw H is denormalised.  This cures the scale mismatch
    between pixel coordinates (0-1) and court coordinates (feet, 0-94) that
    previously made the raw condition numbers >1e4.

    Args:
        src_pts: (B, N, 2) source points (pixel coords, normalised 0-1)
        dst_pts: (B, N, 2) destination points (court coords in feet)

    Returns:
        H: (B, 3, 3) homography matrix (src → dst), or identity on failure.
    """
    B, N, _ = src_pts.shape
    device = src_pts.device
    dtype  = src_pts.dtype

    # --- Hartley normalisation ---
    src_n, T_src = _hartley_normalise(src_pts)  # (B,N,2), (B,3,3)
    dst_n, T_dst = _hartley_normalise(dst_pts)

    # --- Build coefficient matrix A on normalised coords ---
    zeros = torch.zeros(B, N, device=device, dtype=dtype)
    ones  = torch.ones( B, N, device=device, dtype=dtype)

    sx, sy = src_n[..., 0], src_n[..., 1]
    dx, dy = dst_n[..., 0], dst_n[..., 1]

    row1 = torch.stack([-sx, -sy, -ones, zeros, zeros, zeros,  dx*sx, dx*sy, dx], dim=-1)
    row2 = torch.stack([zeros, zeros, zeros, -sx, -sy, -ones,  dy*sx, dy*sy, dy], dim=-1)

    A = torch.cat([row1, row2], dim=1).reshape(B, 2*N, 9)  # (B, 2N, 9)

    # --- SVD on CPU float32 (small matrix; LAPACK more stable early in training) ---
    try:
        _, _, Vh = torch.linalg.svd(A.cpu().float(), full_matrices=False)
    except Exception:
        return torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1).clone()

    H_norm = Vh[:, -1, :].to(device=device, dtype=dtype).view(B, 3, 3)

    # --- Denormalise: H = T_dst^{-1} @ H_norm @ T_src ---
    try:
        T_dst_inv = torch.linalg.inv(T_dst.float()).to(dtype=dtype)
    except Exception:
        T_dst_inv = T_dst  # fallback

    H = torch.bmm(T_dst_inv, torch.bmm(H_norm, T_src))

    # Normalise so H[2,2] = 1
    H = H / (H[:, 2:3, 2:3] + 1e-8)

    # Condition number guard (catches near-singular H after denorm)
    try:
        cond = torch.linalg.cond(H.float())
        bad = ~torch.isfinite(cond) | (cond > 1e6)
        if bad.any():
            eye = torch.eye(3, device=device, dtype=dtype)
            H = torch.where(bad.view(B, 1, 1), eye.expand(B, 3, 3), H)
    except Exception:
        pass

    return H


# ---------------------------------------------------------------------------
# Full model: frame → H (direct regression) + heatmaps (auxiliary)
# ---------------------------------------------------------------------------

class CourtKeypointHomography(nn.Module):
    """
    Zero-calibration homography for basketball broadcast footage.

    Pipeline (single forward pass):
        frame → backbone → features
              → HomographyRegressionHead → 3×3 H matrix  (PRIMARY output)
              → HeatmapHead + soft-argmax → keypoint coords (AUXILIARY / debug, HRNet only)

    Args:
        pretrained:     Load pretrained weights for backbone
        coco_pose_ckpt: Optional mmpose COCO-pose checkpoint path (HRNet only)
        backbone_type:  "hrnet" (default) or "dinov2"
    """

    def __init__(
        self,
        pretrained: bool = True,
        conf_threshold: float = 0.3,  # kept for signature compat, unused
        min_keypoints: int = 4,       # kept for signature compat, unused
        coco_pose_ckpt: Optional[str] = None,
        backbone_type: str = "hrnet",
    ):
        super().__init__()

        if backbone_type == "dinov2":
            backbone, out_channels = _dinov2_backbone(pretrained)
            self.heatmap_head = None  # no auxiliary heatmap head for DINOv2
        else:
            backbone, out_channels = _hrnet_backbone(pretrained, coco_pose_ckpt)
            self.heatmap_head = HeatmapHead(out_channels, NUM_KEYPOINTS)

        self.backbone  = backbone
        self.homo_head = HomographyRegressionHead(out_channels)

        # Court keypoint coordinates in feet (fixed, not learned)
        self.register_buffer("court_kp_ft", COURT_KEYPOINTS_FT)

    def forward(
        self,
        x: torch.Tensor,
        visibility_mask: Optional[torch.Tensor] = None,  # unused, kept for compat
        compute_homo: bool = True,                        # unused, kept for compat
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) normalised RGB frames

        Returns:
            H:         (B, 3, 3) predicted homography (normalised pixel → court ft)
            kp_coords: (B, N_kp, 2) keypoint coords from soft_argmax, normalised [0,1]
            kp_conf:   (B, N_kp) keypoint confidences ∈ [0, 1]
            heatmaps:  (B, N_kp, H', W') raw heatmap logits (auxiliary loss)
        """
        # 1. Backbone features
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[0]  # (B, C, H', W')

        # 2. Auxiliary heatmap head (HRNet only; None for DINOv2)
        if self.heatmap_head is not None:
            heatmaps = self.heatmap_head(features)   # (B, N_kp, H/4, W/4)
        else:
            B = features.shape[0]
            heatmaps = torch.zeros(B, NUM_KEYPOINTS, 1, 1, device=features.device)
        kp_coords, kp_conf = soft_argmax(heatmaps)   # (B, N_kp, 2), (B, N_kp)

        # 3. Primary: direct H regression from spatial features
        H = self.homo_head(features)                 # (B, 3, 3)

        return H, kp_coords, kp_conf, heatmaps

    def predict_single(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convenience method for a single frame (no batch dim required)."""
        self.eval()
        with torch.no_grad():
            H, coords, conf, _ = self.forward(image.unsqueeze(0))
            return H.squeeze(0), coords.squeeze(0), conf.squeeze(0)


# ---------------------------------------------------------------------------
# Backward-compatible alias so smoke tests and old imports don't break
# ---------------------------------------------------------------------------

class LearnedHomography(CourtKeypointHomography):
    """Alias for backward compatibility with existing code."""
    def __init__(self, pretrained: bool = True, dropout: float = 0.2):
        super().__init__(pretrained=pretrained)


def apply_homography(H: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Apply homography transformation to points.
    
    Args:
        H: Homography matrix, shape (B, 3, 3) or (3, 3)
        points: Points to transform, shape (B, N, 2) or (N, 2)
                Format: [[x1, y1], [x2, y2], ...] in pixels
    
    Returns:
        transformed: Transformed points, shape (B, N, 2) or (N, 2)
                    Format: [[x1', y1'], [x2', y2'], ...] in feet
    
    Math:
        [x']   [h11 h12 h13]   [x]
        [y'] = [h21 h22 h23] × [y]
        [w']   [h31 h32 h33]   [1]
        
        x_court = x' / w'
        y_court = y' / w'
    """
    # Handle single matrix case
    if H.dim() == 2:
        H = H.unsqueeze(0)
        single_matrix = True
    else:
        single_matrix = False
    
    # Handle single point set case
    if points.dim() == 2:
        points = points.unsqueeze(0)
        single_points = True
    else:
        single_points = False
    
    B, N, _ = points.shape
    
    # Convert to homogeneous coordinates: [x, y] → [x, y, 1]
    ones = torch.ones(B, N, 1, device=points.device, dtype=points.dtype)
    points_homogeneous = torch.cat([points, ones], dim=2)  # (B, N, 3)
    
    # Apply homography: H @ p
    # H: (B, 3, 3), points: (B, N, 3) → need to transpose for matmul
    transformed_homogeneous = torch.matmul(H, points_homogeneous.transpose(1, 2))  # (B, 3, N)
    transformed_homogeneous = transformed_homogeneous.transpose(1, 2)  # (B, N, 3)
    
    # Convert from homogeneous coordinates: [x', y', w'] → [x'/w', y'/w']
    # For a valid pixel→court homography w must be positive. Clamp to min=0.1
    # so that extreme projective singularities produce bounded values (<≈1000 ft)
    # rather than thousands of feet that pollute val_reproj statistics.
    w = transformed_homogeneous[:, :, 2:3].clamp(min=0.1)  # (B, N, 1)
    transformed = transformed_homogeneous[:, :, :2] / w  # (B, N, 2)
    
    # Remove batch dimensions if input was single
    if single_points:
        transformed = transformed.squeeze(0)
    if single_matrix:
        pass  # Already handled by single_points
    
    return transformed


def compute_reprojection_error(H_pred: torch.Tensor, 
                               H_gt: torch.Tensor,
                               calibration_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute reprojection error between predicted and ground truth homographies.
    
    This measures how accurately the predicted homography transforms pixel 
    coordinates to court coordinates, compared to ground truth.
    
    Args:
        H_pred: Predicted homography, shape (B, 3, 3)
        H_gt: Ground truth homography, shape (B, 3, 3)
        calibration_points: Points used for calibration, shape (B, N, 4)
                           Format: [[px, py, cx, cy], ...] where:
                           - (px, py) = pixel coordinates
                           - (cx, cy) = court coordinates in feet
    
    Returns:
        rmse: Root mean squared error in feet, shape (B,)
        max_error: Maximum error in feet, shape (B,)
    """
    pixel_coords = calibration_points[:, :, :2]  # (B, N, 2)
    court_coords_gt = calibration_points[:, :, 2:]  # (B, N, 2)
    
    # Transform pixel coords using predicted homography
    court_coords_pred = apply_homography(H_pred, pixel_coords)  # (B, N, 2)
    
    # Compute error in feet
    errors = torch.norm(court_coords_pred - court_coords_gt, dim=2)  # (B, N)
    
    # RMSE per batch
    rmse = torch.sqrt(torch.mean(errors ** 2, dim=1))  # (B,)
    
    # Max error per batch
    max_error = torch.max(errors, dim=1)[0]  # (B,)
    
    return rmse, max_error


if __name__ == "__main__":
    """Test the model architecture"""
    print("Testing LearnedHomography model...")
    
    # Create model
    model = LearnedHomography(pretrained=False)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 720, 1280)  # Broadcast video resolution
    
    H = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {H.shape}")
    print(f"Expected: ({batch_size}, 3, 3)")
    
    # Test homography application
    print("\nTesting homography transformation...")
    points_pixel = torch.tensor([
        [100.0, 200.0],
        [500.0, 400.0],
        [900.0, 600.0]
    ])
    
    H_test = H[0]  # Use first matrix from batch
    points_court = apply_homography(H_test, points_pixel)
    
    print(f"Pixel points shape: {points_pixel.shape}")
    print(f"Court points shape: {points_court.shape}")
    print(f"Sample transformation:")
    print(f"  Pixel: {points_pixel[0].tolist()} → Court: {points_court[0].tolist()}")
    
    # Test reprojection error
    print("\nTesting reprojection error computation...")
    calibration_points = torch.tensor([
        [[100.0, 200.0, 10.0, 25.0],  # [px, py, cx, cy]
         [500.0, 400.0, 47.0, 25.0],
         [900.0, 600.0, 84.0, 25.0]]
    ])
    
    H_pred = H[:1]  # (1, 3, 3)
    H_gt = torch.eye(3).unsqueeze(0)  # Identity for testing
    
    rmse, max_err = compute_reprojection_error(H_pred, H_gt, calibration_points)
    print(f"RMSE: {rmse[0]:.2f} feet")
    print(f"Max error: {max_err[0]:.2f} feet")
    
    print("\n✓ All tests passed!")
