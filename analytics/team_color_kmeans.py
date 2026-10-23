"""
analytics/team_color_kmeans.py — K-means color-based team classification.

Automatically learns two jersey color clusters directly from torso crops during
a warmup phase (no prior knowledge of team colors required).  Works well for
broadcast footage where CLIP text-prompt accuracy is limited by small crop sizes
or similar jersey colors (e.g. gold vs navy, dark blue vs dark purple).

Algorithm:
  1. Warmup (first `warmup_frames` frames): collect mean HSV color from each
     player crop.  After collecting enough samples, fit K-means (k=2).
  2. Live phase: assign each new crop to the nearest K-means centroid.
  3. Temporal smoothing: per-track sliding-window majority vote, same as the
     CLIP classifier.  This handles motion-blur, partial occlusions, etc.

Color distance (B-02)
---------------------
Raw OpenCV HSV is a bad metric space: hue is circular (179 and 1 are
neighbours) and its range (0–180) is half that of S/V (0–255), so a
straight-line distance over-weights S/V and can call two nearly identical
reds maximally different.  We instead encode each crop as

    [cos(2πH/180)·S, sin(2πH/180)·S, S, V] / 255

The hue components put hue on a circle, scale it by saturation (a gray
pixel's hue is meaningless), and dividing every channel by 255 makes the
four features commensurate before K-means sees them.

Margin normalization (B-15)
---------------------------
A sparse cluster (e.g. gold jerseys, few pixels in colour space) has a
larger natural spread than a dense one.  A single global `min_margin` would
therefore starve the sparse cluster.  Distances are divided by each
cluster's own RMS spread before the confidence margin is computed.

The centroid with the higher V (brightness) channel in HSV is mapped to the
team that wears lighter jerseys.  When clips.json provides jersey descriptions
(light vs dark), that mapping is used; otherwise the brighter centroid → team 1.
"""
from __future__ import annotations

import sys
from collections import Counter, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

_HUE_PERIOD = 180.0
_FEATURE_SCALE = 255.0
_MIN_CLUSTER_SCALE = 1e-3


class KMeansTeamAssigner:
    """
    K-means colour-based team assignment for broadcast basketball footage.

    Drop-in replacement for CLIPTeamClassifier / TeamAssigner with the same
    `assign_frame(frame, frame_idx, players)` → Dict[track_id, team_id] API.

    Args:
        warmup_frames:   number of frames to collect colour samples before
                         fitting K-means.  Default 30 (≈ 1 second at 30 fps).
        vote_window:     size of per-track majority-vote window.
        min_votes:       votes required before emitting a team_id.
        team1_is_light:  if True, brighter cluster → team 1 (default True).
                         Flip if the home team is actually the darker jersey.
        min_margin:      minimum normalized distance ratio between clusters
                         required to count a vote (prevents near-centre votes).
    """

    def __init__(
        self,
        warmup_frames: int = 30,
        vote_window: int = 9,
        min_votes: int = 1,
        team1_is_light: bool = True,
        min_margin: float = 0.05,
    ):
        self.warmup_frames = max(5, int(warmup_frames))
        self.vote_window   = max(1, int(vote_window))
        self.min_votes     = max(1, int(min_votes))
        self.team1_is_light = team1_is_light
        self.min_margin     = float(min_margin)

        # Collected samples during warmup: list of (track_id, mean_hsv)
        self._warmup_samples: List[np.ndarray] = []  # shape (N, 3) mean HSV
        self._warmup_complete = False
        self._centroids: Optional[np.ndarray] = None  # (2, 4) encoded centroids
        self._cluster_scale: Optional[np.ndarray] = None  # (2,) RMS spread

        # Diagnostics (B-35): surface a silently-failed warmup instead of
        # emitting no team IDs for the whole clip.
        self._crops_seen = 0
        self._last_frame_idx = -1
        self._fit_frame: Optional[int] = None

        # Per-track vote history
        self._votes: Dict[int, Deque[int]] = {}
        self._last_classified: Dict[int, int] = {}

    # ── colour encoding (B-02) ────────────────────────────────────────────────

    @staticmethod
    def _encode_hsv(hsv: np.ndarray) -> np.ndarray:
        """Encode raw OpenCV HSV as the 4-D metric-space feature vector.

        Accepts a single (3,) HSV sample or an (N, 3) batch and returns the
        matching (4,) / (N, 4) float64 array.  See the module docstring for
        why hue is split into cos/sin and scaled by saturation.
        """
        arr = np.asarray(hsv, dtype=np.float64).reshape(-1, 3)
        theta = 2.0 * np.pi * arr[:, 0] / _HUE_PERIOD
        sat = arr[:, 1]
        return np.stack(
            [np.cos(theta) * sat, np.sin(theta) * sat, sat, arr[:, 2]],
            axis=1,
        ) / _FEATURE_SCALE

    # ── crop helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _torso_crop_color(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """
        Extract the mean HSV colour of the torso region.

        Returns a (3,) HSV array, or None if the crop is too small.
        Same crop geometry as CLIPTeamClassifier._torso_crop:
          vertical  20–60 % of bbox  (slightly wider than CLIP's 20–50 % to
                                       capture more jersey)
          horizontal  20 % inset on each side
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(x1 + 1, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(y1 + 1, min(y2, h))
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 20:
            return None
        ty1 = y1 + int(0.20 * bh)
        ty2 = y1 + int(0.60 * bh)
        tx1 = x1 + int(0.20 * bw)
        tx2 = x2 - int(0.20 * bw)
        if ty2 - ty1 < 6 or tx2 - tx1 < 6:
            return None
        crop = frame[ty1:ty2, tx1:tx2]
        hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Use median (robust to small logo / number patches)
        return np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)

    # ── K-means fitting ───────────────────────────────────────────────────────

    def _fit_kmeans(self) -> None:
        """Fit K-means (k=2) on collected warmup samples."""
        if len(self._warmup_samples) < 4:
            return  # Not enough data yet

        data = self._encode_hsv(np.stack(self._warmup_samples)).astype(np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        _, labels, self._centroids = cv2.kmeans(
            data, 2, None, criteria, attempts=10, flags=cv2.KMEANS_PP_CENTERS
        )
        # self._centroids: (2, 4) encoded centroids

        # Per-cluster RMS spread, used to normalize margins (B-15).
        flat_labels = labels.flatten()
        scales = np.empty(2, dtype=np.float32)
        for k in range(2):
            member_d = np.linalg.norm(data[flat_labels == k] - self._centroids[k], axis=1)
            scales[k] = max(
                float(np.sqrt(np.mean(member_d ** 2))) if member_d.size else 0.0,
                _MIN_CLUSTER_SCALE,
            )
        self._cluster_scale = scales

        # Determine which centroid → team 1.
        # Use Value (brightness) channel: higher V = lighter jersey.
        c0_v = self._centroids[0, 3] * _FEATURE_SCALE
        c1_v = self._centroids[1, 3] * _FEATURE_SCALE
        if self.team1_is_light:
            # Brighter centroid → team 1
            self._light_centroid_idx = 0 if c0_v >= c1_v else 1
        else:
            # Darker centroid → team 1
            self._light_centroid_idx = 0 if c0_v < c1_v else 1

        self._warmup_complete = True
        self._fit_frame = self._last_frame_idx
        print(
            f"[KMeans] Centroids fitted on {len(self._warmup_samples)} crops. "
            f"Centroid 0 V={c0_v:.0f} (scale={scales[0]:.3f}), "
            f"Centroid 1 V={c1_v:.0f} (scale={scales[1]:.3f}). "
            f"Team-1 centroid idx={self._light_centroid_idx}."
        )

    def _centroid_team(self, idx: int) -> int:
        """Map centroid index → team_id (1 or 2)."""
        if idx == self._light_centroid_idx:
            return 1
        return 2

    def _assign_color(self, hsv: np.ndarray) -> Tuple[int, float]:
        """
        Assign to nearest centroid in the encoded colour space.

        Distances are divided by each cluster's own RMS spread first (B-15),
        so a sparse cluster is not systematically gated out by a margin
        threshold calibrated on a dense one.

        Returns (team_id, margin) where margin is the relative normalized
        distance difference: 1 - d_nearest / d_other.  Larger = more confident.
        """
        x = self._encode_hsv(hsv)[0]
        d0 = float(np.linalg.norm(x - self._centroids[0]))
        d1 = float(np.linalg.norm(x - self._centroids[1]))
        scales = (
            self._cluster_scale
            if self._cluster_scale is not None
            else np.ones(2, dtype=np.float32)
        )
        d0n = d0 / max(float(scales[0]), _MIN_CLUSTER_SCALE)
        d1n = d1 / max(float(scales[1]), _MIN_CLUSTER_SCALE)
        nearest = 0 if d0n <= d1n else 1
        d_near  = min(d0n, d1n)
        d_far   = max(d0n, d1n)
        margin  = 1.0 - d_near / (d_far + 1e-6)
        return self._centroid_team(nearest), margin

    # ── vote helpers ──────────────────────────────────────────────────────────

    def _emit(self, track_id: int) -> Optional[int]:
        votes = self._votes.get(track_id)
        if not votes or len(votes) < self.min_votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    # ── public API ────────────────────────────────────────────────────────────

    def assign_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        players: Iterable[Tuple[int, Tuple[int, int, int, int]]],
    ) -> Dict[int, int]:
        """
        Same signature as CLIPTeamClassifier.assign_frame.

        Args:
            frame:     full BGR frame.
            frame_idx: 0-based frame index.
            players:   iterable of (track_id, bbox_xyxy).

        Returns:
            dict mapping track_id → team_id (1 or 2).
        """
        out: Dict[int, int] = {}
        players_list = list(players)
        self._last_frame_idx = max(self._last_frame_idx, int(frame_idx))

        for track_id, bbox in players_list:
            hsv = self._torso_crop_color(frame, bbox)
            if hsv is None:
                cur = self._emit(track_id)
                if cur is not None:
                    out[track_id] = cur
                continue

            self._crops_seen += 1

            if not self._warmup_complete:
                # Collect warmup samples
                self._warmup_samples.append(hsv)
                # Attempt fit once we have enough data
                if frame_idx >= self.warmup_frames and len(self._warmup_samples) >= 20:
                    self._fit_kmeans()
                # Before warmup complete, emit any prior votes
                cur = self._emit(track_id)
                if cur is not None:
                    out[track_id] = cur
                continue

            # Live classification
            team_id, margin = self._assign_color(hsv)
            if margin >= self.min_margin:
                dq = self._votes.setdefault(track_id, deque(maxlen=self.vote_window))
                dq.append(team_id)
                self._last_classified[track_id] = frame_idx

            cur = self._emit(track_id)
            if cur is not None:
                out[track_id] = cur

        # Force a fit attempt if warmup_frames has passed and fit not done yet.
        # `>= 10` is the normal path.  The `>= 4` fallback at 2× warmup keeps
        # short clips / low-player frames from silently losing every team ID
        # (B-35); `finalize()` handles anything that still never fit.
        if not self._warmup_complete:
            if frame_idx >= self.warmup_frames and len(self._warmup_samples) >= 10:
                self._fit_kmeans()
            elif frame_idx >= 2 * self.warmup_frames and len(self._warmup_samples) >= 4:
                self._fit_kmeans()

        return out

    # ── diagnostics (B-35) ────────────────────────────────────────────────────

    @property
    def warmup_complete(self) -> bool:
        return self._warmup_complete

    @property
    def team_assignment_available(self) -> bool:
        """False means no team IDs were produced for this clip."""
        return bool(self._warmup_complete and self._centroids is not None)

    def diagnostics(self) -> Dict[str, object]:
        return {
            "warmup_complete": self._warmup_complete,
            "team_assignment_available": self.team_assignment_available,
            "warmup_frames_required": self.warmup_frames,
            "samples_collected": len(self._warmup_samples),
            "crops_seen": self._crops_seen,
            "last_frame_idx": self._last_frame_idx,
            "fit_frame": self._fit_frame,
        }

    def finalize(self) -> bool:
        """Last-chance fit at end of clip.  Loudly reports a failed warmup."""
        if not self._warmup_complete and len(self._warmup_samples) >= 4:
            self._fit_kmeans()
        if not self._warmup_complete:
            print(
                "[KMeans][ERROR] Warmup never completed: "
                f"{len(self._warmup_samples)} usable torso crops across "
                f"{self._last_frame_idx + 1} frames (need >=4). "
                "No team IDs were emitted; downstream team/possession stats "
                "are unavailable for this clip.",
                file=sys.stderr,
            )
        return self.team_assignment_available

    # ── back-compat ───────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """No-op — K-means needs no model loading."""
        pass

    @property
    def player_team_dict(self) -> Dict[int, int]:
        return {tid: self._emit(tid) for tid in self._votes if self._emit(tid) is not None}
