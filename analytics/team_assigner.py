"""
analytics/team_assigner.py — CLIP fashion-clip zero-shot team identification.

Originally ported from abdullahtarek/basketball_analysis (MIT licensed):
  https://github.com/abdullahtarek/basketball_analysis/blob/main/team_assigner/team_assigner.py

Phase 12a polish (over the reference impl):
  * Torso-only crop (kills shorts / court / nearby-player noise sent to CLIP).
  * Prompt ensemble — average softmax across several phrasings per team.
  * Confidence gating — discard low-margin votes instead of committing them.
  * Temporal majority vote — sliding window of recent votes per track; the
    emitted team_id is the mode.  Survives one-off bad frames at track spawn.
  * Periodic re-vote (every `revote_every` frames per track) instead of a
    hard cache wipe — gives smooth drift recovery without losing history.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


# Prompt templates — each "{}" gets the user's class name substituted in.
# Logits are averaged across templates for each class before softmax.
_PROMPT_TEMPLATES: Tuple[str, ...] = (
    "a basketball player wearing a {}",
    "a {} jersey",
    "a person in a {}",
    "{}",
)


class TeamAssigner:
    """
    Assigns each tracked player to one of two teams using CLIP zero-shot
    classification on torso crops, smoothed over a temporal window.

    Args:
        team_1_class_name:   text describing team 1 (e.g. "white jersey").
        team_2_class_name:   text describing team 2 (e.g. "dark blue jersey").
        model_name:          HuggingFace CLIP checkpoint; default fashion-clip.
        revote_every:        re-classify each track every N frames (drift defense).
        vote_window:         number of recent votes kept per track (sliding window).
        min_votes:           need this many votes before emitting a team_id.
        min_margin:          softmax margin (p_top - p_other) required to count a vote.
        device:              "cuda" / "cpu" / None for auto.
    """

    def __init__(
        self,
        team_1_class_name: str = "white jersey",
        team_2_class_name: str = "dark blue jersey",
        model_name: str = "patrickjohncyh/fashion-clip",
        revote_every: int = 10,
        vote_window: int = 9,
        min_votes: int = 2,
        min_margin: float = 0.05,
        device: Optional[str] = None,
        # Back-compat shim: older callers passed `refresh_every`. Treat it as revote_every.
        refresh_every: Optional[int] = None,
    ):
        if refresh_every is not None:
            revote_every = refresh_every
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name
        self.revote_every = max(1, int(revote_every))
        self.vote_window = max(1, int(vote_window))
        self.min_votes = max(1, int(min_votes))
        self.min_margin = float(min_margin)

        # Build prompt list:  [t1_p1, t1_p2, ..., t2_p1, t2_p2, ...]
        self._prompts_team1 = [tpl.format(team_1_class_name) for tpl in _PROMPT_TEMPLATES]
        self._prompts_team2 = [tpl.format(team_2_class_name) for tpl in _PROMPT_TEMPLATES]
        self._all_prompts = self._prompts_team1 + self._prompts_team2
        self._n_per_class = len(_PROMPT_TEMPLATES)

        self._device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Lazy-loaded
        self._model: Optional[CLIPModel] = None
        self._processor: Optional[CLIPProcessor] = None
        self._model_name = model_name

        # Per-track vote history: track_id -> deque of recent {1,2} votes
        self._votes: Dict[int, Deque[int]] = {}
        # Per-track last frame_idx the track was (re)classified
        self._last_classified: Dict[int, int] = {}

    # ── model loading ─────────────────────────────────────────────────────────

    def load_model(self) -> None:
        if self._model is not None:
            return
        self._model = CLIPModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained(self._model_name)

    # ── crop helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _torso_crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Image.Image]:
        """
        Tight central torso crop:
          * vertical: keep 20–50% of bbox (chest area, below head, above waist).
          * horizontal: inset 28% on each side (jersey center, miss arms+background).

        Distant players whose bbox contains lots of court / crowd otherwise dominate
        the CLIP color signal; this tighter window keeps mostly jersey pixels.
        Returns RGB PIL image, or None if the resulting crop is too small.
        """
        x1, y1, x2, y2 = (int(v) for v in bbox)
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h))
        bw, bh = x2 - x1, y2 - y1
        if bw < 10 or bh < 20:
            return None
        ty1 = y1 + int(0.20 * bh)
        ty2 = y1 + int(0.50 * bh)
        tx1 = x1 + int(0.28 * bw)
        tx2 = x2 - int(0.28 * bw)
        if ty2 - ty1 < 6 or tx2 - tx1 < 6:
            return None
        bgr = frame[ty1:ty2, tx1:tx2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    # ── core classification ───────────────────────────────────────────────────

    @torch.no_grad()
    def _classify_batch(self, images: List[Image.Image]) -> List[Tuple[int, float]]:
        """
        Run CLIP on a batch of crops with prompt ensembling.

        Returns one (class_idx, margin) per crop, where:
          class_idx ∈ {0, 1}  →  team 1 / team 2
          margin = p_top - p_other  ∈ [0, 1]
        """
        if not images:
            return []
        self.load_model()
        inputs = self._processor(
            text=self._all_prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        outputs = self._model(**inputs)
        # logits_per_image: (B, n_prompts) where n_prompts = 2 * n_per_class
        logits = outputs.logits_per_image
        # Average logits within each class (prompt ensemble) → (B, 2)
        per_class = logits.view(logits.shape[0], 2, self._n_per_class).mean(dim=2)
        probs = per_class.softmax(dim=1)  # (B, 2)
        top_p, top_idx = probs.max(dim=1)
        other_p = 1.0 - top_p
        margins = (top_p - other_p).clamp(min=0.0)
        return list(zip(top_idx.tolist(), margins.tolist()))

    # ── per-frame public API ──────────────────────────────────────────────────

    def _emit(self, track_id: int) -> Optional[int]:
        """Return the current majority team_id for a track, or None if undecided."""
        votes = self._votes.get(track_id)
        if not votes or len(votes) < self.min_votes:
            return None
        return Counter(votes).most_common(1)[0][0]

    def assign_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        players: Iterable[Tuple[int, Tuple[int, int, int, int]]],
    ) -> Dict[int, int]:
        """
        Assign team_id ∈ {1, 2} to each player in this frame.

        Args:
            frame:     full BGR frame.
            frame_idx: 0-based frame index (used for per-track re-vote scheduling).
            players:   iterable of (track_id, bbox_xyxy) for player-class detections only.

        Returns:
            dict mapping track_id -> team_id (1 or 2). Tracks that have not yet
            accumulated `min_votes` are omitted.
        """
        out: Dict[int, int] = {}
        to_classify: List[Tuple[int, Image.Image]] = []

        for track_id, bbox in players:
            last = self._last_classified.get(track_id)
            need_vote = (
                last is None
                or (frame_idx - last) >= self.revote_every
                or len(self._votes.get(track_id, ())) < self.min_votes
            )
            if need_vote:
                crop = self._torso_crop(frame, bbox)
                if crop is not None:
                    to_classify.append((track_id, crop))
            cur = self._emit(track_id)
            if cur is not None:
                out[track_id] = cur

        if to_classify:
            tids   = [t for t, _ in to_classify]
            crops  = [c for _, c in to_classify]
            preds  = self._classify_batch(crops)
            for tid, (idx, margin) in zip(tids, preds):
                self._last_classified[tid] = frame_idx
                if margin < self.min_margin:
                    continue  # low-confidence vote → skip
                team_id = 1 if idx == 0 else 2
                dq = self._votes.setdefault(tid, deque(maxlen=self.vote_window))
                dq.append(team_id)
                # Re-emit: a freshly-promoted track may now satisfy min_votes
                cur = self._emit(tid)
                if cur is not None:
                    out[tid] = cur

        return out

    # ── back-compat alias used by older callers ───────────────────────────────
    @property
    def player_team_dict(self) -> Dict[int, int]:
        return {tid: self._emit(tid) for tid in self._votes if self._emit(tid) is not None}
