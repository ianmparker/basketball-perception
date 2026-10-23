"""
Phase 12e — Stub caching for expensive YOLO inference.

Ported conceptually from abdullahtarek/basketball_analysis/utils/stubs_utils.py
but redesigned around per-frame payloads (his version pickles the whole video
at once, we pickle one (video, op) entry with a frame-indexed dict so we can
also handle partial runs / --frames truncation cleanly).

Philosophy
──────────
The 4 per-frame GPU ops (court keypoints, player detect, ball detect, player
pose) dominate the runtime budget. Their outputs are *deterministic* for a
fixed (video, model). If we cache raw detections to disk, iterating on the
downstream stateful analytics (trackers, event detectors, possession,
smoothing, rendering) drops from "full inference run" to "replay cached
detections" — typically a 20–50× dev-loop speedup on a 5s clip.

Cache key = (video_path, video_mtime, video_size, model_path, model_mtime,
             stub_version). Any mismatch → ignore cache, recompute.

File layout
───────────
    <stub_dir>/
        <video_stem>__<op>.pkl

    Each pickle: {
        "stub_version": 1,
        "video": "<abs path>",
        "video_mtime": float,
        "video_size": int,
        "model_path": "<abs path>",
        "model_mtime": float,
        "op": "court_pose" | "player_detect" | "ball_detect" | "player_pose",
        "frames": {frame_idx: <op-specific payload>},
    }

Payload formats (all numpy-only, pickle-friendly):
    court_pose:    {"xy": (K,2) float32, "conf": (K,) float32} or None
    player_detect: (N,6) float32 [x1,y1,x2,y2,conf,cls] or empty array
    ball_detect:   (N,6) float32 [x1,y1,x2,y2,conf,cls] or empty array
    player_pose:   {"kps": (M,17,2), "kpc": (M,17), "boxes": (M,4)} or None

Usage
─────
    from inference.stubs import StubCache

    cache = StubCache(
        stub_dir="data/stubs",
        video_path="data/sample/videos/broadcast_74.mp4",
        model_path="runs/detect/phase6_yolo11m/weights/best.pt",
        op="player_detect",
    )
    if cache.hit(frame_idx):
        dets = cache.get(frame_idx)
    else:
        dets = run_yolo(frame)                # expensive
        cache.put(frame_idx, dets)
    ...
    cache.flush()  # writes pickle once at end

The cache is silently disabled if stub_dir is None or any I/O fails — the
pipeline always remains correct, stubs are a pure speedup.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

STUB_VERSION = 1

# 4 canonical op names — used for filename + header validation.
OP_COURT_POSE    = "court_pose"
OP_PLAYER_DETECT = "player_detect"
OP_BALL_DETECT   = "ball_detect"
OP_PLAYER_POSE   = "player_pose"

_VALID_OPS = {OP_COURT_POSE, OP_PLAYER_DETECT, OP_BALL_DETECT, OP_PLAYER_POSE}


def _safe_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


class StubCache:
    """Per-(video, op) on-disk cache of raw YOLO outputs.

    Lazy-loaded: the pickle is read on first `hit()` query. Writes are buffered
    in-memory and flushed to disk via `flush()` at end of run.
    """

    def __init__(
        self,
        stub_dir: Optional[str | os.PathLike],
        video_path: str | os.PathLike,
        model_path: str | os.PathLike,
        op: str,
        *,
        verbose: bool = True,
    ):
        if op not in _VALID_OPS:
            raise ValueError(f"Unknown op {op!r}; must be one of {sorted(_VALID_OPS)}")
        self.op = op
        self._verbose = verbose
        self._disabled = stub_dir is None
        self._loaded = False
        self._dirty = False
        self._frames: Dict[int, Any] = {}

        if self._disabled:
            self._path: Optional[Path] = None
            self._valid_existing = False
            return

        self._video_path = Path(video_path).resolve()
        self._model_path = Path(model_path).resolve()

        stub_dir_p = Path(stub_dir)
        stub_dir_p.mkdir(parents=True, exist_ok=True)
        stem = self._video_path.stem
        self._path = stub_dir_p / f"{stem}__{op}.pkl"

        # Check whether the existing pickle (if any) matches our expected key.
        self._valid_existing = self._read_and_validate()

    # ── public API ────────────────────────────────────────────────────────────

    def hit(self, frame_idx: int) -> bool:
        if self._disabled or not self._valid_existing:
            return False
        return frame_idx in self._frames

    def get(self, frame_idx: int) -> Any:
        return self._frames[frame_idx]

    def put(self, frame_idx: int, payload: Any) -> None:
        if self._disabled:
            return
        self._frames[frame_idx] = payload
        self._dirty = True

    def flush(self) -> None:
        """Write pickle to disk if anything was added this run."""
        if self._disabled or self._path is None or not self._dirty:
            return
        vstat = _safe_stat(self._video_path)
        mstat = _safe_stat(self._model_path)
        payload = {
            "stub_version": STUB_VERSION,
            "video": str(self._video_path),
            "video_mtime": vstat.st_mtime if vstat else 0.0,
            "video_size":  vstat.st_size  if vstat else 0,
            "model_path":  str(self._model_path),
            "model_mtime": mstat.st_mtime if mstat else 0.0,
            "op": self.op,
            "frames": self._frames,
        }
        tmp = self._path.with_suffix(".pkl.tmp")
        try:
            with open(tmp, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, self._path)
            if self._verbose:
                print(f"[stubs] wrote {self._path.name}  ({len(self._frames)} frames)",
                      file=sys.stderr)
        except OSError as e:
            if self._verbose:
                print(f"[stubs] WARN: failed to write {self._path}: {e}", file=sys.stderr)
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        finally:
            self._dirty = False

    @property
    def path(self) -> Optional[Path]:
        return self._path

    @property
    def valid_existing(self) -> bool:
        return self._valid_existing

    # ── internals ─────────────────────────────────────────────────────────────

    def _read_and_validate(self) -> bool:
        """Load pickle if present; return True iff key matches current video+model."""
        if self._path is None or not self._path.exists():
            return False
        try:
            with open(self._path, "rb") as f:
                payload = pickle.load(f)
        except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
            if self._verbose:
                print(f"[stubs] WARN: {self._path.name} unreadable ({e}); ignoring",
                      file=sys.stderr)
            return False

        if not isinstance(payload, dict) or payload.get("stub_version") != STUB_VERSION:
            if self._verbose:
                print(f"[stubs] {self._path.name}: stub_version mismatch; ignoring",
                      file=sys.stderr)
            return False
        if payload.get("op") != self.op:
            return False

        vstat = _safe_stat(self._video_path)
        mstat = _safe_stat(self._model_path)
        cur_v_mt = vstat.st_mtime if vstat else 0.0
        cur_v_sz = vstat.st_size  if vstat else 0
        cur_m_mt = mstat.st_mtime if mstat else 0.0

        # Video fingerprint: path + mtime + size. Very cheap, catches re-encodes.
        # (We intentionally don't hash the full video — too slow on long games.)
        video_match = (
            payload.get("video")      == str(self._video_path)
            and abs(float(payload.get("video_mtime", 0.0)) - cur_v_mt) < 1e-6
            and int(payload.get("video_size", 0)) == cur_v_sz
        )
        model_match = (
            payload.get("model_path") == str(self._model_path)
            and abs(float(payload.get("model_mtime", 0.0)) - cur_m_mt) < 1e-6
        )
        if not (video_match and model_match):
            if self._verbose:
                reasons = []
                if not video_match: reasons.append("video")
                if not model_match: reasons.append("model")
                print(f"[stubs] {self._path.name}: invalidated ({'+'.join(reasons)} changed)",
                      file=sys.stderr)
            return False

        frames = payload.get("frames")
        if not isinstance(frames, dict):
            return False
        self._frames = frames
        self._loaded = True
        if self._verbose:
            print(f"[stubs] loaded  {self._path.name}  ({len(frames)} frames cached)",
                  file=sys.stderr)
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory: build all 4 caches for a given video at once.

def build_all_caches(
    stub_dir: Optional[str | os.PathLike],
    video_path: str | os.PathLike,
    *,
    court_model_path: str | os.PathLike,
    player_model_path: str | os.PathLike,
    ball_model_path: str | os.PathLike,
    player_pose_path: str | os.PathLike,
    verbose: bool = True,
) -> Dict[str, StubCache]:
    """Return a dict op_name → StubCache for all four expensive ops."""
    return {
        OP_COURT_POSE:    StubCache(stub_dir, video_path, court_model_path,    OP_COURT_POSE,    verbose=verbose),
        OP_PLAYER_DETECT: StubCache(stub_dir, video_path, player_model_path,   OP_PLAYER_DETECT, verbose=verbose),
        OP_BALL_DETECT:   StubCache(stub_dir, video_path, ball_model_path,     OP_BALL_DETECT,   verbose=verbose),
        OP_PLAYER_POSE:   StubCache(stub_dir, video_path, player_pose_path,    OP_PLAYER_POSE,   verbose=verbose),
    }
