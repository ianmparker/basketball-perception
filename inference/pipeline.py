"""
Phase 10 — BasketballPipeline

Single importable class that owns all model weights and stateful trackers.
Call `.process_frame(frame, frame_idx)` once per frame; get back a FrameResult.

Usage
─────
    from inference.pipeline import BasketballPipeline, PipelineConfig

    cfg = PipelineConfig(enable_pose=True, enable_events=True)
    pipeline = BasketballPipeline(cfg)

    cap = cv2.VideoCapture("game.mp4")
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = pipeline.process_frame(frame, fi)
        fi += 1

    pipeline.print_summary()
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.ops as tvops
from ultralytics import YOLO

# NMS CPU fallback (torchvision 0.26.0+cpu missing CUDA kernel)
_orig_nms = tvops.nms


def _nms_fallback(b, s, t):
    return _orig_nms(b.cpu(), s.cpu(), float(t)).to(b.device)


tvops.nms = _nms_fallback

# ── internal imports ──────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# BoxMOT moved this module between releases. Support the layouts used by the
# original dev environment and current PyPI packages.
try:  # older layout
    from boxmot.trackers.bbox.botsort.botsort import BotSort
except ImportError:  # BoxMOT 11–13
    try:
        from boxmot.trackers.botsort.botsort import BotSort
    except ImportError:  # BoxMOT 10
        from boxmot.trackers.botsort.bot_sort import BotSort
from events.shot_detector_v2 import OnlineShotDetector, ShotEvent
from events.screen_detector_v2 import OnlineScreenDetector, ScreenEvent
from events.pnr_detector import OnlinePnRDetector, PnREvent
from events.dunk_detector import OnlineDunkDetector, DunkEvent
from tracking.simple_ball_tracker import SimpleBallTracker


# ─────────────────────────── court constants ──────────────────────────────────

# COCO position → NBA court (ft). Origin at near-baseline × bottom-sideline.
# x ∈ [0, 94] runs along court length (near→far). y ∈ [0, 50] across width.
# Identities verified against data/court_keypoints/train/_annotations.coco.json
# `keypoints` (semantic-id order) and `skeleton` (court-line connectivity):
#   chain  0-1-2-3-4-5-6  : near baseline + halfcourt @ bottom sideline
#   edge   0-7            : top-sideline halfcourt closes the near half
#   quad   2-8-9-3        : NEAR free-throw lane (paint, FT line at x=19)
#   chain  6-10-11-12-13-14-15 : far baseline
#   edge   15-7           : top-sideline halfcourt closes the far half
#   quad   13-16-17-12    : FAR free-throw lane (paint, FT line at x=75)
COURT_KPS_FT = np.array([    [ 0.0, 50.0],  # 0  near baseline @ top sideline
    [ 0.0, 47.0],  # 1  near baseline 3pt corner (top side)
    [ 0.0, 33.0],  # 2  near baseline paint corner (top)
    [ 0.0, 17.0],  # 3  near baseline paint corner (bottom)
    [ 0.0,  3.0],  # 4  near baseline 3pt corner (bottom side)
    [ 0.0,  0.0],  # 5  near baseline @ bottom sideline
    [47.0,  0.0],  # 6  halfcourt × bottom sideline
    [47.0, 50.0],  # 7  halfcourt × top sideline
    [19.0, 33.0],  # 8  NEAR FT line × paint top  (paired with 2 in skeleton)
    [19.0, 17.0],  # 9  NEAR FT line × paint bottom (paired with 3)
    [94.0,  0.0],  # 10 far baseline @ bottom sideline
    [94.0,  3.0],  # 11 far baseline 3pt corner (bottom)
    [94.0, 17.0],  # 12 far baseline paint corner (bottom)
    [94.0, 33.0],  # 13 far baseline paint corner (top)
    [94.0, 47.0],  # 14 far baseline 3pt corner (top)
    [94.0, 50.0],  # 15 far baseline @ top sideline
    [75.0, 33.0],  # 16 FAR FT line × paint top (paired with 13)
    [75.0, 17.0],  # 17 FAR FT line × paint bottom (paired with 12)
], dtype=np.float32)

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


# ─────────────────────────── config & result types ───────────────────────────

def is_court_view(gate_enabled: bool, H) -> bool:
    """Whether this frame should report players and events.

    With the gate enabled and no homography, the frame is not a court view
    (crowd shot, bench close-up, replay graphic) and nothing should be reported
    for it. Gate-off always returns True so close-up footage, which has no
    homography by construction, keeps its existing behaviour.
    """
    return not (gate_enabled and H is None)


def load_homography(path) -> np.ndarray:
    """Read a 3x3 court homography from JSON.

    Accepts the file written by ``homography/manual_homography.py`` (a dict with a
    ``homography_matrix`` key) and a bare 3x3 array. Manual calibration is exact
    for a fixed camera and needs no model, so this is the fallback whenever the
    learned court keypoint model does not fire.
    """
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("homography_matrix")
    if data is None:
        raise ValueError(f"{path}: no 'homography_matrix' in file")
    H = np.asarray(data, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"{path}: expected a 3x3 matrix, got {H.shape}")
    if abs(np.linalg.det(H)) < 1e-12:
        raise ValueError(f"{path}: homography is singular")
    return H


@dataclass
class PipelineConfig:
    # Model weights (relative paths resolved from repo root)
    court_model_path:  str = "runs/pose/court_kp_v18_resume/weights/best.pt"
    player_model_path: str = "runs/detect/phase6_yolo11m/weights/best.pt"
    ball_model_path:   str = "models/ball_detector_ebard_v3.pt"  # E-BARD replay-v3 target-domain YOLO11s (4-class: ball/hoop/player/referee); legacy 7-class detector can be selected with --ball-model models/ball_detector_model.pt
    player_pose_path:  str = "yolov8x-pose.pt"   # COCO pretrained

    # Inference options
    enable_pose:   bool = False   # player body pose (adds ~13 ms/frame)
    enable_events: bool = True    # shot / screen / PnR detection
    # Reject frames where the court cannot be located (crowd shots, close-ups).
    # Off by default because the close-up benchmark clips have no homography.
    court_view_gate: bool = False
    # Fixed pixel->court homography (manual calibration). When set, the court
    # keypoint model is skipped and this matrix is used for every frame, which is
    # correct for a single static camera and exact where the model fails.
    fixed_homography: Optional[np.ndarray] = None
    # Shot detection normally requires a valid court projection (it rejects
    # events whose distance cannot be established). Close-up, gym, and phone
    # footage often has no usable homography, so this can be relaxed to fall
    # back to pixel-space gating. Broadcast default stays strict.
    shot_require_court_projection: bool = True
    enable_team_id: bool = False  # CLIP team assignment (Phase 12)
    use_openai_clip: bool = False  # Phase 12f: use OpenAI CLIP instead of K-means (alternative; 85-95% vs 70-80% accuracy but struggles with dark jerseys)
    use_kmeans_teams: bool = True   # K-means colour clustering — PRIMARY DEFAULT team assigner (no prompts needed; works for all jersey combos including gold/navy, dark/dark)
    kmeans_team1_is_light: bool = True  # brighter K-means cluster → team 1 (set False if home team wears darker jersey)
    enable_possession: bool = False  # ball-carrier + team-possession tracking (Phase 12b)

    # Team identification (used when enable_team_id=True)
    team1_desc: str = "white jersey"
    team2_desc: str = "dark blue jersey"
    team_refresh_every: int = 10     # per-track re-vote cadence (sliding window)
    
    # K-means team classifier tuning (Phase 12g; ignored if use_kmeans_teams=False)
    kmeans_warmup_frames: int = 40    # frames to collect before first K-means fit (30 = default)
    kmeans_vote_window: int = 9       # frames to vote over for team stability
    kmeans_min_votes: int = 2         # votes needed before emitting team_id (1 = default; 2+ = more stable)
    # Margin is computed on per-cluster-normalized colour distances (B-02/B-15).
    # The old 0.15 was tuned against a broken (non-circular, unscaled) distance;
    # re-tune it against eval/score.py before treating it as a measured choice.
    kmeans_min_margin: float = 0.15
    
    # OpenAI CLIP settings (Phase 12f; ignored if use_openai_clip=False)
    clip_model_name: str = "ViT-B-32"  # Model size: "ViT-B-32" (fast), "ViT-B-16", "ViT-L-14" (best)
    team1_clip_prompts: Optional[List[str]] = None  # Overrides team1_desc; if None, uses team1_desc
    team2_clip_prompts: Optional[List[str]] = None  # Overrides team2_desc; if None, uses team2_desc

    # Detection thresholds
    court_kp_conf:  float = 0.3
    player_conf:    float = 0.25
    player_pose_conf: float = 0.25
    ball_conf:      float = 0.25    # Replay-v3 best-F1 point on the clean 60-frame exam.
                                    # The legacy model's old .15 tuning is still available via --ball-conf.
    ball_imgsz:     int   = 1280    # Replay-v3 is a 1280 model; the legacy detector can use 640.
    ball_class_id:  int   = 0       # Ball is class 0 in both the replay-v3 and legacy models.
    # Ball candidate selection: "motion" continues the existing track (default),
    # "argmax" restores the original highest-confidence-per-frame behaviour.
    ball_picker:    str   = "motion"
    kp_conf_pose:   float = 0.30

    # Tracker settings
    # Ball tracking: simple picker + offline interpolation (Kalman removed as it overcomplicates
    # basketball's long occlusions 16+ frames). Uses motion gate + linear interp in two-pass mode.
    fps:            float = 30.0

    # Phase 12e — stub caching. When `stub_dir` is set and `video_path` is provided,
    # raw YOLO outputs for the 4 expensive ops (court_pose, player_detect, ball_detect,
    # player_pose) are cached under <stub_dir>/<video_stem>__<op>.pkl on first run
    # and replayed on subsequent runs. Tracker / event-detector state still runs
    # from the cached detections — this is a dev-loop speedup, not a correctness
    # change. Silently disabled if stub_dir is None.
    stub_dir:   Optional[str] = None
    video_path: Optional[str] = None


@dataclass
class PlayerTrack:
    track_id: int
    box_xyxy: Tuple[int, int, int, int]   # (x1, y1, x2, y2) pixels
    conf: float
    cls_id: int
    court_pos: Optional[Tuple[float, float]] = None  # (x_ft, y_ft) or None
    pose_kps: Optional[np.ndarray] = None            # (17, 2) pixel coords
    pose_kpc: Optional[np.ndarray] = None            # (17,) confidences
    team_id: Optional[int] = None                    # 1 or 2 (Phase 12 team ID)
    # Phase 12d — populated offline by analytics/speed_distance.py (two-pass only)
    speed_ft_per_s: float = 0.0
    distance_traveled_ft: float = 0.0


@dataclass
class BallState:
    tracked: bool
    coasted: bool
    cx: float
    cy: float
    coast_frames: int
    court_pos: Optional[Tuple[float, float]] = None  # (x_ft, y_ft)
    det_box: Optional[Tuple[int, int, int, int]] = None
    det_conf: float = 0.0
    interpolated: bool = False                       # Phase 7b: hindsight-filled by smoother


@dataclass
class FrameResult:
    frame_idx: int
    H: Optional[np.ndarray]                    # 3×3 homography or None
    court_kps: Optional[np.ndarray]            # (18, 2) pixel coords
    court_kpc: Optional[np.ndarray]            # (18,) confidences
    players: List[PlayerTrack] = field(default_factory=list)
    ball: Optional[BallState] = None
    # Events (only populated when enable_events=True)
    new_shots:   List[ShotEvent]   = field(default_factory=list)
    new_screens: List[ScreenEvent] = field(default_factory=list)
    new_pnrs:    List[PnREvent]    = field(default_factory=list)
    new_dunks:   List[DunkEvent]   = field(default_factory=list)
    scene_cut: bool = False
    # Phase 12b — possession
    ball_carrier_track_id: Optional[int] = None
    possession_team_id: Optional[int] = None
    possession_distance_mode: Optional[str] = None  # 'court' | 'pixel' | None
    # Per-frame homography quality (B-32): the court mapping is an upstream
    # dependency of every court-space consumer, so make its reliability visible.
    homography_quality: Optional[dict] = None
    # False when the court-view gate rejected this frame (crowd shot, close-up,
    # replay graphic). No players/events are reported for such frames.
    court_view: bool = True

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict for offline analytics."""
        def _arr(a):
            return None if a is None else a.tolist()

        def _player(p: PlayerTrack) -> dict:
            return {
                "track_id": p.track_id,
                "box_xyxy": list(p.box_xyxy),
                "conf": float(p.conf),
                "cls_id": int(p.cls_id),
                "court_pos": list(p.court_pos) if p.court_pos else None,
                "pose_kps": _arr(p.pose_kps),
                "pose_kpc": _arr(p.pose_kpc),
                "team_id": p.team_id,
                "speed_ft_per_s": float(p.speed_ft_per_s),
                "distance_traveled_ft": float(p.distance_traveled_ft),
            }

        def _ball(b: Optional[BallState]) -> Optional[dict]:
            if b is None:
                return None
            return {
                "tracked": b.tracked,
                "coasted": b.coasted,
                "cx": float(b.cx),
                "cy": float(b.cy),
                "coast_frames": b.coast_frames,
                "court_pos": list(b.court_pos) if b.court_pos else None,
                "det_box": list(b.det_box) if b.det_box else None,
                "det_conf": float(b.det_conf),
                "interpolated": b.interpolated,
            }

        def _event(ev) -> dict:
            # ShotEvent / ScreenEvent / PnREvent / DunkEvent are dataclasses
            from dataclasses import asdict
            d = asdict(ev)
            # Strip any numpy arrays / non-JSON types
            return {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in d.items()}

        return {
            "frame_idx":   self.frame_idx,
            "H":           _arr(self.H),
            "court_kps":   _arr(self.court_kps),
            "court_kpc":   _arr(self.court_kpc),
            "players":     [_player(p) for p in self.players],
            "ball":        _ball(self.ball),
            "new_shots":   [_event(e) for e in self.new_shots],
            "new_screens": [_event(e) for e in self.new_screens],
            "new_pnrs":    [_event(e) for e in self.new_pnrs],
            "new_dunks":   [_event(e) for e in self.new_dunks],
            "scene_cut":   self.scene_cut,
            "ball_carrier_track_id": self.ball_carrier_track_id,
            "possession_team_id":    self.possession_team_id,
            "possession_distance_mode": self.possession_distance_mode,
            "homography_quality": self.homography_quality,
            "court_view": self.court_view,
        }


# ─────────────────────────── pipeline ────────────────────────────────────────

class BasketballPipeline:
    """
    Stateful per-frame basketball analytics pipeline.

    Owns all model weights, trackers and event detectors.
    Thread-safe to call sequentially; NOT safe to call from multiple threads.
    """

    # Player class names that count as players (vs refs / hoops)
    _PLAYER_KEYWORDS = ("player", "person")

    def __init__(self, config: PipelineConfig | None = None):
        self.cfg = config or PipelineConfig()
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load models
        self._court_model  = YOLO(self.cfg.court_model_path)
        self._player_model = YOLO(self.cfg.player_model_path)
        self._ball_model   = YOLO(self.cfg.ball_model_path)
        self._pose_model: Optional[YOLO] = (
            YOLO(self.cfg.player_pose_path) if self.cfg.enable_pose else None
        )

        # Optional CLIP team assigner (Phase 12)
        self._team_assigner = None
        # use_kmeans_teams implicitly enables team identification
        _team_id_enabled = self.cfg.enable_team_id or self.cfg.use_kmeans_teams
        if _team_id_enabled:
            if self.cfg.use_kmeans_teams:
                # K-means colour clustering — PRIMARY DEFAULT (no text prompts, learns from the video itself)
                # Works reliably on all jersey color combinations: gold/navy, white/dark, etc.
                # Outperforms CLIP on dark-on-dark combos (Wolves/Thunder).
                from analytics.team_color_kmeans import KMeansTeamAssigner
                self._team_assigner = KMeansTeamAssigner(
                    warmup_frames=self.cfg.kmeans_warmup_frames,
                    vote_window=self.cfg.kmeans_vote_window,
                    min_votes=self.cfg.kmeans_min_votes,
                    team1_is_light=self.cfg.kmeans_team1_is_light,
                    min_margin=self.cfg.kmeans_min_margin,
                )
                self._team_assigner.load_model()  # no-op but keeps the API uniform
                print("Using K-means colour clustering for team classification (PRIMARY DEFAULT)")
            elif self.cfg.use_openai_clip:
                # Phase 12f: Use OpenAI CLIP for better accuracy (85-95% vs 70-80%)
                from analytics.team_classifier_clip import CLIPTeamClassifier
                team1_prompts = self.cfg.team1_clip_prompts
                team2_prompts = self.cfg.team2_clip_prompts
                if team1_prompts is None:
                    # Generate default prompts from team descriptions
                    team1_prompts = [
                        f"a basketball player wearing a {self.cfg.team1_desc}",
                        f"player in {self.cfg.team1_desc}",
                        self.cfg.team1_desc,
                    ]
                if team2_prompts is None:
                    # Generate default prompts from team descriptions
                    team2_prompts = [
                        f"a basketball player wearing a {self.cfg.team2_desc}",
                        f"player in {self.cfg.team2_desc}",
                        self.cfg.team2_desc,
                    ]
                self._team_assigner = CLIPTeamClassifier(
                    team_1_prompts=team1_prompts,
                    team_2_prompts=team2_prompts,
                    model_name=self.cfg.clip_model_name,
                    revote_every=self.cfg.team_refresh_every,
                    min_margin=0.02,  # Phase 13a: Very aggressive threshold
                    min_votes=1,       # Phase 13a: Emit after just 1 vote
                )
                self._team_assigner.load_model()  # eager so first frame isn't extra slow
                print(f"Using OpenAI CLIP ({self.cfg.clip_model_name}) for team classification")
            else:
                # Phase 12: Use fashion-clip (original, still supported for backwards compatibility)
                from analytics.team_assigner import TeamAssigner
                self._team_assigner = TeamAssigner(
                    team_1_class_name=self.cfg.team1_desc,
                    team_2_class_name=self.cfg.team2_desc,
                    refresh_every=self.cfg.team_refresh_every,
                )
                self._team_assigner.load_model()  # eager so first frame isn't extra slow

        # Optional possession tracker (Phase 12b)
        self._possession_tracker = None
        if self.cfg.enable_possession:
            from analytics.possession import PossessionTracker
            self._possession_tracker = PossessionTracker()

        # Attempt torch.compile on Linux/WSL (Triton required)
        self._try_compile()

        # Derive player class ID sets from model labels
        names = self._player_model.model.names
        self.player_cls = {k for k, v in names.items() if v.lower() in self._PLAYER_KEYWORDS}

        # Stateful trackers
        # Ball tracking: simple picker + offline interpolation (Tarek's proven approach)
        # Kalman removed: it overcomplicates basketball's long occlusions (16+ frames)
        self._player_tracker = BotSort(
            reid_weights=pathlib.Path("osnet_x0_25_msmt17.pt"),
            device=self._device,
            half=False,
            with_reid=False,
            frame_rate=int(self.cfg.fps),
            track_buffer=30,
            track_high_thresh=self.cfg.player_conf,
            new_track_thresh=self.cfg.player_conf + 0.05,
        )

        # Event detectors
        self._shot_det:   Optional[OnlineShotDetector]   = (
            OnlineShotDetector(
                fps=self.cfg.fps,
                require_court_projection=self.cfg.shot_require_court_projection,
            ) if self.cfg.enable_events else None
        )
        self._screen_det: Optional[OnlineScreenDetector] = (
            OnlineScreenDetector(fps=self.cfg.fps) if self.cfg.enable_events else None
        )
        self._pnr_det:    Optional[OnlinePnRDetector]    = OnlinePnRDetector()    if self.cfg.enable_events else None
        self._dunk_det:   Optional[OnlineDunkDetector]   = OnlineDunkDetector()   if self.cfg.enable_events else None

        # Stub cache (Phase 12e) — optional per-frame YOLO output cache.
        # One StubCache per expensive op; all four share (video, stub_dir).
        self._stubs: dict = {}
        self._OP_COURT_POSE    = "court_pose"
        self._OP_PLAYER_DETECT = "player_detect"
        self._OP_BALL_DETECT   = "ball_detect"
        self._OP_PLAYER_POSE   = "player_pose"
        if self.cfg.stub_dir is not None and self.cfg.video_path is not None:
            from inference.stubs import build_all_caches
            self._stubs = build_all_caches(
                stub_dir=self.cfg.stub_dir,
                video_path=self.cfg.video_path,
                court_model_path=self.cfg.court_model_path,
                player_model_path=self.cfg.player_model_path,
                ball_model_path=self.cfg.ball_model_path,
                player_pose_path=self.cfg.player_pose_path,
            )

        # Scene-cut state
        self._prev_hist: Optional[np.ndarray] = None
        self._SCENE_CUT_THR = 0.60

        # Homography temporal cache — persists the last good H up to N frames
        self._cached_H: Optional[np.ndarray] = None
        self._H_age: int = 0                  # frames since last real H
        self._H_MAX_AGE: int = 30             # drop stale H after 1 second
        self._H_JUMP_PX: float = 80.0         # reject new H if corners shift >this many px

        # Homography temporal cache — persists the last good H up to N frames
        self._cached_H: Optional[np.ndarray] = None
        self._H_age: int = 0                  # frames since last real H
        self._H_MAX_AGE: int = 30             # drop stale H after 1 second
        self._H_JUMP_PX: float = 80.0         # reject new H if corners shift >this many px
        self._H_source: str = "none"          # 'current' | 'cached' | 'none'

        # Accumulated stats
        self.total_frames   = 0
        self.ball_frames    = 0
        self.coasted_frames = 0
        self.scene_cuts     = 0
        self.player_total   = 0
        self.non_court_frames = 0
        # Ball picker state: last accepted centre and its velocity, so the
        # motion-aware picker can score candidates by distance to the prediction.
        self._ball_last: Optional[Tuple[float, float]] = None
        self._ball_vel: Optional[Tuple[float, float]] = None
        self._ball_last_frame: Optional[int] = None

    # ── public API ────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> FrameResult:
        """
        Run the full pipeline on one BGR frame.

        Returns a FrameResult with all detections, tracks and events for this frame.
        Handles scene-cut detection and resets trackers automatically.
        """
        self.total_frames += 1
        H_mat = frame.shape[0]  # frame height in pixels
        W_mat = frame.shape[1]

        # --- Scene-cut detection ---
        scene_cut = False
        cur_hist = self._frame_hist(frame)
        if self._prev_hist is not None:
            corr = cv2.compareHist(self._prev_hist, cur_hist, cv2.HISTCMP_CORREL)
            if corr < self._SCENE_CUT_THR:
                self._reset_stateful()
                scene_cut = True
                self.scene_cuts += 1
        self._prev_hist = cur_hist

        # --- Court homography ---
        if self.cfg.fixed_homography is not None:
            H = np.asarray(self.cfg.fixed_homography, dtype=np.float64)
            kps = kpc = None
            self._H_source = "fixed"
            homography_quality = {"source": "fixed", "inlier_ratio": None, "rmse_ft": None}
        else:
            H, kps, kpc = self._build_H_cached(frame, frame_idx)
            homography_quality = self._homography_quality(H, kps, kpc)

        # --- Court-view gate ---
        # Broadcast footage is full of crowd shots, bench close-ups and replay
        # graphics. Without this the detector happily reports spectators as
        # players and assigns them a team. If the court cannot be located in this
        # frame, there is nothing on-court to report.
        #
        # Off by default: the labelled benchmark clips are close-up gym footage
        # where the court-keypoint model finds no homography at all, so gating
        # unconditionally would suppress every event and zero the scorecard.
        if not is_court_view(self.cfg.court_view_gate, H):
            self.non_court_frames += 1
            return FrameResult(
                frame_idx=frame_idx,
                H=None,
                court_kps=kps,
                court_kpc=kpc,
                players=[],
                ball=None,
                scene_cut=scene_cut,
                homography_quality=homography_quality,
                court_view=False,
            )

        # --- Player tracking ---
        players, players_court = self._run_player_tracking(frame, H, frame_idx)
        self.player_total += len(players)

        # --- Team identification (optional, Phase 12) ---
        if self._team_assigner is not None and players:
            tid_to_team = self._team_assigner.assign_frame(
                frame,
                frame_idx,
                [(p.track_id, p.box_xyxy) for p in players if p.cls_id in self.player_cls],
            )
            for p in players:
                if p.track_id in tid_to_team:
                    p.team_id = tid_to_team[p.track_id]

        # --- Player body pose (optional) ---
        if self._pose_model is not None and len(players) > 0:
            self._attach_pose(frame, players, frame_idx)

        # --- Ball detection + Kalman ---
        ball = self._run_ball(frame, H, H_mat, W_mat, frame_idx)

        # Pull pose arrays from players for shot detector compatibility
        kps_all  = np.stack([p.pose_kps  for p in players if p.pose_kps  is not None], axis=0) if any(p.pose_kps  is not None for p in players) else None
        kpc_all  = np.stack([p.pose_kpc  for p in players if p.pose_kpc  is not None], axis=0) if any(p.pose_kpc  is not None for p in players) else None
        pboxes   = np.array([(p.box_xyxy) for p in players if p.pose_kps is not None], dtype=np.float32) if any(p.pose_kps is not None for p in players) else None

        # BoT-SORT raw tracks array (N, 8) for shot detector
        tracks_arr = np.array([
            [p.box_xyxy[0], p.box_xyxy[1], p.box_xyxy[2], p.box_xyxy[3],
             p.track_id, p.conf, p.cls_id, 0]
            for p in players
        ], dtype=np.float32) if players else np.empty((0, 8), dtype=np.float32)

        # --- Events ---
        new_shots, new_screens, new_pnrs, new_dunks = [], [], [], []

        if self._screen_det is not None:
            new_screens = self._screen_det.update(
                frame_idx=frame_idx,
                players_court=players_court,
                ball_court=ball.court_pos if ball else None,
            )
            if self._pnr_det is not None:
                for sev in new_screens:
                    self._pnr_det.record_screen(sev)

        # Dunk detector — pixel-space lookback, fires when ball was near hoop & goes coast
        if self._dunk_det is not None and ball is not None:
            # Compute hoop pixel positions via inverse homography (if H available)
            hoop_pixels = []
            if H is not None:
                try:
                    H_inv = np.linalg.inv(H)
                    for cx, cy, side in [(5.25, 25.0, "left"), (88.75, 25.0, "right")]:
                        pt = np.array([[[cx, cy]]], dtype=np.float32)
                        px = cv2.perspectiveTransform(pt, H_inv)[0][0]
                        hoop_pixels.append(((float(px[0]), float(px[1])), side))
                except np.linalg.LinAlgError:
                    pass
            # All player centres in pixel space (no court_pos required)
            players_px = [
                ((p.box_xyxy[0] + p.box_xyxy[2]) / 2,
                 (p.box_xyxy[1] + p.box_xyxy[3]) / 2,
                 p.track_id,
                 p.court_pos)
                for p in players
            ]
            dunk_ev = self._dunk_det.update(
                frame_idx=frame_idx,
                coast_frames=ball.coast_frames,
                ball_cx=ball.cx,
                ball_cy=ball.cy,
                hoop_pixels=hoop_pixels,
                players_px=players_px,
            )
            if dunk_ev is not None:
                new_dunks.append(dunk_ev)

        if self._shot_det is not None and ball is not None and ball.tracked:
            shot_ev = self._shot_det.update(
                frame_idx=frame_idx,
                ball_cx=ball.cx,
                ball_cy=ball.cy,
                ball_tracked=ball.tracked,
                ball_coasted=ball.coasted,
                frame_h=H_mat,
                tracks=tracks_arr,
                H=H,
                pose_kps=kps_all,
                pose_kpc=kpc_all,
                pose_boxes=pboxes,
            )
            if shot_ev is not None:
                new_shots.append(shot_ev)
                if self._pnr_det is not None:
                    pnr_ev = self._pnr_det.check_shot(shot_ev)
                    if pnr_ev is not None:
                        new_pnrs.append(pnr_ev)

        # --- Possession tracking (Phase 12b) ---
        carrier_tid: Optional[int] = None
        possession_team: Optional[int] = None
        possession_distance_mode: Optional[str] = None
        if self._possession_tracker is not None:
            ball_court = ball.court_pos if (ball is not None and ball.court_pos is not None) else None
            ball_pixel = (ball.cx, ball.cy) if ball is not None else None
            ball_visible = bool(
                ball is not None
                and ball.tracked
                and not ball.coasted
                and not getattr(ball, "interpolated", False)
            )
            players_court_input = [
                (p.track_id, p.court_pos, p.team_id)
                for p in players if p.cls_id in self.player_cls and p.court_pos is not None
            ]
            players_pixel_input = [
                (
                    p.track_id,
                    ((p.box_xyxy[0] + p.box_xyxy[2]) / 2.0,
                     (p.box_xyxy[1] + p.box_xyxy[3]) / 2.0),
                    p.team_id,
                )
                for p in players if p.cls_id in self.player_cls
            ]
            carrier_tid, possession_team = self._possession_tracker.update(
                ball_court=ball_court,
                ball_pixel=ball_pixel,
                players_court=players_court_input,
                players_pixel=players_pixel_input,
                ball_visible=ball_visible,
            )
            possession_distance_mode = self._possession_tracker.state.distance_mode

        return FrameResult(
            frame_idx=frame_idx,
            H=H,
            court_kps=kps,
            court_kpc=kpc,
            players=players,
            ball=ball,
            new_shots=new_shots,
            new_screens=new_screens,
            new_pnrs=new_pnrs,
            new_dunks=new_dunks,
            scene_cut=scene_cut,
            ball_carrier_track_id=carrier_tid,
            possession_team_id=possession_team,
            possession_distance_mode=possession_distance_mode,
            homography_quality=homography_quality,
        )

    def finalize(self) -> dict:
        """Close out per-clip state and report clip-level availability flags.

        Called by `inference/run_video.py` after the last frame.  Keeps a
        silently-failed team warmup from looking like a successful run (B-35).
        """
        info: dict = {}
        if self._team_assigner is not None:
            finalize = getattr(self._team_assigner, "finalize", None)
            if callable(finalize):
                info["team_assignment_available"] = bool(finalize())
            else:
                info["team_assignment_available"] = True
            diagnostics = getattr(self._team_assigner, "diagnostics", None)
            if callable(diagnostics):
                info["team_assignment"] = diagnostics()
        return info

    @property
    def team_assignment_available(self) -> Optional[bool]:
        if self._team_assigner is None:
            return None
        return bool(getattr(self._team_assigner, "team_assignment_available", True))

    def reset(self):
        """Full pipeline reset (e.g. new video)."""
        self._reset_stateful()
        self._prev_hist = None
        self.total_frames = self.ball_frames = self.coasted_frames = 0
        self.scene_cuts   = self.player_total = self.non_court_frames = 0

    def flush_stubs(self):
        """Phase 12e: write any accumulated stub-cache updates to disk. Safe no-op
        if stubs are not enabled. Call once at end of video."""
        for cache in self._stubs.values():
            cache.flush()

    def print_summary(self):
        t = self.total_frames or 1
        print(f"Frames processed:   {self.total_frames}")
        print(f"Ball detected:      {self.ball_frames}/{t} ({100*self.ball_frames/t:.1f}%)")
        print(f"Ball coasted:       {self.coasted_frames}/{t} ({100*self.coasted_frames/t:.1f}%)")
        print(f"Ball tracked total: {self.ball_frames+self.coasted_frames}/{t} "
              f"({100*(self.ball_frames+self.coasted_frames)/t:.1f}%)")
        print(f"Scene cuts:         {self.scene_cuts}")
        if self.cfg.court_view_gate:
            print(f"Non-court frames:   {self.non_court_frames}/{t} "
                  f"({100*self.non_court_frames/t:.1f}%) rejected by court-view gate")
        print(f"Avg players/frame:  {self.player_total/t:.1f}")
        if self._shot_det is not None:
            summary = self._shot_det.summary()
            offline = getattr(self, "offline_shot_count", None)
            if offline is not None:
                # The online detector's count is misleading when --offline-events
                # replayed the smoothed track; report the authoritative number.
                summary = (f"{offline} shot(s) after offline re-detection "
                           f"(online pass found: {summary})")
            print(f"Shot events:        {summary}")
        if self._screen_det is not None:
            print(f"Screen events:      {self._screen_det.summary()}")
        if self._pnr_det is not None:
            print(f"PnR events:         {self._pnr_det.summary()}")
        if self._dunk_det is not None:
            print(f"Dunk events:        {self._dunk_det.summary()}")

    @property
    def shot_events(self):
        return self._shot_det.events if self._shot_det else []

    @property
    def screen_events(self):
        return self._screen_det.events if self._screen_det else []

    @property
    def pnr_events(self):
        return self._pnr_det.events if self._pnr_det else []

    @property
    def dunk_events(self):
        return self._dunk_det.events if self._dunk_det else []

    # ── private helpers ───────────────────────────────────────────────────────

    def _reset_stateful(self):
        self._player_tracker.reset()
        if self._shot_det:   self._shot_det.reset()
        if self._screen_det: self._screen_det.reset()
        if self._pnr_det:    self._pnr_det.reset()
        if self._possession_tracker: self._possession_tracker.reset()
        self._cached_H = None
        self._H_age = 0
        self._H_source = "none"

    def _try_compile(self):
        try:
            import triton  # noqa: F401
            models = [self._court_model, self._player_model, self._ball_model]
            if self._pose_model:
                models.append(self._pose_model)
            for m in models:
                m.model.forward = torch.compile(m.model.forward, mode="reduce-overhead")
        except (ImportError, Exception):
            pass

    @staticmethod
    def _frame_hist(frame: np.ndarray) -> np.ndarray:
        hists = [cv2.calcHist([frame], [c], None, [64], [0, 256]) for c in range(3)]
        h = np.concatenate(hists)
        cv2.normalize(h, h)
        return h

    def _build_H_cached(self, frame: np.ndarray, frame_idx: int = 0):
        """Build H with temporal caching, outlier-rejection, and age-based fallback.

        Returns (H, kps, kpc) where H is the best available estimate, possibly
        from a recent previous frame if the current frame's estimate is noisy.
        """
        H_new, kps, kpc = self._build_H(frame, frame_idx)

        if H_new is not None:
            accept = True
            if self._cached_H is not None:
                # Reject if projected court corners jump too far in pixel space
                # (guards against noisy one-frame keypoint estimates)
                try:
                    H_inv_new = np.linalg.inv(H_new)
                    H_inv_old = np.linalg.inv(self._cached_H)
                    test_court_pts = np.float32([
                        [[0.0, 0.0]], [[94.0, 0.0]],
                        [[0.0, 50.0]], [[94.0, 50.0]],
                        [[5.25, 25.0]], [[88.75, 25.0]],
                    ])
                    px_new = cv2.perspectiveTransform(test_court_pts, H_inv_new)
                    px_old = cv2.perspectiveTransform(test_court_pts, H_inv_old)
                    max_shift = float(np.max(
                        np.hypot(px_new[:, 0, 0] - px_old[:, 0, 0],
                                 px_new[:, 0, 1] - px_old[:, 0, 1])
                    ))
                    if max_shift > self._H_JUMP_PX:
                        accept = False  # outlier — keep using cached
                except np.linalg.LinAlgError:
                    accept = False

            if accept:
                self._cached_H = H_new
                self._H_age = 0

        # Fallback: use cached H if available and not too stale
        if self._cached_H is not None and self._H_age <= self._H_MAX_AGE:
            self._H_age += 1
            self._H_source = "current" if H_new is not None and H_new is self._cached_H else "cached"
            return self._cached_H, kps, kpc

        self._H_source = "current" if H_new is not None else "none"
        return H_new, kps, kpc

    def _build_H_cached_DUPLICATE_UNUSED(self, frame: np.ndarray, frame_idx: int = 0):
        """(Kept only to preserve pre-existing duplicate; overridden by method above.)"""
        H_new, kps, kpc = self._build_H(frame, frame_idx)

        if H_new is not None:
            accept = True
            if self._cached_H is not None:
                # Reject if projected court corners jump too far in pixel space
                # (guards against noisy one-frame keypoint estimates)
                try:
                    H_inv_new = np.linalg.inv(H_new)
                    H_inv_old = np.linalg.inv(self._cached_H)
                    test_court_pts = np.float32([
                        [[0.0, 0.0]], [[94.0, 0.0]],
                        [[0.0, 50.0]], [[94.0, 50.0]],
                        [[5.25, 25.0]], [[88.75, 25.0]],
                    ])
                    px_new = cv2.perspectiveTransform(test_court_pts, H_inv_new)
                    px_old = cv2.perspectiveTransform(test_court_pts, H_inv_old)
                    max_shift = float(np.max(
                        np.hypot(px_new[:, 0, 0] - px_old[:, 0, 0],
                                 px_new[:, 0, 1] - px_old[:, 0, 1])
                    ))
                    if max_shift > self._H_JUMP_PX:
                        accept = False  # outlier — keep using cached
                except np.linalg.LinAlgError:
                    accept = False

            if accept:
                self._cached_H = H_new
                self._H_age = 0

        # Fallback: use cached H if available and not too stale
        if self._cached_H is not None and self._H_age <= self._H_MAX_AGE:
            self._H_age += 1
            return self._cached_H, kps, kpc

        return H_new, kps, kpc

    def _build_H(self, frame: np.ndarray, frame_idx: int = 0):
        """Returns (H, kps, kpc) — H may be None if estimation fails."""
        h, w = frame.shape[:2]

        # Phase 12e: stub-cached YOLO call
        cache = self._stubs.get(self._OP_COURT_POSE) if self._stubs else None
        if cache is not None and cache.hit(frame_idx):
            payload = cache.get(frame_idx)
            if payload is None:
                return None, None, None
            kp = payload["xy"]
            cf = payload["conf"]
        else:
            res = self._court_model(frame, conf=self.cfg.court_kp_conf, verbose=False)
            if not res or res[0].keypoints is None or len(res[0].keypoints.xy) == 0:
                if cache is not None:
                    cache.put(frame_idx, None)
                return None, None, None
            kp = res[0].keypoints.xy[0].cpu().numpy()
            cf = res[0].keypoints.conf[0].cpu().numpy()
            if cache is not None:
                cache.put(frame_idx, {"xy": kp, "conf": cf})
        mask = (cf >= self.cfg.court_kp_conf) & (kp[:, 0] > 1) & (kp[:, 0] < w - 1) & (kp[:, 1] > 1) & (kp[:, 1] < h - 1)
        # Need ≥6 reliable correspondences for a stable RANSAC fit. With only 4
        # near-collinear points (e.g. all on one baseline) H is highly unstable.
        if mask.sum() < 6:
            return None, kp, cf
        src = kp[mask].astype(np.float32)
        dst = COURT_KPS_FT[mask].astype(np.float32)
        centered = dst - dst.mean(axis=0)
        _, sv, _ = np.linalg.svd(centered)
        if sv[0] < 1e-6 or sv[1] / sv[0] < 0.01:
            return None, kp, cf
        # RANSAC with 5 ft reprojection threshold (court-coord units).
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or inliers is None or int(inliers.sum()) < 4:
            return None, kp, cf
        det = abs(np.linalg.det(H[:2, :2]))
        if det < 1e-6 or det > 1e10:
            return None, kp, cf
        return H, kp, cf

    def _homography_quality(self, H, kps, kpc) -> dict:
        """Summarize how well H fits this frame's court keypoints (B-32).

        Error is measured in court feet: project the detected pixel keypoints
        through H and compare with the known court coordinates.  The 5 ft
        threshold matches the RANSAC reprojection threshold in `_build_H`.
        """
        quality = {
            "source": self._H_source,
            "inlier_ratio": None,
            "rmse_ft": None,
        }
        if H is None or kps is None or kpc is None:
            return quality

        mask = np.asarray(kpc) >= self.cfg.court_kp_conf
        if not mask.any():
            return quality

        src = np.asarray(kps)[mask].astype(np.float32).reshape(-1, 1, 2)
        dst = COURT_KPS_FT[mask].astype(np.float32)
        try:
            projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        except cv2.error:
            return quality
        err = np.linalg.norm(projected - dst, axis=1)
        inliers = err <= 5.0
        quality["inlier_ratio"] = float(inliers.mean())
        if inliers.any():
            quality["rmse_ft"] = float(np.sqrt(np.mean(err[inliers] ** 2)))
        return quality

    @staticmethod
    def project_to_court(H, px: float, py: float) -> Optional[Tuple[float, float]]:
        if H is None:
            return None
        ph = np.array([px, py, 1.0])
        ch3 = H @ ph
        if abs(ch3[2]) < 1e-6:
            return None
        x = ch3[0] / ch3[2]
        y = ch3[1] / ch3[2]
        xc = float(np.clip(x, 0.0, 94.0))
        yc = float(np.clip(y, 0.0, 50.0))
        if abs(x - xc) > 10.0 or abs(y - yc) > 10.0:
            return None
        return xc, yc

    def _run_player_tracking(self, frame, H, frame_idx: int = 0):
        # Phase 12e: stub-cached raw detections
        cache = self._stubs.get(self._OP_PLAYER_DETECT) if self._stubs else None
        if cache is not None and cache.hit(frame_idx):
            dets_np = cache.get(frame_idx)
        else:
            player_res = self._player_model(frame, conf=self.cfg.player_conf, verbose=False)
            raw_boxes = []
            if player_res and player_res[0].boxes is not None:
                for box in player_res[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(float)
                    raw_boxes.append([x1, y1, x2, y2, float(box.conf.cpu()), float(box.cls.cpu())])
            dets_np = np.array(raw_boxes, dtype=np.float32) if raw_boxes else np.empty((0, 6), dtype=np.float32)
            if cache is not None:
                cache.put(frame_idx, dets_np)

        tracks = self._player_tracker.update(dets_np, frame)

        players: List[PlayerTrack] = []
        players_court: List[Tuple] = []

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            track_id = int(t[4])
            conf     = float(t[5])
            cls_id   = int(t[6])

            court_pos = None
            if cls_id in self.player_cls:
                court_pos = self.project_to_court(H, (x1 + x2) / 2, y2)
                if court_pos and 0 <= court_pos[0] <= 94 and 0 <= court_pos[1] <= 50:
                    players_court.append((*court_pos, track_id))
                else:
                    court_pos = None

            players.append(PlayerTrack(
                track_id=track_id,
                box_xyxy=(x1, y1, x2, y2),
                conf=conf,
                cls_id=cls_id,
                court_pos=court_pos,
            ))

        return players, players_court

    def _attach_pose(self, frame, players: List[PlayerTrack], frame_idx: int = 0):
        # Phase 12e: stub-cached raw pose outputs
        cache = self._stubs.get(self._OP_PLAYER_POSE) if self._stubs else None
        if cache is not None and cache.hit(frame_idx):
            payload = cache.get(frame_idx)
            if payload is None:
                return
            kps_all = payload["kps"]
            kpc_all = payload["kpc"]
            pboxes  = payload["boxes"]
        else:
            pose_res = self._pose_model(frame, conf=self.cfg.player_pose_conf, verbose=False)
            if not pose_res or pose_res[0].keypoints is None or len(pose_res[0].keypoints.xy) == 0:
                if cache is not None:
                    cache.put(frame_idx, None)
                return
            kps_all = pose_res[0].keypoints.xy.cpu().numpy()
            kpc_all = pose_res[0].keypoints.conf.cpu().numpy()
            pboxes  = pose_res[0].boxes.xyxy.cpu().numpy()
            if cache is not None:
                cache.put(frame_idx, {"kps": kps_all, "kpc": kpc_all, "boxes": pboxes})

        for p in players:
            tx1, ty1, tx2, ty2 = p.box_xyxy
            best_m, best_iou = -1, 0.25
            for m in range(len(pboxes)):
                px1, py1, px2, py2 = pboxes[m]
                ix1 = max(tx1, px1); iy1 = max(ty1, py1)
                ix2 = min(tx2, px2); iy2 = min(ty2, py2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                ua = ((tx2 - tx1) * (ty2 - ty1) + (px2 - px1) * (py2 - py1) - inter)
                iou = inter / ua if ua > 0 else 0.0
                if iou > best_iou:
                    best_iou = iou; best_m = m
            if best_m >= 0:
                p.pose_kps = kps_all[best_m]
                p.pose_kpc = kpc_all[best_m]

    def _run_ball(self, frame, H, frame_h, frame_w, frame_idx: int = 0) -> Optional[BallState]:
        # Phase 12e: stub-cached raw ball detections as (N,6) [x1,y1,x2,y2,conf,cls]
        cache = self._stubs.get(self._OP_BALL_DETECT) if self._stubs else None
        if cache is not None and cache.hit(frame_idx):
            dets = cache.get(frame_idx)
        else:
            ball_res = self._ball_model(
                frame,
                conf=self.cfg.ball_conf,
                imgsz=self.cfg.ball_imgsz,
                classes=[self.cfg.ball_class_id],
                verbose=False,
            )
            rows = []
            if ball_res and ball_res[0].boxes is not None and len(ball_res[0].boxes):
                boxes = ball_res[0].boxes
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy()
                for idx in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[idx]
                    rows.append([float(x1), float(y1), float(x2), float(y2),
                                 float(confs[idx]), float(clss[idx])])
            dets = np.array(rows, dtype=np.float32) if rows else np.empty((0, 6), dtype=np.float32)
            if cache is not None:
                cache.put(frame_idx, dets)

        # ── Ball picker ──
        # "motion" continues the existing track; "argmax" is the original
        # highest-confidence-per-frame pick. Argmax has no memory, so a confident
        # false positive steals the frame and the motion gate then discards it:
        # measured coverage 38% vs 67% over a 150-frame window, with the 90th
        # percentile jump falling from 543 px to 114 px. Precision is not measured
        # (no box-level ball labels yet), so the picker stays switchable.
        if self.cfg.ball_picker == "motion":
            pick = SimpleBallTracker.pick_motion_aware(
                dets,
                self.cfg.ball_class_id,
                last=self._ball_last,
                velocity=self._ball_vel,
            )
        else:
            pick = SimpleBallTracker.pick_best(dets, self.cfg.ball_class_id)
        if pick is None:
            return None

        bx1, by1, bx2, by2, best_conf = pick
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        if self._ball_last is not None:
            prev = self._ball_last_frame if self._ball_last_frame is not None else frame_idx - 1
            elapsed = max(1, frame_idx - prev)
            self._ball_vel = ((cx - self._ball_last[0]) / elapsed,
                              (cy - self._ball_last[1]) / elapsed)
        self._ball_last = (cx, cy)
        self._ball_last_frame = frame_idx
        best_box = (int(bx1), int(by1), int(bx2), int(by2))
        self.ball_frames += 1
        court_pos = self.project_to_court(H, cx, cy) if H is not None else None
        return BallState(
            tracked=True, coasted=False, cx=cx, cy=cy, coast_frames=0,
            court_pos=court_pos, det_box=best_box, det_conf=best_conf,
        )
