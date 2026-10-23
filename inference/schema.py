"""
inference/schema.py â€” versioned `ClipResult` pydantic schema (Phase 15a).

Purpose
-------
The existing `FrameResult.to_dict()` in `inference/pipeline.py` produces a
JSON-serialisable dict, and `inference/run_video.py --json-out` writes a
`{video, fps, width, height, frames: [...]}` envelope around it. That was fine
for quick debugging but it is not a stable contract. Phase 15 (eval harness),
Phase 16 (Coach's Cutups), and Phase 17 (Scout Report PDF) all need to read
this JSON and rely on its structure.

This module:

1. Defines pydantic v2 models mirroring the current `FrameResult` dict, with
   one strict addition â€” clip-level aggregated event lists (`shots`, `screens`,
   `pnrs`, `dunks`, `passes`, `interceptions`) at the top level rather than
   hidden inside per-frame `new_*` streams.
2. Stamps every clip with `schema_version`, `git_sha`, `created_at` and the
   effective `pipeline_config` so any downstream reader can detect stale /
   incompatible inputs.
3. Provides a single helper `ClipResult.from_pipeline_run(...)` used by
   `inference/run_video.py` to build the object without callers needing to
   know the field layout.

Design notes
------------
- Numpy arrays â†’ lists (JSON-native). Any reader that wants ndarrays should
  `np.asarray(field)` itself.
- We keep the per-frame representation *close* to the existing dict so we
  don't have to migrate any downstream consumer in one go. Clip-level event
  aggregates are additive.
- `model_config = ConfigDict(extra="forbid")` on every model so typos in
  hand-written JSON (e.g. labels, replay stubs) fail loudly.

v1.1.0 (additive) - evaluation/health fields
- `FrameResultModel.homography_quality`, `possession_distance_mode`, and
  `ClipResult.team_assignment_available`.

v1.2.0 (additive) - state layer
--------------------------------
- `GameState`, `Possession`, and `Play` are optional clip-level state objects.
- Event models carry optional `possession_id` / `play_id` links.
- Existing clips remain valid; fields default to unknown/empty.

v1.3.0 (additive) - lineup intervals
------------------------------------
- `LineupInterval` records a five-player unit over a frame interval.
- `ClipResult.lineups` is optional and defaults to empty.
"""
from __future__ import annotations

import datetime as _dt
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

# Bump on breaking changes. Follow semver-ish:
#   major  â€” field removed / meaning changed
#   minor  â€” field added, optional with default
#   patch  â€” doc / validation tightening only
SCHEMA_VERSION = "1.3.0"


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ primitives â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlayerState(_Model):
    track_id: int
    box_xyxy: Tuple[int, int, int, int]
    conf: float
    cls_id: int
    court_pos: Optional[Tuple[float, float]] = None
    pose_kps: Optional[List[List[float]]] = None  # (17, 2) or None
    pose_kpc: Optional[List[float]] = None        # (17,) or None
    team_id: Optional[int] = None
    # Phase 12d â€” speed & distance analytics (two-pass only; 0 if unavailable)
    speed_ft_per_s: float = 0.0
    distance_traveled_ft: float = 0.0


class BallState(_Model):
    tracked: bool
    coasted: bool
    cx: float
    cy: float
    coast_frames: int
    court_pos: Optional[Tuple[float, float]] = None
    det_box: Optional[Tuple[int, int, int, int]] = None
    det_conf: float = 0.0
    interpolated: bool = False


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ events â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Mirror of the event dataclasses in events/*.py. Kept as flat models here so
# the schema is readable without cross-module imports.

class ShotEventModel(_Model):
    frame_start: int
    ball_px: Tuple[float, float]
    ball_court_ft: Optional[Tuple[float, float]] = None
    shot_type: Optional[str] = None          # 'layup' | 'mid_range' | '3_pointer'
    shot_distance_ft: Optional[float] = None
    shooter_track_id: Optional[int] = None
    confirmed_by_pose: bool = False
    confidence: float = 0.0
    possession_id: Optional[int] = None
    play_id: Optional[int] = None


class ScreenEventModel(_Model):
    frame: int
    screener_id: int
    cutter_id: int
    screen_type: str                          # 'on_ball' | 'off_ball'
    screener_pos_ft: Tuple[float, float]
    cutter_pos_ft: Tuple[float, float]
    separation_ft: float
    confidence: float
    possession_id: Optional[int] = None
    play_id: Optional[int] = None


class PnREventModel(_Model):
    frame: int
    screen_frame: int
    shot_frame: int
    screener_id: int
    ball_handler_id: int
    shooter_id: Optional[int] = None
    outcome: str                              # 'pull_up' | 'roll_shot' | 'unknown'
    shot_type: Optional[str] = None
    shot_distance_ft: Optional[float] = None
    separation_ft: float
    confidence: float
    possession_id: Optional[int] = None
    play_id: Optional[int] = None


class DunkEventModel(_Model):
    frame: int
    player_track_id: int
    hoop_side: str                            # 'left' | 'right'
    ball_hoop_dist_px: float
    player_hoop_dist_px: float
    player_court_ft: Optional[Tuple[float, float]] = None
    possession_id: Optional[int] = None
    play_id: Optional[int] = None


class PassEventModel(_Model):
    frame_idx: int
    team_id: int
    from_track_id: int
    to_track_id: int
    possession_id: Optional[int] = None
    play_id: Optional[int] = None


class InterceptionEventModel(_Model):
    frame_idx: int
    team_id: int
    from_track_id: int
    to_track_id: int
    possession_id: Optional[int] = None
    play_id: Optional[int] = None


# -- state layer (v1.2) -----------------------------------------------------

class GameState(_Model):
    """Game context attached to a clip.

    Empty lineups mean unknown, not zero players. The state layer is additive:
    old clips without this object remain valid.
    """

    period: int = Field(ge=1)
    clock_s: Optional[float] = Field(default=None, ge=0.0)
    score_home: int = Field(default=0, ge=0)
    score_away: int = Field(default=0, ge=0)
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    possession_team_id: Optional[int] = None
    lineup_home: List[int] = Field(default_factory=list)
    lineup_away: List[int] = Field(default_factory=list)
    ruleset: str = "unknown"       # nba_2025_26 | fiba_2026 | nfhs | ncaa | unknown
    footage_type: str = "unknown"  # nba_broadcast | fiba_broadcast | fixed_gym | handheld | unknown
    camera_spec: Optional[dict] = None


class Possession(_Model):
    possession_id: int
    offense_team_id: int
    defense_team_id: int
    start_frame: int
    end_frame: Optional[int] = None
    points: int = 0
    end_reason: Optional[str] = None   # made_shot | defensive_rebound | turnover | period_end | game_end | other
    confidence: float = 0.0


class Play(_Model):
    play_id: int
    possession_id: int
    play_type: str
    coverage: Optional[str] = None
    family: Optional[str] = None
    confidence: float = 0.0
    evidence_frames: List[int] = Field(default_factory=list)


class LineupInterval(_Model):
    """A five-player unit active over a frame interval."""
    team_id: int
    start_frame: int
    end_frame: Optional[int] = None
    player_ids: List[int] = Field(default_factory=list)
    source: str = "unknown"
    confidence: float = 0.0


class HomographyQuality(_Model):
    """How much to trust this frame's court mapping (B-32)."""

    source: str = "none"                      # 'current' | 'cached' | 'none'
    inlier_ratio: Optional[float] = None      # fraction of keypoints within 5 ft
    rmse_ft: Optional[float] = None           # RMSE over inliers, court feet


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ per-frame â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FrameResultModel(_Model):
    frame_idx: int
    # 3Ã—3 homography flattened to a 3Ã—3 list of lists, or None if this frame
    # failed the collinearity / keypoint-count gates.
    H: Optional[List[List[float]]] = None
    court_kps: Optional[List[List[float]]] = None   # (18, 2) or None
    court_kpc: Optional[List[float]] = None         # (18,)  or None
    players: List[PlayerState] = Field(default_factory=list)
    ball: Optional[BallState] = None

    # Events *fired this frame*. These are the transient deltas the live
    # detectors emit. The canonical, de-duplicated lists live at clip level
    # on ClipResult (see `shots`, `screens`, `pnrs`, `dunks`).
    new_shots:   List[ShotEventModel]   = Field(default_factory=list)
    new_screens: List[ScreenEventModel] = Field(default_factory=list)
    new_pnrs:    List[PnREventModel]    = Field(default_factory=list)
    new_dunks:   List[DunkEventModel]   = Field(default_factory=list)

    scene_cut: bool = False

    # Phase 12b â€” possession
    ball_carrier_track_id: Optional[int] = None
    possession_team_id: Optional[int] = None
    possession_distance_mode: Optional[str] = None   # 'court' | 'pixel' | None

    # B-32 â€” upstream health of the court mapping this frame
    homography_quality: Optional[HomographyQuality] = None

    # False when the court-view gate rejected this frame (crowd shot, close-up,
    # replay graphic). Such frames carry no players, ball or events.
    court_view: bool = True


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ clip-level â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class ClipMetadata(_Model):
    video: str
    fps: float
    width: int
    height: int
    frame_count: int
    pipeline_config: dict = Field(default_factory=dict)   # snapshot of PipelineConfig
    git_sha: Optional[str] = None
    created_at: str = Field(default_factory=lambda: _dt.datetime.utcnow().isoformat(timespec="seconds"))


class ClipResult(_Model):
    """Full pipeline output for one clip. The stable I/O contract for eval,
    cutups, and scout-report downstream consumers."""

    schema_version: str = SCHEMA_VERSION
    metadata: ClipMetadata
    frames: List[FrameResultModel] = Field(default_factory=list)

    # Canonical clip-level event lists (de-duplicated across frame streams).
    shots:         List[ShotEventModel]        = Field(default_factory=list)
    screens:       List[ScreenEventModel]      = Field(default_factory=list)
    pnrs:          List[PnREventModel]         = Field(default_factory=list)
    dunks:         List[DunkEventModel]        = Field(default_factory=list)
    passes:        List[PassEventModel]        = Field(default_factory=list)
    interceptions: List[InterceptionEventModel] = Field(default_factory=list)

    # -- state layer (v1.2) -------------------------------------------------
    # Optional game context. Empty/None means unknown; never assume a value.
    game_state: Optional[GameState] = None
    possessions: List[Possession] = Field(default_factory=list)
    plays: List[Play] = Field(default_factory=list)
    lineups: List[LineupInterval] = Field(default_factory=list)

    # B-35 â€” None = team identification disabled; False = enabled but warmup
    # never produced clusters (no team IDs are present anywhere in `frames`).
    team_assignment_available: Optional[bool] = None

    # â”€â”€ constructors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    @classmethod
    def from_pipeline_run(
        cls,
        *,
        video: str,
        fps: float,
        width: int,
        height: int,
        frame_results: Sequence[Any],              # list of FrameResult dataclasses
        pipeline_config: dict | None = None,
        pass_interception_result: Any | None = None,  # analytics.passes.PassInterceptionResult | None
        team_assignment_available: bool | None = None,
        game_state: GameState | None = None,
        possessions: Sequence[Possession] | None = None,
        plays: Sequence[Play] | None = None,
        lineups: Sequence[LineupInterval] | None = None,
    ) -> "ClipResult":
        """Build a ClipResult from an in-memory pipeline run.

        `frame_results` is the `List[FrameResult]` produced by
        `BasketballPipeline.process_frame(...)` in two-pass mode. Each item is
        serialised via its existing `.to_dict()` method, so we inherit all
        ndarray-to-list conversion already implemented there.
        """
        frame_dicts = [fr.to_dict() for fr in frame_results]
        frames = [FrameResultModel.model_validate(d) for d in frame_dicts]

        # Aggregate clip-level events by walking the per-frame deltas.
        shots   = [ev for f in frames for ev in f.new_shots]
        screens = [ev for f in frames for ev in f.new_screens]
        pnrs    = [ev for f in frames for ev in f.new_pnrs]
        dunks   = [ev for f in frames for ev in f.new_dunks]

        passes: List[PassEventModel] = []
        interceptions: List[InterceptionEventModel] = []
        if pass_interception_result is not None:
            passes = [
                PassEventModel(
                    frame_idx=p.frame_idx,
                    team_id=p.team_id,
                    from_track_id=p.from_track_id,
                    to_track_id=p.to_track_id,
                ) for p in pass_interception_result.passes
            ]
            interceptions = [
                InterceptionEventModel(
                    frame_idx=i.frame_idx,
                    team_id=i.team_id,
                    from_track_id=i.from_track_id,
                    to_track_id=i.to_track_id,
                ) for i in pass_interception_result.interceptions
            ]

        metadata = ClipMetadata(
            video=video,
            fps=float(fps),
            width=int(width),
            height=int(height),
            frame_count=len(frames),
            pipeline_config=pipeline_config or {},
            git_sha=_current_git_sha(),
        )

        return cls(
            metadata=metadata,
            frames=frames,
            shots=shots,
            screens=screens,
            pnrs=pnrs,
            dunks=dunks,
            passes=passes,
            interceptions=interceptions,
            game_state=game_state,
            possessions=list(possessions or []),
            plays=list(plays or []),
            lineups=list(lineups or []),
            team_assignment_available=team_assignment_available,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ClipResult":
        """Load a ClipResult from a JSON file previously written by `to_json`.

        Also accepts the legacy `{video, fps, width, height, frames}` envelope
        produced by older `inference/run_video.py --json-out` runs â€” those are
        upgraded in-memory to the current schema (no events, no git_sha)."""
        with open(path, "r") as f:
            data = json.load(f)

        if "schema_version" in data:
            return cls.model_validate(data)

        # Legacy upgrade path.
        return cls(
            metadata=ClipMetadata(
                video=data.get("video", str(path)),
                fps=float(data.get("fps", 30.0)),
                width=int(data.get("width", 0)),
                height=int(data.get("height", 0)),
                frame_count=len(data.get("frames", [])),
            ),
            frames=[FrameResultModel.model_validate(f) for f in data.get("frames", [])],
        )

    # â”€â”€ serialisation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def to_json(self, path: str | Path, *, indent: int | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=indent))


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _current_git_sha() -> Optional[str]:
    """Best-effort `git rev-parse HEAD`. Returns None if git is unavailable
    (CI sandbox, shipped app) rather than blowing up the pipeline."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


__all__ = [
    "SCHEMA_VERSION",
    "PlayerState",
    "BallState",
    "ShotEventModel",
    "ScreenEventModel",
    "PnREventModel",
    "DunkEventModel",
    "PassEventModel",
    "InterceptionEventModel",
    "GameState",
    "Possession",
    "Play",
    "LineupInterval",
    "HomographyQuality",
    "FrameResultModel",
    "ClipMetadata",
    "ClipResult",
]
