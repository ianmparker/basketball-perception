"""
eval/ground_truth.py — Ground truth annotation schema for Phase 15 eval harness.

Defines Pydantic models for storing labeled ground truth events (shots, passes, PnRs, etc.)
and team/tracking ground truth for evaluation against pipeline outputs.

Phase 15: Quantified evaluation starts here.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum
import json
from pathlib import Path


class ShotType(str, Enum):
    """Basketball shot types."""
    TWO_POINT = "two_point"
    THREE_POINT = "three_point"
    LAYUP = "layup"
    DUNK = "dunk"
    FREE_THROW = "free_throw"
    OTHER = "other"


class ShotOutcome(str, Enum):
    """Shot outcome."""
    MADE = "made"
    MISSED = "missed"
    BLOCKED = "blocked"
    # The labeler could not see the result (cut, camera angle, occlusion).
    # Recording this beats guessing: outcome metrics skip unknown shots.
    UNKNOWN = "unknown"


class ScreenType(str, Enum):
    """Screen types."""
    ON_BALL = "on_ball"
    OFF_BALL = "off_ball"


@dataclass
class ShotAnnotation:
    """Ground truth for a detected shot."""
    frame_idx: int  # Frame where shot release occurs
    track_id: int  # Track ID of shooter
    shot_type: ShotType = ShotType.OTHER
    outcome: ShotOutcome = ShotOutcome.MISSED
    shot_distance_ft: Optional[float] = None  # Estimated distance (for reference)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PassAnnotation:
    """Ground truth for a detected pass."""
    frame_start: int  # Frame where pass begins (passer has ball)
    frame_end: int  # Frame where receiver gains possession
    from_track_id: int  # Passer track ID
    to_track_id: int  # Receiver track ID
    team_id: Optional[int] = None  # Passing team
    outcome: str = "complete"  # "complete", "turnover", "intercepted"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ScreenAnnotation:
    """Ground truth for a detected screen."""
    frame_idx: int  # Frame where screen occurs
    screener_id: int  # Track ID of screener
    cutter_id: int  # Track ID of cutter
    screen_type: ScreenType = ScreenType.ON_BALL
    team_id: Optional[int] = None  # Screening team
    
    def to_dict(self):
        return asdict(self)


@dataclass
class PnRAnnotation:
    """Ground truth for a pick-and-roll."""
    frame_screen: int  # Frame of screen
    frame_roll: int  # Frame of roll/separation (if detected)
    screener_id: int  # Picker track ID
    ball_handler_id: int  # Ball handler track ID
    pnr_type: str = "drag"  # "drag", "middle", "side"
    team_id: Optional[int] = None  # PnR team
    outcome: str = "incomplete"  # "shot", "pass", "TO", "incomplete"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class BlockAnnotation:
    """Ground truth for a blocked shot."""
    frame_shot: int  # Frame where shot occurs
    frame_block: int  # Frame where block contact happens
    shooter_id: int  # Track ID of shooter
    blocker_id: int  # Track ID of blocker
    block_type: str = "contested"  # "contested", "clean"
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ReboundAnnotation:
    """Ground truth for a rebound."""
    frame_idx: int  # Frame where rebound is secured
    rebounder_id: int  # Track ID of rebounder
    rebound_type: str = "defensive"  # "offensive", "defensive"
    shot_frame: Optional[int] = None  # Frame of preceding shot (if known)
    team_id: Optional[int] = None  # Rebounding team
    
    def to_dict(self):
        return asdict(self)


@dataclass
class DunkAnnotation:
    """Ground truth for a dunk."""
    frame_idx: int  # Frame of dunk
    player_id: int  # Track ID of dunker
    team_id: Optional[int] = None
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TeamAssignmentGT:
    """Ground truth for team assignment on a per-frame basis (spot check)."""
    frame_idx: int
    track_id: int
    team_id: int  # 1 or 2


@dataclass
class TrackingGT:
    """Ground truth for player tracking on select frames (spot check)."""
    frame_idx: int
    expected_num_players: int  # Expected number of visible players in frame
    players_by_position: Optional[Dict[str, int]] = None  # e.g., {"pg": 1, "sg": 1, ...}


@dataclass
class GroundTruth:
    """Complete ground truth annotation for a single video clip."""
    video_path: str
    fps: float
    total_frames: int

    # Grouping keys (B-03).  Frames inside one clip are not independent, so
    # every aggregate should be reported per clip and grouped by game.
    game_id: str = ""
    footage_type: str = "unknown"   # e.g. 'nba_broadcast', 'college_broadcast', 'phone_gym'
    
    # Event ground truth
    shots: List[ShotAnnotation] = field(default_factory=list)
    passes: List[PassAnnotation] = field(default_factory=list)
    screens: List[ScreenAnnotation] = field(default_factory=list)
    pnrs: List[PnRAnnotation] = field(default_factory=list)
    blocks: List[BlockAnnotation] = field(default_factory=list)
    rebounds: List[ReboundAnnotation] = field(default_factory=list)
    dunks: List[DunkAnnotation] = field(default_factory=list)
    
    # Spot-check ground truth (for validation, not comprehensive)
    team_assignments: List[TeamAssignmentGT] = field(default_factory=list)
    tracking_checks: List[TrackingGT] = field(default_factory=list)
    
    # Metadata
    annotator: str = "unknown"
    annotation_date: str = ""
    notes: str = ""

    @classmethod
    def from_json(cls, path: str) -> "GroundTruth":
        """Load ground truth from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct typed objects
        gt = cls(
            video_path=data["video_path"],
            fps=data["fps"],
            total_frames=data["total_frames"],
            game_id=data.get("game_id", Path(path).stem),
            footage_type=data.get("footage_type", "unknown"),
            annotator=data.get("annotator", "unknown"),
            annotation_date=data.get("annotation_date", ""),
            notes=data.get("notes", ""),
        )
        
        # Reconstruct events
        for shot_dict in data.get("shots", []):
            gt.shots.append(ShotAnnotation(
                frame_idx=shot_dict["frame_idx"],
                track_id=shot_dict["track_id"],
                shot_type=ShotType(shot_dict.get("shot_type", "other")),
                outcome=ShotOutcome(shot_dict.get("outcome", "missed")),
                shot_distance_ft=shot_dict.get("shot_distance_ft"),
            ))
        
        for pass_dict in data.get("passes", []):
            gt.passes.append(PassAnnotation(
                frame_start=pass_dict["frame_start"],
                frame_end=pass_dict["frame_end"],
                from_track_id=pass_dict["from_track_id"],
                to_track_id=pass_dict["to_track_id"],
                team_id=pass_dict.get("team_id"),
                outcome=pass_dict.get("outcome", "complete"),
            ))
        
        for screen_dict in data.get("screens", []):
            gt.screens.append(ScreenAnnotation(
                frame_idx=screen_dict["frame_idx"],
                screener_id=screen_dict["screener_id"],
                cutter_id=screen_dict["cutter_id"],
                screen_type=ScreenType(screen_dict.get("screen_type", "on_ball")),
                team_id=screen_dict.get("team_id"),
            ))
        
        for pnr_dict in data.get("pnrs", []):
            gt.pnrs.append(PnRAnnotation(
                frame_screen=pnr_dict["frame_screen"],
                frame_roll=pnr_dict.get("frame_roll", pnr_dict["frame_screen"]),
                screener_id=pnr_dict["screener_id"],
                ball_handler_id=pnr_dict["ball_handler_id"],
                pnr_type=pnr_dict.get("pnr_type", "drag"),
                team_id=pnr_dict.get("team_id"),
                outcome=pnr_dict.get("outcome", "incomplete"),
            ))
        
        for block_dict in data.get("blocks", []):
            gt.blocks.append(BlockAnnotation(
                frame_shot=block_dict["frame_shot"],
                frame_block=block_dict["frame_block"],
                shooter_id=block_dict["shooter_id"],
                blocker_id=block_dict["blocker_id"],
                block_type=block_dict.get("block_type", "contested"),
            ))
        
        for rebound_dict in data.get("rebounds", []):
            gt.rebounds.append(ReboundAnnotation(
                frame_idx=rebound_dict["frame_idx"],
                rebounder_id=rebound_dict["rebounder_id"],
                rebound_type=rebound_dict.get("rebound_type", "defensive"),
                shot_frame=rebound_dict.get("shot_frame"),
                team_id=rebound_dict.get("team_id"),
            ))
        
        for dunk_dict in data.get("dunks", []):
            gt.dunks.append(DunkAnnotation(
                frame_idx=dunk_dict["frame_idx"],
                player_id=dunk_dict["player_id"],
                team_id=dunk_dict.get("team_id"),
            ))
        
        for ta_dict in data.get("team_assignments", []):
            gt.team_assignments.append(TeamAssignmentGT(
                frame_idx=ta_dict["frame_idx"],
                track_id=ta_dict["track_id"],
                team_id=ta_dict["team_id"],
            ))
        
        for tck_dict in data.get("tracking_checks", []):
            gt.tracking_checks.append(TrackingGT(
                frame_idx=tck_dict["frame_idx"],
                expected_num_players=tck_dict["expected_num_players"],
                players_by_position=tck_dict.get("players_by_position"),
            ))
        
        return gt

    def to_json(self, path: str) -> None:
        """Save ground truth to JSON file."""
        data = {
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "game_id": self.game_id,
            "footage_type": self.footage_type,
            "annotator": self.annotator,
            "annotation_date": self.annotation_date,
            "notes": self.notes,
            "shots": [s.to_dict() for s in self.shots],
            "passes": [p.to_dict() for p in self.passes],
            "screens": [s.to_dict() for s in self.screens],
            "pnrs": [p.to_dict() for p in self.pnrs],
            "blocks": [b.to_dict() for b in self.blocks],
            "rebounds": [r.to_dict() for r in self.rebounds],
            "dunks": [d.to_dict() for d in self.dunks],
            "team_assignments": [
                {
                    "frame_idx": ta.frame_idx,
                    "track_id": ta.track_id,
                    "team_id": ta.team_id,
                }
                for ta in self.team_assignments
            ],
            "tracking_checks": [
                {
                    "frame_idx": tck.frame_idx,
                    "expected_num_players": tck.expected_num_players,
                    "players_by_position": tck.players_by_position,
                }
                for tck in self.tracking_checks
            ],
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def summary(self) -> str:
        """Print summary of annotations."""
        return (
            f"GroundTruth({self.video_path})\n"
            f"  Game: {self.game_id}  Footage: {self.footage_type}\n"
            f"  Shots: {len(self.shots)}\n"
            f"  Passes: {len(self.passes)}\n"
            f"  Screens: {len(self.screens)}\n"
            f"  PnRs: {len(self.pnrs)}\n"
            f"  Blocks: {len(self.blocks)}\n"
            f"  Rebounds: {len(self.rebounds)}\n"
            f"  Dunks: {len(self.dunks)}\n"
            f"  Team assignments (spot-check): {len(self.team_assignments)}\n"
            f"  Tracking checks (spot-check): {len(self.tracking_checks)}"
        )
