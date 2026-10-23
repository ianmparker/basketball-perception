"""Manual game-state input for gym / non-PBP footage.

The manual form is deliberately first-class: a coach or analyst can set period,
clock, score, lineup, and camera context at stoppages without waiting for OCR.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from inference.schema import GameState


class ManualGameState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    game_id: str
    video_path: Optional[str] = None
    home_team_id: int
    away_team_id: int
    period: int = Field(ge=1)
    clock_s: Optional[float] = Field(default=None, ge=0.0)
    score_home: int = Field(default=0, ge=0)
    score_away: int = Field(default=0, ge=0)
    possession_team_id: Optional[int] = None
    lineup_home: List[int] = Field(default_factory=list)
    lineup_away: List[int] = Field(default_factory=list)
    ruleset: str = "unknown"
    footage_type: str = "unknown"
    camera_spec: Optional[dict] = None
    # Frame anchors for video alignment. Each item: {frame, period, clock_s}.
    clock_anchors: List[dict] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _check_state(self):
        if self.home_team_id == self.away_team_id:
            raise ValueError("home_team_id and away_team_id must differ")
        if self.lineup_home and len(self.lineup_home) != 5:
            raise ValueError("lineup_home must be empty or contain exactly 5 players")
        if self.lineup_away and len(self.lineup_away) != 5:
            raise ValueError("lineup_away must be empty or contain exactly 5 players")
        for anchor in self.clock_anchors:
            if set(anchor) != {"frame", "period", "clock_s"}:
                raise ValueError("clock anchors require frame, period, clock_s")
            if int(anchor["frame"]) < 0 or int(anchor["period"]) < 1:
                raise ValueError("clock anchor frame/period out of range")
            if float(anchor["clock_s"]) < 0:
                raise ValueError("clock anchor clock_s must be >= 0")
        return self

    def to_game_state(self) -> GameState:
        return GameState(
            period=self.period,
            clock_s=self.clock_s,
            score_home=self.score_home,
            score_away=self.score_away,
            home_team_id=self.home_team_id,
            away_team_id=self.away_team_id,
            possession_team_id=self.possession_team_id,
            lineup_home=self.lineup_home,
            lineup_away=self.lineup_away,
            ruleset=self.ruleset,
            footage_type=self.footage_type,
            camera_spec=self.camera_spec,
        )


def load_manual_state(path: str | Path) -> ManualGameState:
    return ManualGameState.model_validate_json(Path(path).read_text(encoding="utf-8"))
