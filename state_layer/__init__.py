"""State layer: game state, possessions, lineups, and PBP ingestion.

This package is additive. It does not alter perception or event thresholds.
"""
from state_layer.manual import ManualGameState, load_manual_state
from state_layer.espn import (
    PbpPlay,
    PbpTimeline,
    build_timeline,
    fetch_summary,
    parse_clock,
    validate_timeline,
)
from state_layer.alignment import ClockAnchor, align_plays, frame_for_clock
from state_layer.extract import extract_state, format_extraction_scorecard
from state_layer.lineups import extract_lineups, starters_from_boxscore
from state_layer.possession import boxscore_possessions, extract_possessions

__all__ = [
    "ManualGameState",
    "load_manual_state",
    "PbpPlay",
    "PbpTimeline",
    "build_timeline",
    "fetch_summary",
    "parse_clock",
    "validate_timeline",
    "ClockAnchor",
    "align_plays",
    "frame_for_clock",
    "extract_state",
    "format_extraction_scorecard",
    "extract_lineups",
    "starters_from_boxscore",
    "boxscore_possessions",
    "extract_possessions",
]
