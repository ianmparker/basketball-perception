"""State-layer v0 tests: schema, manual state, PBP parsing, clock anchors."""
from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from inference.schema import (
    ClipMetadata,
    ClipResult,
    GameState,
    LineupInterval,
    Play,
    Possession,
    ShotEventModel,
)
from state_layer.alignment import ClockAnchor, frame_for_clock
from state_layer.espn import (
    build_timeline,
    parse_clock,
    validate_timeline,
)
from state_layer.extract import extract_state
from state_layer.lineups import extract_lineups
from state_layer.manual import ManualGameState
from state_layer.possession import extract_possessions


def _summary() -> dict:
    return {
        "header": {
            "competitions": [{
                "date": "2026-01-01T00:00Z",
                "competitors": [
                    {"homeAway": "home", "team": {"id": "18", "displayName": "Home"},
                     "score": "2"},
                    {"homeAway": "away", "team": {"id": "16", "displayName": "Away"},
                     "score": "0"},
                ],
            }]
        },
        "plays": [
            {"sequenceNumber": "1", "period": {"number": 1},
             "clock": {"displayValue": "12:00"}, "awayScore": 0, "homeScore": 0,
             "type": {"text": "Jumpball"}, "text": "tip", "scoringPlay": False,
             "shootingPlay": False, "scoreValue": 0},
            {"sequenceNumber": "2", "period": {"number": 1},
             "clock": {"displayValue": "11:00"}, "awayScore": 0, "homeScore": 2,
             "type": {"text": "Shooting"}, "text": "make", "scoringPlay": True,
             "shootingPlay": True, "scoreValue": 2, "team": {"id": "18"}},
            {"sequenceNumber": "3", "period": {"number": 2},
             "clock": {"displayValue": "12:00"}, "awayScore": 0, "homeScore": 2,
             "type": {"text": "Jumpball"}, "text": "q2", "scoringPlay": False,
             "shootingPlay": False, "scoreValue": 0},
            {"sequenceNumber": "4", "period": {"number": 2},
             "clock": {"displayValue": "0.0"}, "awayScore": 0, "homeScore": 2,
             "type": {"text": "End Game"}, "text": "end", "scoringPlay": False,
             "shootingPlay": False, "scoreValue": 0},
        ],
    }


def _summary_with_boxscore() -> dict:
    summary = _summary()
    summary["boxscore"] = {
        "teams": [
            {"team": {"id": "18"}, "statistics": [
                {"name": "fieldGoalsMade-fieldGoalsAttempted",
                 "displayValue": "2-2"},
                {"name": "freeThrowsMade-freeThrowsAttempted",
                 "displayValue": "0-0"},
                {"name": "offensiveRebounds", "displayValue": "1"},
                {"name": "turnovers", "displayValue": "1"},
            ]},
            {"team": {"id": "16"}, "statistics": [
                {"name": "fieldGoalsMade-fieldGoalsAttempted",
                 "displayValue": "0-1"},
                {"name": "freeThrowsMade-freeThrowsAttempted",
                 "displayValue": "0-0"},
                {"name": "offensiveRebounds", "displayValue": "1"},
                {"name": "turnovers", "displayValue": "1"},
            ]},
        ],
        "players": [
            {"team": {"id": "18"}, "statistics": [{"athletes": [
                {"athlete": {"id": str(1801 + i),
                             "displayName": f"Home {i}"},
                 "starter": True, "didNotPlay": False}
                for i in range(5)
            ] + [
                {"athlete": {"id": "1806", "displayName": "Home Bench"},
                 "starter": False, "didNotPlay": False},
            ]}]},
            {"team": {"id": "16"}, "statistics": [{"athletes": [
                {"athlete": {"id": str(1601 + i),
                             "displayName": f"Away {i}"},
                 "starter": True, "didNotPlay": False}
                for i in range(5)
            ] + [
                {"athlete": {"id": "1606", "displayName": "Away Bench"},
                 "starter": False, "didNotPlay": True},
            ]}]},
        ],
    }
    return summary


def test_parse_clock_variants():
    assert parse_clock("12:00") == 720
    assert parse_clock("0.0") == 0
    assert parse_clock("1:04.5") == 64.5


def test_build_and_validate_timeline():
    timeline = build_timeline(_summary(), "test-game")
    assert timeline.home_team_id == 18
    assert timeline.away_team_id == 16
    assert timeline.final_home == 2
    checks = validate_timeline(timeline)
    assert checks["passed"], checks


def test_manual_state_and_game_state():
    manual = ManualGameState(
        game_id="test-game",
        home_team_id=18,
        away_team_id=16,
        period=1,
        clock_s=720,
        score_home=0,
        score_away=0,
        clock_anchors=[{"frame": 0, "period": 1, "clock_s": 720}],
    )
    state = manual.to_game_state()
    assert isinstance(state, GameState)
    assert state.period == 1
    with pytest.raises(ValidationError):
        ManualGameState(
            game_id="bad", home_team_id=1, away_team_id=1, period=1,
        )


def test_schema_round_trip_state():
    clip = ClipResult(
        metadata=ClipMetadata(video="clip.mp4", fps=30, width=640, height=360,
                              frame_count=0),
        game_state=GameState(period=1, clock_s=720, home_team_id=18,
                             away_team_id=16),
        possessions=[Possession(possession_id=1, offense_team_id=18,
                                defense_team_id=16, start_frame=0)],
        plays=[Play(play_id=1, possession_id=1, play_type="pnr_ball_handler")],
        lineups=[LineupInterval(team_id=18, start_frame=0, end_frame=100,
                                player_ids=[1801, 1802, 1803, 1804, 1805],
                                source="test", confidence=1.0)],
        shots=[ShotEventModel(frame_start=10, ball_px=(1, 2),
                              possession_id=1, play_id=1)],
    )
    loaded = ClipResult.model_validate_json(clip.model_dump_json())
    assert loaded.game_state.home_team_id == 18
    assert loaded.plays[0].play_type == "pnr_ball_handler"
    assert loaded.shots[0].possession_id == 1
    assert loaded.lineups[0].player_ids == [1801, 1802, 1803, 1804, 1805]


def test_frame_for_clock_interpolates_and_refuses_outside():
    anchors = [
        ClockAnchor(frame=100, period=1, clock_s=700),
        ClockAnchor(frame=500, period=1, clock_s=300),
    ]
    assert frame_for_clock(1, 500, anchors) == 300
    assert frame_for_clock(1, 720, anchors) is None
    assert frame_for_clock(2, 500, anchors) is None


def _extraction_anchors():
    return [
        ClockAnchor(frame=100, period=1, clock_s=720),
        ClockAnchor(frame=1000, period=1, clock_s=0),
        ClockAnchor(frame=2000, period=4, clock_s=0),
    ]


def _timeline_from_plays(plays, final_home=0, final_away=0):
    summary = _summary_with_boxscore()
    summary["header"]["competitions"][0]["competitors"][0]["score"] = str(
        final_home
    )
    summary["header"]["competitions"][0]["competitors"][1]["score"] = str(
        final_away
    )
    summary["plays"] = plays
    return summary, build_timeline(summary, "test-game")


def _play(sequence, period, clock, type_text, *, team=None, scoring=False,
          shooting=False, points=0, home_score=0, away_score=0,
          text="", participants=None):
    out = {
        "sequenceNumber": str(sequence),
        "period": {"number": period},
        "clock": {"displayValue": clock},
        "awayScore": away_score,
        "homeScore": home_score,
        "type": {"text": type_text},
        "text": text or type_text,
        "scoringPlay": scoring,
        "shootingPlay": shooting,
        "scoreValue": points,
    }
    if team is not None:
        out["team"] = {"id": str(team)}
    if participants:
        out["participants"] = [{"athlete": {"id": str(p)}} for p in participants]
    return out


def test_possession_ledger_reconciles_points_and_switches_on_turnover():
    plays = [
        _play(1, 1, "12:00", "Jumpball", team=18),
        _play(2, 1, "11:00", "Shooting", team=18, scoring=True, shooting=True,
              points=2, home_score=2),
        _play(3, 1, "10:55", "Substitution", team=18,
              participants=[1806, 1801],
              text="Home Bench enters the game for Home 0"),
        _play(4, 1, "10:50", "Defensive Rebound", team=16, home_score=2),
        # This turnover occurs while the ledger has no established offense
        # after the made shot. It must establish 16, end it, and hand the ball
        # to 18 rather than leaving 16 as the offense.
        _play(5, 1, "10:40", "Bad Pass\nTurnover", team=16,
              text="Away bad pass turnover", home_score=2),
        _play(6, 1, "10:20", "Shooting", team=18, scoring=True, shooting=True,
              points=2, home_score=4),
        _play(7, 4, "0:00", "End Game", home_score=4),
    ]
    summary, timeline = _timeline_from_plays(plays, final_home=4)
    possessions, diagnostics = extract_possessions(
        timeline, _extraction_anchors()
    )
    assert diagnostics["points_by_team"] == {"18": 4, "16": 0}
    assert diagnostics["points_match_header"] is True
    assert diagnostics["unmatched_team_events"] == 0
    assert sum(p.points for p in possessions) == 4
    assert [p.end_reason for p in possessions] == [
        "made_shot", "turnover", "made_shot",
    ]

    state = extract_state(summary, timeline, _extraction_anchors())
    assert state["passed"], state["checks"]
    assert state["possessions"][0]["offense_team_id"] == 18
    assert state["lineups"]


def test_lineup_extraction_applies_substitution_and_keeps_five():
    plays = [
        _play(1, 1, "12:00", "Jumpball", team=18),
        _play(2, 1, "10:00", "Substitution", team=18,
              participants=[1806, 1801],
              text="Home Bench enters the game for Home 0"),
        _play(3, 4, "0:00", "End Game"),
    ]
    summary, timeline = _timeline_from_plays(plays)
    lineups, diagnostics = extract_lineups(
        summary, timeline, _extraction_anchors()
    )
    assert diagnostics["substitutions"] == 1
    assert diagnostics["errors"] == []
    assert diagnostics["all_intervals_have_five"] is True
    assert len(lineups) == 3
    home = sorted((i for i in lineups if i.team_id == 18),
                  key=lambda i: i.start_frame)
    assert len(home) == 2
    assert 1801 in home[0].player_ids
    assert 1801 not in home[1].player_ids
    assert 1806 in home[1].player_ids
    assert all(len(i.player_ids) == 5 for i in lineups)


def test_extraction_refuses_frames_outside_anchors():
    plays = [
        _play(1, 1, "12:00", "Jumpball", team=18),
        _play(2, 4, "0:00", "End Game"),
    ]
    _, timeline = _timeline_from_plays(plays)
    anchors = [ClockAnchor(frame=2000, period=4, clock_s=0)]
    with pytest.raises(ValueError, match="no clock anchor"):
        extract_possessions(timeline, anchors)
    with pytest.raises(ValueError, match="no clock anchor"):
        extract_lineups(_summary_with_boxscore(), timeline, anchors)
