import pytest

from analytics.playtypes import (
    league_averages,
    normalize,
    player_rows,
    report,
    report_player,
    report_team_players,
    team_rows,
)


def row(team_id, team, play_type, freq, ppp, efg, tov, ft, poss=100.0, pts=100.0, pct=0.5):
    return {
        "SEASON_ID": "22023",
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": team,
        "TEAM_NAME": team,
        "PLAY_TYPE": play_type,
        "TYPE_GROUPING": "Offensive",
        "PERCENTILE": pct,
        "GP": 82,
        "POSS_PCT": freq,
        "PPP": ppp,
        "FG_PCT": 0.45,
        "FT_POSS_PCT": ft,
        "TOV_POSS_PCT": tov,
        "SF_POSS_PCT": 0.1,
        "PLUSONE_POSS_PCT": 0.02,
        "SCORE_POSS_PCT": 0.45,
        "EFG_PCT": efg,
        "POSS": poss,
        "PTS": pts,
        "FGM": 40,
        "FGA": 85,
        "FGMX": 45,
    }


def test_normalize_numeric_fields():
    r = normalize(row(1, "IND", "Isolation", 0.12, 1.05, 0.50, 0.08, 0.13))
    assert r["team_id"] == 1
    assert r["poss_pct"] == pytest.approx(0.12)
    assert r["ppp"] == pytest.approx(1.05)
    assert r["percentile"] == pytest.approx(0.5)


def test_team_rows_sorted_by_frequency():
    rows = [
        normalize(row(1, "IND", "Isolation", 0.10, 1.0, 0.5, 0.08, 0.1)),
        normalize(row(1, "IND", "Spotup", 0.20, 1.1, 0.55, 0.05, 0.05)),
        normalize(row(2, "CLE", "Isolation", 0.15, 1.0, 0.5, 0.08, 0.1)),
    ]
    assert [r["play_type"] for r in team_rows(rows, 1)] == ["Spotup", "Isolation"]


def test_league_averages_weight_ppp_by_possessions():
    rows = [
        normalize(row(1, "IND", "Isolation", 0.10, 1.00, 0.5, 0.08, 0.1, poss=100, pts=100)),
        normalize(row(2, "CLE", "Isolation", 0.20, 1.20, 0.5, 0.08, 0.1, poss=300, pts=360)),
    ]
    league = league_averages(rows)
    assert league["Isolation"]["poss_pct"] == pytest.approx(0.15)
    assert league["Isolation"]["ppp"] == pytest.approx(460 / 400)


def test_report_returns_summary():
    rows = [
        normalize(row(1, "IND", "Isolation", 0.10, 1.0, 0.5, 0.08, 0.1)),
        normalize(row(1, "IND", "Spotup", 0.20, 1.1, 0.55, 0.05, 0.05)),
        normalize(row(2, "CLE", "Isolation", 0.15, 1.0, 0.5, 0.08, 0.1)),
        normalize(row(2, "CLE", "Spotup", 0.18, 1.05, 0.54, 0.05, 0.06)),
    ]
    summary = report(rows, 1)
    assert summary["team_id"] == 1
    assert summary["rows"][0]["play_type"] == "Spotup"
    assert "Isolation" in summary["league"]


def test_report_unknown_team_raises():
    with pytest.raises(ValueError):
        report([normalize(row(1, "IND", "Isolation", 0.1, 1.0, 0.5, 0.08, 0.1))], 99)

def player_row(pid, name, team_id, team, play_type, freq, ppp, poss, pts, pct=0.5):
    raw = row(team_id, team, play_type, freq, ppp, 0.5, 0.08, 0.1, poss=poss, pts=pts, pct=pct)
    raw["PLAYER_ID"] = pid
    raw["PLAYER_NAME"] = name
    return normalize(raw)


def test_player_rows_filter_and_report():
    rows = [
        player_row(1, "Haliburton", 1610612754, "IND", "PRBallHandler", 0.40, 1.0, 12.0, 12.0),
        player_row(1, "Haliburton", 1610612754, "IND", "Spotup", 0.10, 1.1, 3.0, 3.3),
        player_row(2, "Turner", 1610612754, "IND", "Postup", 0.20, 1.1, 5.0, 5.5),
        normalize(row(1610612754, "IND", "Spotup", 0.20, 1.1, 0.5, 0.05, 0.05)),
    ]
    assert all(r["level"] == "player" for r in rows[:3])
    assert [r["play_type"] for r in player_rows(rows, 1)] == ["PRBallHandler", "Spotup"]
    summary = report_player(rows, 1)
    assert summary["player"] == "Haliburton"
    assert summary["rows"][0]["play_type"] == "PRBallHandler"


def test_report_team_players_ranks_by_total_possessions():
    rows = [
        player_row(1, "Haliburton", 1610612754, "IND", "PRBallHandler", 0.40, 1.0, 12.0, 12.0),
        player_row(1, "Haliburton", 1610612754, "IND", "Spotup", 0.10, 1.1, 3.0, 3.3),
        player_row(2, "Turner", 1610612754, "IND", "Postup", 0.20, 1.1, 5.0, 5.5),
    ]
    summary = report_team_players(rows, 1610612754)
    assert summary["players"][0]["player"] == "Haliburton"
    assert summary["players"][0]["total_poss"] == pytest.approx(15.0)
    assert summary["players"][1]["player"] == "Turner"
