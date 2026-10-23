import pytest

from scripts.nba_game_report import build_markdown


def team(team_id, abbr, pts, fgm, fga, fg3m, fta, oreb, dreb, to, min_="240:00"):
    return {
        "TEAM_ID": team_id,
        "TEAM_ABBREVIATION": abbr,
        "MIN": min_,
        "FGM": fgm,
        "FGA": fga,
        "FG3M": fg3m,
        "FTA": fta,
        "OREB": oreb,
        "DREB": dreb,
        "TO": to,
        "PTS": pts,
    }


def shot(team_id, zone, made, shot_type="2PT Field Goal"):
    return {"team_id": team_id, "zone_basic": zone, "shot_type": shot_type, "made": made}


def test_build_markdown_contains_sections():
    teams = [team(1, "IND", 121, 45, 86, 15, 24, 7, 33, 18), team(2, "CLE", 116, 44, 84, 8, 25, 5, 30, 13)]
    shots = [
        shot(1, "Restricted Area", 1), shot(1, "Restricted Area", 0),
        shot(1, "Above the Break 3", 1, "3PT Field Goal"), shot(1, "Mid-Range", 0),
        shot(2, "Restricted Area", 1), shot(2, "Above the Break 3", 0, "3PT Field Goal"),
        shot(2, "Mid-Range", 1), shot(2, "Mid-Range", 0),
    ]
    md = build_markdown("test-game", teams, shots)
    assert "# IND 121 vs CLE 116" in md
    assert "## Four Factors" in md
    assert "## Shot profile" in md
    assert "Restricted Area" in md
    assert "## Coach read" in md
    assert "Pace:" in md
    assert "## Caveats" in md


def test_build_markdown_requires_two_teams():
    with pytest.raises(ValueError):
        build_markdown("test", [team(1, "IND", 100, 40, 80, 10, 20, 5, 30, 10)], [])
