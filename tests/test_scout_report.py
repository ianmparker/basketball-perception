from analytics.playtypes import normalize as normalize_play
from scripts.nba_scout_report import _ordinal, _read, build_markdown


def adv():
    return {
        "TEAM_ID": 1, "TEAM_NAME": "Test Team", "W": 50, "L": 32,
        "OFF_RATING": 118.0, "DEF_RATING": 112.0, "NET_RATING": 6.0, "PACE": 101.0,
        "OFF_RATING_RANK": 3, "DEF_RATING_RANK": 12, "NET_RATING_RANK": 4, "PACE_RANK": 7,
    }


def ff():
    return {
        "TEAM_ID": 1, "TEAM_NAME": "Test Team",
        "EFG_PCT": 0.56, "EFG_PCT_RANK": 5, "OPP_EFG_PCT": 0.53, "OPP_EFG_PCT_RANK": 22,
        "TM_TOV_PCT": 0.12, "TM_TOV_PCT_RANK": 4, "OPP_TOV_PCT": 0.14, "OPP_TOV_PCT_RANK": 18,
        "OREB_PCT": 0.30, "OREB_PCT_RANK": 6, "OPP_OREB_PCT": 0.27, "OPP_OREB_PCT_RANK": 26,
        "FTA_RATE": 0.24, "FTA_RATE_RANK": 10, "OPP_FTA_RATE": 0.20, "OPP_FTA_RATE_RANK": 30,
    }


def play(team_id, team, play_type, freq, ppp, efg, pct=0.8, poss=10, pts=10):
    return normalize_play({
        "SEASON_ID": "22023", "TEAM_ID": team_id, "TEAM_ABBREVIATION": team,
        "TEAM_NAME": team, "PLAY_TYPE": play_type, "TYPE_GROUPING": "Offensive",
        "PERCENTILE": pct, "GP": 82, "POSS_PCT": freq, "PPP": ppp, "FG_PCT": 0.5,
        "FT_POSS_PCT": 0.1, "TOV_POSS_PCT": 0.1, "SF_POSS_PCT": 0.1,
        "PLUSONE_POSS_PCT": 0.02, "SCORE_POSS_PCT": 0.45, "EFG_PCT": efg,
        "POSS": poss, "PTS": pts, "FGM": 4, "FGA": 8, "FGMX": 4,
    })


def test_build_markdown_contains_all_sections():
    plays = [
        play(1, "TST", "PRBallHandler", 0.30, 1.05, 0.55),
        play(1, "TST", "Spotup", 0.20, 1.10, 0.58),
        play(2, "OTH", "PRBallHandler", 0.20, 0.95, 0.50),
        play(2, "OTH", "Spotup", 0.25, 1.00, 0.52),
    ]
    shots = [
        {"zone_basic": "Restricted Area", "shot_type": "2PT Field Goal", "made": 1},
        {"zone_basic": "Above the Break 3", "shot_type": "3PT Field Goal", "made": 0},
    ]
    md = build_markdown(1, "2023-24", adv(), ff(), [], plays, None, shots)
    assert "# Test Team — 2023-24 scouting report" in md
    assert "## Four Factors" in md
    assert "## Play-type profile" in md
    assert "## Shot profile" in md
    assert "## Scouting read" in md
    assert "PnR ball handler" in md


def test_build_markdown_without_shots_or_plays():
    md = build_markdown(1, "2023-24", adv(), ff(), [], [], None, [])
    assert "## Identity" in md
    assert "## Play-type profile" in md
    assert "## Shot profile" in md


def test_ordinal_suffixes():
    assert _ordinal(1) == "1st"
    assert _ordinal(2) == "2nd"
    assert _ordinal(3) == "3rd"
    assert _ordinal(4) == "4th"
    assert _ordinal(11) == "11th"
    assert _ordinal(12) == "12th"
    assert _ordinal(13) == "13th"
    assert _ordinal(21) == "21st"
    assert _ordinal(22) == "22nd"
    assert _ordinal(30) == "30th"


def test_read_attack_uses_defensive_rank_semantics():
    lines = _read(adv(), ff(), [], {}, {})
    text = "\n".join(lines)
    assert "3rd in ORtg" in text
    assert "crash the offensive glass" in text
    assert "draw contact" in text

