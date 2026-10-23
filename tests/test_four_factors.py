import pytest

from analytics.four_factors import parse_minutes, report, team_factors


def team(**overrides):
    base = {
        "TEAM_ABBREVIATION": "IND",
        "MIN": "240:00",
        "FGM": 45,
        "FGA": 86,
        "FG3M": 15,
        "FTA": 24,
        "OREB": 7,
        "DREB": 33,
        "TOV": 18,
        "PTS": 121,
    }
    base.update(overrides)
    return base


def opp():
    return {
        "TEAM_ABBREVIATION": "CLE",
        "MIN": "240:00",
        "FGM": 44,
        "FGA": 84,
        "FG3M": 8,
        "FTA": 25,
        "OREB": 5,
        "DREB": 30,
        "TOV": 13,
        "PTS": 116,
    }


def test_parse_minutes_handles_clock_and_number():
    assert parse_minutes("240:00") == pytest.approx(240.0)
    assert parse_minutes("53:30") == pytest.approx(53.5)
    assert parse_minutes(48) == pytest.approx(48.0)
    assert parse_minutes("") == 0.0


def test_team_factors_match_hand_calculation():
    f = team_factors(team(), opp())
    assert f["poss"] == pytest.approx(86 + 0.44 * 24 - 7 + 18)
    assert f["opp_poss"] == pytest.approx(84 + 0.44 * 25 - 5 + 13)
    assert f["ortg"] == pytest.approx(100 * 121 / f["poss"])
    assert f["drtg"] == pytest.approx(100 * 116 / f["opp_poss"])
    assert f["efg"] == pytest.approx((45 + 0.5 * 15) / 86)
    assert f["tov_pct"] == pytest.approx(18 / (86 + 0.44 * 24 + 18))
    assert f["orb_pct"] == pytest.approx(7 / (7 + 30))
    assert f["ftr"] == pytest.approx(24 / 86)
    assert f["pace"] == pytest.approx((f["poss"] + f["opp_poss"]) / 2)


def test_zero_denominators_do_not_crash():
    f = team_factors({}, {})
    assert f["poss"] == 0
    assert f["ortg"] == 0
    assert f["pace"] == 0
    assert f["efg"] == 0


def test_nba_style_to_key_is_used_for_turnovers():
    f = team_factors(team(TO=18, TOV=None), opp())
    assert f["poss"] == pytest.approx(86 + 0.44 * 24 - 7 + 18)
    assert f["tov_pct"] == pytest.approx(18 / (86 + 0.44 * 24 + 18))


def test_report_returns_both_teams_and_pace():
    summary = report([team(), opp()])
    assert len(summary["teams"]) == 2
    assert summary["pace"] > 0
    assert summary["teams"][0]["team"] == "IND"
    assert summary["teams"][1]["team"] == "CLE"


def test_report_requires_two_teams():
    with pytest.raises(ValueError):
        report([team()])
