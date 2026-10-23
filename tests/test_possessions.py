import pytest

from analytics.possessions import build_possessions, parse_clock, summarize


def act(num, atype, team, sub="", value=0, result=""):
    return {
        "action_number": num,
        "action_type": atype,
        "team_id": team,
        "sub_type": sub,
        "shot_value": value,
        "shot_result": result,
    }


def test_parse_clock():
    assert parse_clock("PT12M00.00S") == pytest.approx(720.0)
    assert parse_clock("PT0M05.40S") == pytest.approx(5.4)
    assert parse_clock("garbage") is None


def test_made_shot_closes_and_switches_possession():
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Made Shot", 1, value=2, result="Made"),
        act(4, "period", 0, "end"),
    ]
    poss = build_possessions(actions)
    assert len(poss) == 1
    assert poss[0]["offense_team_id"] == 1
    assert poss[0]["points"] == 2
    assert poss[0]["end_reason"] == "made_shot"


def test_defensive_rebound_switches_possession():
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Missed Shot", 1),
        act(4, "Rebound", 2),
        act(5, "Made Shot", 2, value=3, result="Made"),
    ]
    poss = build_possessions(actions)
    assert [p["offense_team_id"] for p in poss] == [1, 2]
    assert poss[0]["end_reason"] == "defensive_rebound"
    assert poss[1]["points"] == 3


def test_offensive_rebound_keeps_possession():
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Missed Shot", 1),
        act(4, "Rebound", 1),
        act(5, "Made Shot", 1, value=2, result="Made"),
    ]
    poss = build_possessions(actions)
    assert len(poss) == 1
    assert poss[0]["points"] == 2
    assert poss[0]["end_reason"] == "made_shot"


def test_turnover_switches_possession():
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Turnover", 1),
        act(4, "Made Shot", 2, value=2, result="Made"),
    ]
    poss = build_possessions(actions)
    assert [p["offense_team_id"] for p in poss] == [1, 2]
    assert poss[0]["end_reason"] == "turnover"


def test_last_made_free_throw_closes_possession():
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Free Throw", 1, "Free Throw 1 of 2", 1, "Made"),
        act(4, "Free Throw", 1, "Free Throw 2 of 2", 1, "Made"),
        act(5, "Made Shot", 2, value=2, result="Made"),
    ]
    poss = build_possessions(actions)
    assert [p["offense_team_id"] for p in poss] == [1, 2]
    assert poss[0]["points"] == 2
    assert poss[0]["end_reason"] == "made_free_throw"


def test_summarize_counts_and_points():
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Made Shot", 1, value=2, result="Made"),
        act(4, "Made Shot", 2, value=3, result="Made"),
        act(5, "period", 0, "end"),
    ]
    summary = summarize(build_possessions(actions))
    assert summary["total"] == 2
    assert summary["by_team"][1]["points"] == 2
    assert summary["by_team"][2]["points"] == 3
    assert summary["points"] == 5


def test_blank_free_throw_result_is_made_unless_description_says_miss():
    made = act(3, "Free Throw", 1, "Free Throw 1 of 2", 0, "")
    made["description"] = "Turner Free Throw 1 of 2 (6 PTS)"
    missed = act(4, "Free Throw", 1, "Free Throw 2 of 2", 0, "")
    missed["description"] = "MISS Turner Free Throw 2 of 2"
    actions = [act(1, "period", 0, "start"), act(2, "Jump Ball", 1), made, missed]
    poss = build_possessions(actions)
    assert poss[0]["points"] == 1


def test_technical_free_throw_is_a_standalone_point():
    tech = act(5, "Free Throw", 1, "Free Throw Technical", 0, "")
    tech["description"] = "Turner Free Throw Technical (7 PTS)"
    actions = [
        act(1, "period", 0, "start"),
        act(2, "Jump Ball", 1),
        act(3, "Missed Shot", 1),
        act(4, "Rebound", 2),
        tech,
        act(6, "Made Shot", 2, value=2, result="Made"),
    ]
    summary = summarize(build_possessions(actions))
    assert summary["by_team"][1]["points"] == 1
    assert summary["by_team"][2]["points"] == 2
    assert summary["total"] == 2
