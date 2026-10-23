from analytics.shot_events import (
    build_shot_events,
    find_rebound,
    parse_distance,
    select_validation,
    shot_type,
    shot_value,
)


def play(seq, period, clock, text, typ="Jump Shot", team=16, scoring=True,
         shooting=True, points=3, participants=None):
    return {
        "sequence": seq,
        "period": period,
        "clock_s": clock,
        "text": text,
        "type": typ,
        "team_id": team,
        "scoring": scoring,
        "shooting": shooting,
        "points": points,
        "participants": participants or [1],
    }


def test_parse_distance_and_shot_value():
    assert parse_distance("Randle makes 25-foot three point jumper") == 25.0
    assert parse_distance("layup") is None
    assert shot_value("25-foot three point jumper", 3, True) == 3
    assert shot_value("26-foot three point jumper", 0, False) == 3
    assert shot_value("driving layup", 2, True) == 2
    assert shot_type("driving dunk", 2) == "dunk"
    assert shot_type("driving layup", 2) == "layup"
    assert shot_type("25-foot three point jumper", 3) == "three_pointer"
    assert shot_type("pullup jump shot", 2) == "mid_range"


def test_find_rebound_matches_next_rebound_within_window():
    plays = [
        play(1, 1, 700, "miss", scoring=False, points=0),
        {"sequence": 2, "period": 1, "clock_s": 698, "type": "Defensive Rebound",
         "team_id": 18, "participants": [4]},
    ]
    reb = find_rebound(plays, 0)
    assert reb is not None
    assert reb["team_id"] == 18


def test_build_shot_events_excludes_free_throws_and_maps_rebound():
    plays = [
        play(1, 1, 700, "Randle makes 25-foot three point jumper", points=3,
             participants=[1, 2]),
        play(2, 1, 690, "DiVincenzo misses 26-foot three point jumper",
             scoring=False, points=0, participants=[3]),
        {"sequence": 3, "period": 1, "clock_s": 688, "type": "Defensive Rebound",
         "team_id": 18, "participants": [4]},
        play(4, 1, 680, "Towns makes free throw 1 of 2",
             typ="Free Throw - 1 of 2", points=1, participants=[5]),
    ]
    anchors = [
        {"frame": 18000, "period": 1, "clock_s": 720},
        {"frame": 19800, "period": 1, "clock_s": 600},
    ]
    events = build_shot_events(
        plays,
        {"1": "Randle", "2": "DiVincenzo", "3": "DiVincenzo", "4": "Robinson"},
        {"16": "Wolves", "18": "Knicks"},
        anchors,
    )
    assert len(events) == 2
    made, missed = events
    assert made["made"] is True
    assert made["shot_value"] == 3
    assert made["shooter"] == "Randle"
    assert made["rebound"] is None  # 12 s away, outside the 10 s window
    assert missed["made"] is False
    assert missed["shot_value"] == 3
    assert missed["rebound"]["team"] == "Knicks"
    assert missed["rebound"]["offensive"] is False
    assert missed["rebound"]["player"] == "Robinson"
    assert events[0]["frame"] == 18300


def test_select_validation_is_balanced_and_deterministic():
    events = [
        {"event_id": f"e{i}", "frame": i, "sequence": i, "made": i % 2 == 0}
        for i in range(10)
    ]
    a = select_validation(events, count=4, seed=3)
    b = select_validation(events, count=4, seed=3)
    assert [e["event_id"] for e in a] == [e["event_id"] for e in b]
    assert len(a) == 4
    assert sum(1 for e in a if e["made"]) == 2
    assert a == sorted(a, key=lambda e: (e["frame"], e["sequence"]))
