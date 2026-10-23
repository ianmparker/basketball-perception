from analytics.play_labels import (
    build_windows,
    lineup_names,
    period_for_frame,
    select_windows,
    validate_label_payload,
)


def possession(pid, start, end, offense=1, defense=2, points=2, reason="made_shot"):
    return {
        "possession_id": pid,
        "start_frame": start,
        "end_frame": end,
        "offense_team_id": offense,
        "defense_team_id": defense,
        "points": points,
        "end_reason": reason,
    }


def test_period_for_frame_uses_sorted_anchors():
    anchors = [
        {"frame": 0, "period": 1},
        {"frame": 7200, "period": 2},
        {"frame": 14400, "period": 3},
    ]
    assert period_for_frame(10, anchors) == 1
    assert period_for_frame(7200, anchors) == 2
    assert period_for_frame(9000, anchors) == 2
    assert period_for_frame(20000, anchors) == 3
    assert period_for_frame(0, []) == 0


def test_lineup_names_picks_interval_and_falls_back_to_id():
    lineups = [
        {"team_id": 1, "start_frame": 0, "end_frame": 100, "player_ids": [10, 11]},
        {"team_id": 1, "start_frame": 101, "end_frame": 200, "player_ids": [12, 13]},
    ]
    names = {"10": "A", "11": "B", "12": "C"}
    assert lineup_names(lineups, 1, 50, names) == ["A", "B"]
    assert lineup_names(lineups, 1, 150, names) == ["C", "13"]
    assert lineup_names(lineups, 2, 150, names) == []


def test_build_windows_skips_collapsed_boundaries_and_caps_lookback():
    plays = [
        possession(1, 0, 600),
        possession(2, 600, 600),
        possession(3, 600, 900),
        possession(4, 1200, 1300),
    ]
    windows = build_windows(
        plays, fps=30.0, total_frames=10_000,
        anchors=[{"frame": 0, "period": 1}],
        lineups=[], names={}, before_s=14.0, after_s=3.0, min_duration_s=5.0,
    )
    assert [w["possession_id"] for w in windows] == [1, 3, 4]
    assert windows[0]["clip_start_frame"] == 180
    assert windows[0]["clip_end_frame"] == 690
    assert windows[1]["clip_start_frame"] == 600
    assert windows[1]["clip_end_frame"] == 990
    assert windows[2]["clip_start_frame"] == 1200
    assert windows[2]["clip_end_frame"] == 1390
    assert all(w["duration_s"] >= 5.0 for w in windows)


def test_short_possession_is_extended_to_minimum_duration():
    windows = build_windows(
        [possession(1, 1000, 1050)],
        fps=30.0, total_frames=10_000,
        anchors=[{"frame": 0, "period": 1}],
        lineups=[], names={}, before_s=14.0, after_s=3.0, min_duration_s=5.0,
    )
    assert len(windows) == 1
    assert windows[0]["clip_start_frame"] == 810
    assert windows[0]["clip_end_frame"] == 1140
    assert windows[0]["duration_s"] == 11.0


def test_select_windows_is_reproducible_and_sorted():
    windows = [possession(i, i * 100, i * 100 + 200) for i in range(10)]
    a = select_windows(windows, count=4, seed=7)
    b = select_windows(windows, count=4, seed=7)
    assert [w["possession_id"] for w in a] == [w["possession_id"] for w in b]
    assert len(a) == 4
    assert a == sorted(a, key=lambda w: (w.get("clip_start_frame", w["start_frame"]), w["possession_id"]))
    assert len(select_windows(windows, count=0)) == 10


def test_validate_label_payload_accepts_complete_payload():
    tools = {"isolation", "transition"}
    payload = {"labels": {
        "1": {"play_type": "isolation", "ambiguous": True, "note": "iso left"},
        "2": {"play_type": "transition"},
    }}
    errors, summary = validate_label_payload(payload, [possession(1, 0, 90), possession(2, 100, 190)], tools)
    assert errors == []
    assert summary["labeled"] == 2
    assert summary["ambiguous"] == 1
    assert summary["distribution"] == {"isolation": 1, "transition": 1}


def test_validate_label_payload_rejects_unknown_ids_and_types():
    errors, _ = validate_label_payload(
        {"labels": {"3": {"play_type": "isolation"}, "1": {"play_type": "not_a_play"}}},
        [possession(1, 0, 90)], {"isolation"},
    )
    assert any("possession not in manifest" in e for e in errors)
    assert any("unknown play_type" in e for e in errors)


def test_validate_label_payload_requires_all_when_asked():
    errors, summary = validate_label_payload(
        {"labels": {"1": {"play_type": "isolation"}}},
        [possession(1, 0, 90), possession(2, 100, 190)],
        {"isolation"}, require_all=True,
    )
    assert summary["missing"] == 1
    assert any("missing a play_type" in e for e in errors)
