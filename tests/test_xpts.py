import math

import pytest

from analytics.xpts import (
    calibration_table,
    canonical_zone,
    evaluate,
    fit,
    grouped_split,
    report,
    shot_points,
    summarize,
)


def shot(game, zone, made, shot_type="2PT Field Goal"):
    return {
        "game_id": game,
        "game_date": "2024-01-01",
        "zone_basic": zone,
        "shot_type": shot_type,
        "made": made,
    }


def test_zone_mapping_collapses_corners_and_paint():
    assert canonical_zone(shot("g", "Left Corner 3", 1, "3PT Field Goal")) == "Corner 3"
    assert canonical_zone(shot("g", "Right Corner 3", 0, "3PT Field Goal")) == "Corner 3"
    assert canonical_zone(shot("g", "In The Paint (Non-RA)", 1)) == "Paint (non-RA)"
    assert canonical_zone(shot("g", "Restricted Area", 1)) == "Restricted Area"
    assert canonical_zone(shot("g", "Weird Zone", 1)) == "Other"


def test_shot_points():
    assert shot_points(shot("g", "Restricted Area", 1)) == 2
    assert shot_points(shot("g", "Above the Break 3", 0, "3PT Field Goal")) == 3


def test_summarize_computes_fg_and_xpts():
    shots = [
        shot("g1", "Restricted Area", 1),
        shot("g1", "Restricted Area", 1),
        shot("g1", "Restricted Area", 1),
        shot("g1", "Restricted Area", 0),
        shot("g2", "Left Corner 3", 1, "3PT Field Goal"),
        shot("g2", "Right Corner 3", 0, "3PT Field Goal"),
    ]
    table = summarize(shots)
    ra = table["Restricted Area"]
    assert ra["attempts"] == 4
    assert ra["makes"] == 3
    assert ra["fg"] == pytest.approx(0.75)
    assert ra["xpts"] == pytest.approx(1.5)
    corner = table["Corner 3"]
    assert corner["attempts"] == 2
    assert corner["makes"] == 1
    assert corner["fg"] == pytest.approx(0.5)
    assert corner["xpts"] == pytest.approx(1.5)


def test_grouped_split_never_leaks_a_game():
    shots = [shot(f"g{i}", "Mid-Range", i % 2) for i in range(1, 11)]
    train, test = grouped_split(shots, test_frac=0.3, seed=1)
    train_games = {s["game_id"] for s in train}
    test_games = {s["game_id"] for s in test}
    assert train_games.isdisjoint(test_games)
    assert len(train) + len(test) == len(shots)
    assert test_games


def test_grouped_split_requires_two_games():
    with pytest.raises(ValueError):
        grouped_split([shot("g1", "Mid-Range", 1)])


def test_evaluate_matches_known_values():
    shots = [shot("g1", "Restricted Area", 1), shot("g2", "Restricted Area", 0)]
    metrics = evaluate(shots, {"Restricted Area": 0.5}, 0.5)
    assert metrics["n"] == 2
    assert metrics["brier"] == pytest.approx(0.25)
    assert metrics["log_loss"] == pytest.approx(math.log(2))


def test_fit_shrinks_small_samples_toward_prior():
    shots = [shot("g1", "Restricted Area", 1)] + [shot(f"g{i}", "Mid-Range", 0) for i in range(2, 12)]
    raw, prior = fit(shots, k=0)
    shrunk, _ = fit(shots, k=50)
    assert raw["Restricted Area"] == pytest.approx(1.0)
    assert 0.0 < shrunk["Restricted Area"] < raw["Restricted Area"]
    assert prior == pytest.approx(1 / 11)


def test_calibration_table_is_bounded():
    shots = [shot(f"g{i}", "Mid-Range", i % 2) for i in range(10)]
    baseline, fallback = fit(shots)
    table = calibration_table(shots, baseline, fallback)
    assert table
    for row in table:
        assert 0.0 <= row["predicted"] <= 1.0
        assert 0.0 <= row["observed"] <= 1.0


def test_report_returns_summary():
    shots = [shot(f"g{i}", "Restricted Area", i % 2) for i in range(1, 21)]
    summary = report(shots, label="test", test_frac=0.25)
    assert summary["shots"] == 20
    assert summary["metrics"]["n"] > 0
    assert "Restricted Area" in summary["zones"]
