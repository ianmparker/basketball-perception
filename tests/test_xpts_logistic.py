import math

import pytest

from analytics.xpts_logistic import (
    calibration,
    evaluate,
    features,
    fit,
    predict,
    report,
    sigmoid,
)


def shot(game, distance, made, shot_type="2PT Field Goal", zone="Restricted Area"):
    return {
        "game_id": game,
        "game_date": "2024-01-01",
        "shot_distance": distance,
        "loc_x": 0,
        "loc_y": distance * 10,
        "shot_type": shot_type,
        "zone_basic": zone,
        "made": made,
    }


def synthetic():
    shots = []
    for i in range(10):
        for j in range(5):
            shots.append(shot(f"g{i}", 2, 1))
            shots.append(shot(f"g{i}", 25, 0, "3PT Field Goal", "Above the Break 3"))
    return shots


def test_features_are_finite():
    for s in synthetic():
        row = features(s)
        assert len(row) == 6
        assert all(math.isfinite(v) for v in row)


def test_sigmoid_bounds_and_midpoint():
    assert sigmoid(0) == pytest.approx(0.5)
    assert 0 < sigmoid(-20) < 0.001
    assert 0.999 < sigmoid(20) < 1


def test_logistic_learns_direction():
    shots = synthetic()
    model = fit(shots)
    near = predict(shot("g0", 2, 1), model)
    deep = predict(shot("g0", 25, 0, "3PT Field Goal", "Above the Break 3"), model)
    assert near > deep
    assert 0.5 < near <= 1.0
    assert 0.0 <= deep < 0.5


def test_evaluate_returns_finite_metrics():
    shots = synthetic()
    model = fit(shots)
    metrics = evaluate(shots, model)
    assert metrics["n"] == len(shots)
    assert 0 <= metrics["brier"] <= 1
    assert metrics["log_loss"] > 0


def test_calibration_bins_are_bounded():
    shots = synthetic()
    model = fit(shots)
    bins = calibration(shots, model)
    assert bins
    for row in bins:
        assert 0 <= row["predicted"] <= 1
        assert 0 <= row["observed"] <= 1


def test_report_returns_summary():
    shots = synthetic()
    summary = report(shots, label="synthetic", test_frac=0.3)
    assert summary["zone"]["n"] > 0
    assert summary["logistic"]["n"] > 0
    assert "distance" in summary["weights"]
