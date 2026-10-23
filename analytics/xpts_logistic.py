"""Logistic expected points (xPTS) for NBA shot-chart data.

A small, dependency-free logistic regression over interpretable shot features:
distance, distance squared, angle, 3-point indicator, corner-3 indicator, and
restricted-area indicator. Trained with batch gradient descent and L2 regularization.

The zone baseline in `analytics/xpts.py` is the bar. Evaluate on a game-grouped split;
never split by shot.
"""
from __future__ import annotations

import math
import random
from typing import Iterable

from analytics.xpts import (
    canonical_zone,
    evaluate as evaluate_zone,
    fit as fit_zone,
    grouped_split,
    shot_points,
)

FEATURES = ["distance", "distance_sq", "angle", "is_three", "is_corner_three", "is_restricted"]


def features(shot: dict) -> list[float]:
    """Feature vector for one shot. All values are finite and scale-stable."""
    distance = float(shot.get("shot_distance", 0) or 0) / 10.0
    x = abs(float(shot.get("loc_x", 0) or 0)) / 10.0
    y = max(float(shot.get("loc_y", 0) or 0) / 10.0, 0.1)
    angle = math.atan2(x, y)
    zone = canonical_zone(shot)
    return [
        distance,
        distance * distance,
        angle,
        1.0 if shot_points(shot) == 3 else 0.0,
        1.0 if zone == "Corner 3" else 0.0,
        1.0 if zone == "Restricted Area" else 0.0,
    ]


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _standardize(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    n = len(rows)
    m = len(rows[0]) if rows else 0
    mean = [sum(row[j] for row in rows) / n for j in range(m)]
    std = []
    for j in range(m):
        var = sum((row[j] - mean[j]) ** 2 for row in rows) / n
        std.append(math.sqrt(var) or 1.0)
    return mean, std


def _scale(row: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(v - mean[j]) / std[j] for j, v in enumerate(row)]


def train(
    rows: list[list[float]],
    labels: list[int],
    *,
    l2: float = 0.01,
    lr: float = 0.05,
    epochs: int = 1500,
) -> tuple[list[float], float, list[float], list[float]]:
    """Batch gradient descent. Returns (weights, bias, mean, std)."""
    if not rows:
        raise ValueError("no training rows")
    mean, std = _standardize(rows)
    X = [_scale(row, mean, std) for row in rows]
    n, m = len(X), len(X[0])
    weights = [0.0] * m
    base = min(max(sum(labels) / n, 1e-6), 1 - 1e-6)
    bias = math.log(base / (1 - base))
    for _ in range(epochs):
        grad_w = [0.0] * m
        grad_b = 0.0
        for row, y in zip(X, labels):
            p = sigmoid(bias + sum(w * x for w, x in zip(weights, row)))
            err = p - y
            grad_b += err
            for j in range(m):
                grad_w[j] += err * row[j]
        for j in range(m):
            weights[j] -= lr * (grad_w[j] / n + l2 * weights[j])
        bias -= lr * grad_b / n
    return weights, bias, mean, std


def fit(shots: Iterable[dict], **kwargs) -> dict:
    shots = list(shots)
    rows = [features(s) for s in shots]
    labels = [int(s.get("made", 0) or 0) for s in shots]
    weights, bias, mean, std = train(rows, labels, **kwargs)
    return {"features": FEATURES, "weights": weights, "bias": bias, "mean": mean, "std": std}


def predict(shot: dict, model: dict) -> float:
    row = _scale(features(shot), model["mean"], model["std"])
    return sigmoid(model["bias"] + sum(w * x for w, x in zip(model["weights"], row)))


def evaluate(shots: Iterable[dict], model: dict) -> dict:
    shots = list(shots)
    if not shots:
        return {"n": 0, "brier": 0.0, "log_loss": 0.0}
    brier = logloss = 0.0
    for shot in shots:
        y = int(shot.get("made", 0) or 0)
        p = min(max(predict(shot, model), 1e-6), 1 - 1e-6)
        brier += (p - y) ** 2
        logloss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    n = len(shots)
    return {"n": n, "brier": brier / n, "log_loss": logloss / n}


def calibration(shots: Iterable[dict], model: dict, bins: int = 5) -> list[dict]:
    pairs = sorted(((predict(s, model), int(s.get("made", 0) or 0)) for s in shots), key=lambda x: x[0])
    if not pairs:
        return []
    out = []
    size = max(1, len(pairs) // bins)
    for i in range(0, len(pairs), size):
        chunk = pairs[i:i + size]
        if chunk:
            out.append({
                "n": len(chunk),
                "predicted": sum(p for p, _ in chunk) / len(chunk),
                "observed": sum(y for _, y in chunk) / len(chunk),
            })
    return out


def report(shots: Iterable[dict], label: str = "shots", test_frac: float = 0.2, seed: int = 42) -> dict:
    """Compare the zone baseline and logistic model on one game-grouped split."""
    shots = list(shots)
    train_shots, test_shots = grouped_split(shots, test_frac=test_frac, seed=seed)
    if not train_shots or not test_shots:
        raise ValueError("grouped split produced an empty side")

    zone_model, zone_fallback = fit_zone(train_shots, k=50)
    zone_metrics = evaluate_zone(test_shots, zone_model, zone_fallback)

    logistic_model = fit(train_shots)
    logistic_metrics = evaluate(test_shots, logistic_model)
    bins = calibration(test_shots, logistic_model)

    print(f"\n{label}")
    print(f"  train={len(train_shots)} shots / {len({s['game_id'] for s in train_shots})} games  "
          f"test={len(test_shots)} shots / {len({s['game_id'] for s in test_shots})} games")
    print(f"  {'model':<10} {'brier':>8} {'log_loss':>9} {'n':>6}")
    print(f"  {'zone':<10} {zone_metrics['brier']:>8.4f} {zone_metrics['log_loss']:>9.4f} {zone_metrics['n']:>6}")
    print(f"  {'logistic':<10} {logistic_metrics['brier']:>8.4f} {logistic_metrics['log_loss']:>9.4f} {logistic_metrics['n']:>6}")
    print(f"  {'bin':<10} {'n':>6} {'pred':>7} {'obs':>7}")
    for i, row in enumerate(bins):
        print(f"  bin {i:<6} {row['n']:>6} {row['predicted']:>7.3f} {row['observed']:>7.3f}")

    return {
        "label": label,
        "train_shots": len(train_shots),
        "test_shots": len(test_shots),
        "zone": zone_metrics,
        "logistic": logistic_metrics,
        "calibration": bins,
        "weights": dict(zip(FEATURES, logistic_model["weights"])),
    }
