"""Zone-baseline expected points (xPTS) for NBA shot-chart data.

This is the v0 shot-quality model: a shrunk zone-average points-per-shot baseline.
It is deliberately simple and interpretable. The next step is a logistic model with
distance, angle, contest, and shooter effects.

Shot records are the normalized rows produced by `scripts/nba_shot_ingest.py`.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable

ZONES = [
    "Restricted Area",
    "Paint (non-RA)",
    "Mid-Range",
    "Corner 3",
    "Above the Break 3",
    "Backcourt",
    "Other",
]
ZONE_ORDER = {z: i for i, z in enumerate(ZONES)}


def canonical_zone(shot: dict) -> str:
    """Collapse the NBA's basic zones into a small, stable set."""
    zone = (shot.get("zone_basic") or "").strip()
    if zone in ("Left Corner 3", "Right Corner 3"):
        return "Corner 3"
    if zone == "In The Paint (Non-RA)":
        return "Paint (non-RA)"
    if zone in ZONES:
        return zone
    return "Other"


def shot_points(shot: dict) -> int:
    """2 or 3, from the NBA's shot type."""
    return 3 if str(shot.get("shot_type", "")).startswith("3") else 2


def _made(shot: dict) -> int:
    return int(shot.get("made", 0) or 0)


def summarize(shots: Iterable[dict]) -> dict[str, dict]:
    """Raw counts, FG%, and xPTS per canonical zone."""
    counts: dict[str, dict] = defaultdict(lambda: {"attempts": 0, "makes": 0, "points": 0})
    for shot in shots:
        zone = canonical_zone(shot)
        pts = shot_points(shot)
        counts[zone]["attempts"] += 1
        counts[zone]["makes"] += _made(shot)
        counts[zone]["points"] += pts
    table: dict[str, dict] = {}
    for zone, row in counts.items():
        attempts = row["attempts"]
        makes = row["makes"]
        # All shots in a zone share a point value in practice; use the mean if not.
        points = row["points"] / attempts if attempts else 0.0
        table[zone] = {
            "attempts": attempts,
            "makes": makes,
            "fg": makes / attempts if attempts else 0.0,
            "points": points,
            "xpts": (makes / attempts) * points if attempts else 0.0,
        }
    return dict(sorted(table.items(), key=lambda kv: ZONE_ORDER.get(kv[0], 99)))


def _global_prior(shots: Iterable[dict]) -> float:
    """Overall FG% across all shots; used for the 2/3 fallback."""
    attempts = makes = 0
    for shot in shots:
        attempts += 1
        makes += _made(shot)
    return makes / attempts if attempts else 0.5


def fit(shots: Iterable[dict], k: float = 0.0) -> tuple[dict[str, float], float]:
    """Fit zone FG% with optional shrinkage toward the global prior.

    Returns (zone_probabilities, fallback_probability). `k` is the pseudo-count:
    k=0 is the raw zone average; larger k shrinks small samples toward the global mean.
    """
    shots = list(shots)
    prior = _global_prior(shots)
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [makes, attempts]
    for shot in shots:
        zone = canonical_zone(shot)
        counts[zone][0] += _made(shot)
        counts[zone][1] += 1
    baseline = {}
    for zone, (makes, attempts) in counts.items():
        baseline[zone] = (makes + k * prior) / (attempts + k) if (attempts + k) else prior
    return baseline, prior


def predict(shot: dict, baseline: dict[str, float], fallback: float) -> float:
    return float(baseline.get(canonical_zone(shot), fallback))


def evaluate(shots: Iterable[dict], baseline: dict[str, float], fallback: float) -> dict:
    """Brier score and log loss for a probabilistic baseline."""
    shots = list(shots)
    if not shots:
        return {"n": 0, "brier": 0.0, "log_loss": 0.0}
    brier = 0.0
    logloss = 0.0
    for shot in shots:
        y = _made(shot)
        p = min(max(predict(shot, baseline, fallback), 1e-6), 1 - 1e-6)
        brier += (p - y) ** 2
        logloss += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    n = len(shots)
    return {"n": n, "brier": brier / n, "log_loss": logloss / n}


def calibration_table(shots: Iterable[dict], baseline: dict[str, float], fallback: float, bins: int = 5) -> list[dict]:
    """Predicted vs observed rate in equal-count bins."""
    rows = sorted(
        ((predict(s, baseline, fallback), _made(s)) for s in shots),
        key=lambda x: x[0],
    )
    if not rows:
        return []
    out = []
    size = max(1, len(rows) // bins)
    for i in range(0, len(rows), size):
        chunk = rows[i:i + size]
        if not chunk:
            continue
        out.append({
            "n": len(chunk),
            "predicted": sum(p for p, _ in chunk) / len(chunk),
            "observed": sum(y for _, y in chunk) / len(chunk),
        })
    return out


def grouped_split(shots: Iterable[dict], test_frac: float = 0.2, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split by game, never by shot. Prevents the same game leaking into train and test."""
    shots = list(shots)
    games = sorted({str(s.get("game_id", "")) for s in shots})
    if len(games) < 2:
        raise ValueError("need at least 2 games for a grouped split")
    rng = random.Random(seed)
    rng.shuffle(games)
    n_test = max(1, int(round(len(games) * test_frac)))
    test_games = set(games[:n_test])
    train = [s for s in shots if str(s.get("game_id", "")) not in test_games]
    test = [s for s in shots if str(s.get("game_id", "")) in test_games]
    return train, test


def report(shots: Iterable[dict], label: str = "shots", k: float = 50.0, test_frac: float = 0.2) -> dict:
    """Print a zone table plus grouped train/test metrics. Returns a JSON-able summary."""
    shots = list(shots)
    zones = summarize(shots)
    total = len(shots)
    games = {str(s.get("game_id", "")) for s in shots}
    dates = sorted(str(s.get("game_date", "")) for s in shots if s.get("game_date"))

    print(f"\n{label}")
    print(f"  shots={total}  games={len(games)}  dates={dates[0] if dates else '?'}..{dates[-1] if dates else '?'}")
    print(f"  {'zone':<20} {'att':>6} {'share':>7} {'fg%':>7} {'xPTS':>7}")
    for zone, row in zones.items():
        share = row["attempts"] / total if total else 0.0
        print(f"  {zone:<20} {row['attempts']:>6} {share:>6.1%} {row['fg']:>7.3f} {row['xpts']:>7.3f}")

    train, test = grouped_split(shots, test_frac=test_frac)
    baseline, fallback = fit(train, k=k)
    metrics = evaluate(test, baseline, fallback)
    print(f"\n  grouped split: train={len(train)} shots / {len({s['game_id'] for s in train})} games  "
          f"test={len(test)} shots / {len({s['game_id'] for s in test})} games")
    print(f"  test Brier={metrics['brier']:.4f}  log_loss={metrics['log_loss']:.4f}  n={metrics['n']}")
    print(f"  {'bin':<20} {'n':>6} {'pred':>7} {'obs':>7}")
    for i, row in enumerate(calibration_table(test, baseline, fallback)):
        print(f"  bin {i:<16} {row['n']:>6} {row['predicted']:>7.3f} {row['observed']:>7.3f}")

    return {
        "label": label,
        "shots": total,
        "games": len(games),
        "zones": zones,
        "split": {"train_shots": len(train), "test_shots": len(test), "test_frac": test_frac},
        "metrics": metrics,
        "calibration": calibration_table(test, baseline, fallback),
    }
