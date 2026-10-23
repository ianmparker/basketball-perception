"""Event-level scoring for labeled broadcast shot candidates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "score_broadcast_events", ROOT / "scripts" / "score_broadcast_events.py"
)
score = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score)


def run(events: list[dict]) -> dict:
    return {"id": "test", "run_json": "x.json", "events": events}


def test_perfect_match():
    events = [
        {"label": 1, "predicted_frame": 100, "release_frame": 100},
        {"label": 1, "predicted_frame": 200, "release_frame": 200},
    ]
    result = score.score_run(run(events), [100, 200], tolerance=5)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0


def test_false_positive_and_miss():
    events = [
        {"label": 0, "predicted_frame": 100},
        {"label": 1, "predicted_frame": None, "release_frame": 200},
    ]
    result = score.score_run(run(events), [100], tolerance=5)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)
    assert result["precision"] == 0.0
    assert result["recall"] == 0.0


def test_tolerance_matching_uses_release_frame():
    events = [{"label": 1, "predicted_frame": 100, "release_frame": 110}]
    assert score.score_run(run(events), [100], tolerance=15)["tp"] == 1
    assert score.score_run(run(events), [100], tolerance=5)["fn"] == 1


def test_unlabeled_candidates_are_skipped():
    events = [{"label": None, "predicted_frame": 100}]
    result = score.score_run(run(events), [100], tolerance=5)
    assert result["labeled"] == 0
    assert result["gt"] == 0


def test_gather_only_is_excluded_from_precision_and_recall():
    events = [{"label": "gather", "predicted_frame": 100}]
    result = score.score_run(run(events), [100], tolerance=5)
    assert result["labeled"] == 0
    assert result["gather_only"] == 1
    assert result["tp"] == 0
    assert result["fp"] == 0


def test_scorecard_counts_unique_events_not_tolerance_rows():
    labels = {
        "status": "labeled",
        "runs": [
            {
                "id": "r1",
                "run_json": "x.json",
                "events": [
                    {"label": 1, "predicted_frame": 100, "release_frame": 100},
                    {"label": 0, "predicted_frame": 200},
                ],
            }
        ],
    }
    result = score.score_labels(labels, [5, 15])
    text = score.render_scorecard(result, labels)
    assert "**Judged candidates:** 2" in text
    assert "**Judged candidates:** 4" not in text


def test_scorecard_groups_regimes_separately(tmp_path):
    run_json = tmp_path / "run.json"
    run_json.write_text(
        json.dumps({"shots": [{"frame_start": 100}, {"frame_start": 200}]}),
        encoding="utf-8",
    )
    labels = {
        "status": "labeled",
        "runs": [
            {
                "id": "nba",
                "regime": "nba_broadcast",
                "run_json": str(run_json),
                "events": [
                    {"label": 1, "predicted_frame": 100, "release_frame": 100}
                ],
            },
            {
                "id": "hs",
                "regime": "hs_top_down",
                "run_json": str(run_json),
                "events": [{"label": 0, "predicted_frame": 200}],
            },
        ],
    }
    result = score.score_labels(labels, [5])
    text = score.render_scorecard(result, labels)
    assert "NBA broadcast" in text
    assert "HS top-down" in text
    assert "All runs (mixed regimes)" in text
    assert "0.50" in text


def test_wilson_interval_is_bounded():
    lo, hi = score.wilson_interval(3, 3)
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == 1.0


def test_render_without_labels_is_honest():
    labels = {
        "status": "template",
        "runs": [{"id": "r1", "run_json": "x.json", "events": [{"label": None}]}],
    }
    result = score.score_labels(labels, [15])
    text = score.render_scorecard(result, labels)
    assert "No labels yet" in text
    assert "template" in text
