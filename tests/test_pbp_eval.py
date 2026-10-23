"""Event-level matching for the PBP shot benchmark."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "eval_pbp_shots", ROOT / "scripts" / "eval_pbp_shots.py"
)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def record(kind="positive", gt=180, preds=None):
    return {
        "kind": kind,
        "gt_offset": gt,
        "predicted_frames": preds or [],
    }


def test_perfect_match_and_extra_shot():
    result = EVAL.score([record(preds=[180])], 30)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 0, 0)
    result = EVAL.score([record(preds=[180, 210])], 30)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 0)


def test_miss_and_negative_false_positive():
    records = [record(preds=[]), record(kind="negative", gt=None, preds=[100])]
    result = EVAL.score(records, 30)
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


def test_tolerance_controls_match():
    assert EVAL.score([record(preds=[210])], 30)["tp"] == 1
    assert EVAL.score([record(preds=[210])], 15)["fn"] == 1


def test_offset_is_prediction_minus_ground_truth():
    result = EVAL.score([record(preds=[200])], 30)
    assert result["offset_median"] == 20
    assert result["offset_mean"] == 20.0


def test_audit_excludes_misaligned_and_adjacent(tmp_path):
    pos_root = tmp_path / "pos"
    neg_root = tmp_path / "neg"
    results = tmp_path / "results"
    for path in (pos_root, neg_root, results):
        path.mkdir()
    positives = {
        "_root": str(pos_root),
        "events": [
            {"event_id": "p_keep", "video": "p1.mp4", "frame": 100, "clip_start_frame": 0},
            {"event_id": "p_drop", "video": "p2.mp4", "frame": 100, "clip_start_frame": 0},
        ],
    }
    negatives = {
        "_root": str(neg_root),
        "items": [
            {"event_id": "n_keep", "video": "n1.mp4"},
            {"event_id": "n_adjacent", "video": "n2.mp4"},
            {"event_id": "n_misaligned", "video": "n3.mp4"},
        ],
    }
    for event in positives["events"]:
        (pos_root / event["video"]).write_bytes(b"")
    for event in negatives["items"]:
        (neg_root / event["video"]).write_bytes(b"")
    for event_id in ("p_keep", "p_drop", "n_keep", "n_adjacent", "n_misaligned"):
        (results / f"{event_id}.json").write_text(
            json.dumps({"shots": [{"frame_start": 10}]}), encoding="utf-8"
        )

    audit = {
        "p_drop": {"verdict": "misaligned"},
        "n_adjacent": {"verdict": "adjacent"},
        "n_misaligned": {"verdict": "misaligned"},
    }
    records, summary = EVAL.clip_records(
        positives, negatives, results, refresh=False, timeout=1, audit=audit
    )

    assert [r["event_id"] for r in records] == ["p_keep", "n_keep"]
    assert summary == {
        "audited": 3,
        "positives": ["p_drop"],
        "negatives": ["n_adjacent", "n_misaligned"],
    }
