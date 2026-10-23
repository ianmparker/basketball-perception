"""The shot-labeling app writes labels and reports progress."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "label_broadcast_events", ROOT / "scripts" / "label_broadcast_events.py"
)
labeler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(labeler)


def sample_labels() -> dict:
    return {
        "version": 1,
        "status": "template",
        "runs": [
            {
                "id": "r1",
                "clip": "r1",
                "events": [
                    {
                        "id": "e1",
                        "predicted_frame": 100,
                        "label": None,
                        "release_frame": None,
                        "shooter_track_id_gt": None,
                        "made": None,
                        "notes": "",
                        "clip": None,
                    },
                    {
                        "id": "e2",
                        "predicted_frame": 200,
                        "label": None,
                        "release_frame": None,
                        "shooter_track_id_gt": None,
                        "made": None,
                        "notes": "",
                        "clip": None,
                    },
                ],
            }
        ],
    }


def test_progress_and_status_flip_to_labeled():
    data = sample_labels()
    assert labeler.progress(data) == {"total": 2, "labeled": 0, "remaining": 2}
    labeler.update_event(data, "r1", "e1", {"label": 1})
    assert data["status"] == "in_progress"
    assert labeler.progress(data)["labeled"] == 1
    labeler.update_event(data, "r1", "e2", {"label": 0})
    assert data["status"] == "labeled"
    assert labeler.progress(data)["remaining"] == 0


def test_bad_label_is_rejected():
    data = sample_labels()
    try:
        labeler.update_event(data, "r1", "e1", {"label": 2})
    except ValueError:
        pass
    else:
        raise AssertionError("label 2 should be rejected")


def test_gather_label_is_accepted():
    data = sample_labels()
    labeler.update_event(data, "r1", "e1", {"label": "gather"})
    assert data["runs"][0]["events"][0]["label"] == "gather"
    assert data["status"] == "in_progress"


def test_clip_path_rejects_non_video_and_outside_paths(tmp_path):
    data = sample_labels()
    data["runs"][0]["events"][0]["clip"] = "README.md"
    assert labeler.clip_path(data, "r1", "e1") is None
    data["runs"][0]["events"][0]["clip"] = "../../outside.mp4"
    assert labeler.clip_path(data, "r1", "e1") is None


def test_app_serves_page_and_writes_label(tmp_path):
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(sample_labels()), encoding="utf-8")
    app = labeler.create_app(path)
    client = app.test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Broadcast Shot Labeling" in page.data

    labels = client.get("/api/labels").get_json()
    assert labels["runs"][0]["events"][0]["label"] is None

    response = client.post(
        "/api/label",
        json={"run_id": "r1", "event_id": "e1", "label": 1, "notes": "test"},
    )
    assert response.status_code == 200
    assert response.get_json()["labeled"] == 1

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["runs"][0]["events"][0]["label"] == 1
    assert saved["runs"][0]["events"][0]["notes"] == "test"
    assert saved["status"] == "in_progress"

    health = client.get("/health").get_json()
    assert health["total"] == 2
    assert health["labeled"] == 1
