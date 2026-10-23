"""The broadcast demo page builds from a ClipResult without a video or a GPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_demo_page", ROOT / "scripts" / "build_demo_page.py"
)
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)

CLIP_SPEC = importlib.util.spec_from_file_location(
    "export_event_clips", ROOT / "scripts" / "export_event_clips.py"
)
clips = importlib.util.module_from_spec(CLIP_SPEC)
CLIP_SPEC.loader.exec_module(clips)


def sample_clip() -> dict:
    return {
        "schema_version": "1.3.0",
        "metadata": {
            "video": "test.mp4",
            "fps": 30.0,
            "width": 1920,
            "height": 1080,
            "frame_count": 2,
            "created_at": "2026-10-22T00:00:00",
        },
        "team_assignment_available": True,
        "shots": [
            {
                "frame_start": 1,
                "ball_court_ft": [75.0, 20.0],
                "shot_type": "mid_range",
                "shot_distance_ft": 15.0,
                "confidence": 0.8,
                "shooter_track_id": 7,
            }
        ],
        "screens": [
            {
                "frame": 0,
                "screen_type": "off_ball",
                "screener_id": 1,
                "cutter_id": 2,
                "separation_ft": 4.0,
                "confidence": 0.5,
            }
        ],
        "pnrs": [],
        "dunks": [],
        "passes": [],
        "interceptions": [],
        "frames": [
            {
                "ball": {"tracked": True, "det_box": [1, 2, 3, 4], "interpolated": False},
                "players": [
                    {"track_id": 7, "team_id": 1, "distance_traveled_ft": 12.0}
                ],
                "possession_team_id": 1,
                "court_view": True,
                "homography_quality": {"rmse_ft": 0.1},
            },
            {
                "ball": {"tracked": True, "det_box": None, "interpolated": True},
                "players": [
                    {"track_id": 7, "team_id": 1, "distance_traveled_ft": 15.0}
                ],
                "possession_team_id": None,
                "court_view": True,
                "homography_quality": {"rmse_ft": 0.2},
            },
        ],
    }


def test_summary_counts_events_and_tracking():
    summary = demo.summarize(sample_clip(), title="Test Demo")
    assert summary["event_counts"]["shots"] == 1
    assert summary["event_counts"]["screens"] == 1
    assert summary["ball"]["detected_frames"] == 1
    assert summary["ball"]["tracked_frames"] == 2
    assert summary["possession"]["frames"] == 1
    assert summary["teams"]["counts"]["1"] == 2
    assert summary["shots"][0]["x_ft"] == 75.0


def test_shot_chart_svg_contains_the_shot():
    summary = demo.summarize(sample_clip())
    svg = demo.shot_chart_svg(summary["shots"])
    assert svg.startswith("<svg")
    assert "Shot 1" in svg
    assert "<circle" in svg


def test_build_demo_writes_page_report_and_summary(tmp_path):
    run_json = tmp_path / "clip.json"
    run_json.write_text(json.dumps(sample_clip()), encoding="utf-8")
    out_dir = tmp_path / "demo"
    summary = demo.build_demo(run_json, None, out_dir, "Test Demo")

    assert (out_dir / "index.html").is_file()
    assert (out_dir / "shot_chart.svg").is_file()
    assert (out_dir / "report.md").is_file()
    assert (out_dir / "summary.json").is_file()
    assert (out_dir / "clip.json").is_file()
    assert summary["event_counts"]["shots"] == 1

    page = (out_dir / "index.html").read_text(encoding="utf-8")
    assert "Test Demo" in page
    assert "Shot chart" in page
    assert "Event timeline" in page
    assert "mid_range" in page
    assert "No rendered video was provided" in page


def test_video_toolbar_has_open_and_fullscreen_controls():
    summary = demo.summarize(sample_clip())
    page = demo.build_html(
        summary,
        demo.shot_chart_svg(summary["shots"]),
        "assets/demo.mp4",
        None,
        None,
        Path("."),
    )
    assert "Open video" in page
    assert 'id="fs-button"' in page
    assert "requestFullscreen" in page
    assert "theater" in page


def test_report_is_honest_about_limitations(tmp_path):
    run_json = tmp_path / "clip.json"
    run_json.write_text(json.dumps(sample_clip()), encoding="utf-8")
    out_dir = tmp_path / "demo"
    demo.build_demo(run_json, None, out_dir, "Test Demo")
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert "Known limitations" in report
    assert "not redistributed" in report


def test_event_clip_specs_cover_shots_screens_and_pnrs():
    data = sample_clip()
    data["pnrs"] = [{"frame": 1, "outcome": "drive", "confidence": 0.25}]
    events = clips._events(data)
    kinds = {event["kind"] for event in events}
    assert kinds == {"shot", "screen", "pnr"}
    assert all("frame" in event for event in events)


def test_export_clips_writes_manifest_and_video(tmp_path):
    import cv2

    video = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 180)
    )
    for i in range(30):
        frame = 255 * (i % 2) * np.ones((180, 320, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    run_json = tmp_path / "clip.json"
    run_json.write_text(json.dumps(sample_clip()), encoding="utf-8")
    out_dir = tmp_path / "clips"
    manifest = clips.export_clips(run_json, video, out_dir, max_clips=2, max_width=320)
    assert (out_dir / "clips.json").is_file()
    assert manifest["clips"]
    assert (out_dir / manifest["clips"][0]["file"]).is_file()
