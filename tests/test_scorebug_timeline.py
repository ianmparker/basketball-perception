"""Scorebug timeline alignment helpers."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_scorebug_timeline", ROOT / "scripts" / "build_scorebug_timeline.py"
)
TIMELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TIMELINE)


def test_period_ranges_from_manual_anchors():
    manual = {
        "clock_anchors": [
            {"frame": 100, "period": 1, "clock_s": 720},
            {"frame": 200, "period": 1, "clock_s": 600},
            {"frame": 300, "period": 2, "clock_s": 720},
            {"frame": 400, "period": 2, "clock_s": 600},
        ]
    }
    assert TIMELINE.period_ranges(manual) == {1: (100, 200), 2: (300, 400)}


def test_align_events_uses_matching_clock_and_cuts_clip(tmp_path, monkeypatch):
    calls = []

    def fake_cut(video, out, start_s, duration_s, width, skip, fps=None):
        calls.append((out, start_s, duration_s, width, skip, fps))

    monkeypatch.setattr(TIMELINE, "cut_clip", fake_cut)
    timeline = {
        "video": "dummy.mp4",
        "fps": 30.0,
        "total_frames": 1000,
        "samples": [
            {"frame": 300, "clock_s": 500.0, "period": 1, "confidence": 0.9},
            {"frame": 301, "clock_s": 500.0, "period": 1, "confidence": 0.9},
        ],
    }
    source = {
        "event_id": "401810030",
        "events": [{"event_id": "shot_x", "period": 1, "clock_s": 500.0, "frame": 310}],
    }
    out = tmp_path / "validation_ocr.json"
    clips = tmp_path / "clips"
    clips.mkdir()
    payload = TIMELINE.align_events(
        source, timeline, out, clips, before_s=6, after_s=4,
        clip_width=640, tolerance_s=1.5, skip_existing=False,
    )
    event = payload["events"][0]
    assert payload["count"] == 1
    assert event["frame"] in (300, 301)
    assert event["clip_start_frame"] == event["frame"] - 180
    assert event["clip_end_frame"] == event["frame"] + 120
    assert event["ocr"]["aligned"] is True
    assert event["video"] == "clips/shot_x.mp4"
    assert len(calls) == 1
    assert json.loads(out.read_text(encoding="utf-8"))["count"] == 1


def test_align_events_resamples_and_uses_chronological_clock(tmp_path, monkeypatch):
    calls = []

    def fake_cut(video, out, start_s, duration_s, width, skip, fps=None):
        calls.append((start_s, duration_s, fps))

    monkeypatch.setattr(TIMELINE, "cut_clip", fake_cut)
    timeline = {
        "video": "dummy.mp4",
        "fps": 60.0,
        "total_frames": 2000,
        "samples": [
            {"frame": 300, "clock_s": 500.0, "period": 1},
            {"frame": 301, "clock_s": 500.0, "period": 1},
        ],
    }
    source = {
        "event_id": "401766463",
        "events": [{"event_id": "shot_x", "period": 1, "clock_s": 500.0, "frame": 0}],
    }
    out = tmp_path / "validation_ocr.json"
    clips = tmp_path / "clips"
    clips.mkdir()
    payload = TIMELINE.align_events(
        source, timeline, out, clips, before_s=6, after_s=4,
        clip_width=640, tolerance_s=1.5, skip_existing=False,
        clip_fps=30, chronological=True,
    )
    event = payload["events"][0]
    assert payload["fps"] == 30
    assert payload["source_fps"] == 60
    assert event["source_frame"] == 300
    assert event["frame"] == 150
    assert event["clip_start_frame"] == 0
    assert calls[0][2] == 30
