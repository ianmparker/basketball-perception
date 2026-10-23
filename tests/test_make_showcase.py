"""Showcase cutting: which spans of the render get included.

A full render is ~190 MB of 1080p, which is too large to review or share. The
showcase keeps only the windows containing detections, so the selection logic is
what decides whether the demo actually shows the events.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_showcase import MAX_WINDOW_S, POST_ROLL_S, PRE_ROLL_S, _events, _windows


def test_events_are_read_from_a_clip_result(tmp_path):
    run = tmp_path / "run.json"
    run.write_text(
        json.dumps(
            {
                "shots": [{"frame_start": 439}, {"frame_start": 595}],
                "screens": [{"frame": 544}, {"frame": 560}],
            }
        ),
        encoding="utf-8",
    )
    shots, screens = _events(run)
    assert shots == [439, 595]
    assert screens == [544, 560]


def test_event_frames_are_sorted_even_if_the_json_is_not():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        run = Path(d) / "r.json"
        run.write_text(json.dumps({"shots": [{"frame_start": 900}, {"frame_start": 100}]}),
                       encoding="utf-8")
        shots, _ = _events(run)
        assert shots == [100, 900]


def test_a_window_brackets_each_event():
    fps = 30.0
    (lo, hi), = _windows([300], fps, total=1000)
    assert lo == int(300 - PRE_ROLL_S * fps)
    assert hi == int(300 + POST_ROLL_S * fps)


def test_windows_merge_when_events_are_close():
    """Two events a few frames apart should not produce two near-identical cuts."""
    fps = 30.0
    spans = _windows([300, 320], fps, total=1000)
    assert len(spans) == 1
    assert spans[0][0] <= 300 and spans[0][1] >= 320


def test_windows_stay_separate_when_events_are_far_apart():
    fps = 30.0
    spans = _windows([100, 900], fps, total=1000)
    assert len(spans) == 2


def test_windows_are_clamped_to_the_clip():
    fps = 30.0
    spans = _windows([5, 995], fps, total=1000)
    assert spans[0][0] == 0
    assert spans[-1][1] == 1000


def test_window_length_is_capped():
    fps = 30.0
    (lo, hi), = _windows([500], fps, total=5000)
    assert hi - lo <= MAX_WINDOW_S * fps


def test_no_events_still_yields_a_usable_window():
    spans = _windows([], 30.0, total=600)
    assert len(spans) == 1
    assert spans[0][0] == 0 and spans[0][1] > 0
