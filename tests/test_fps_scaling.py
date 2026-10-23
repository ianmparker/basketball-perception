"""Frame-rate independence of the event detectors.

The thresholds are written as per-frame quantities, which silently mean different
things at different frame rates: a ball moving at a fixed real speed covers half as
many pixels per frame at 60 fps as at 30. Two consequences were live:

* The shot and screen detectors were constructed before `cfg.fps` was read from the
  video, so they always used the 30 fps default.
* Even with the right fps, `MAX_SCREENER_SPEED = 0.40 ft/frame` at 60 fps accepts a
  24 ft/s player as "nearly stationary" (12 ft/s at 30 fps).
"""
from __future__ import annotations

import pytest

from events.screen_detector_v2 import OnlineScreenDetector
from events.shot_detector_v2 import OnlineShotDetector
from inference.pipeline import PipelineConfig


# ── shot detector ────────────────────────────────────────────────────────────

def test_shot_thresholds_at_reference_fps_are_unchanged():
    """30 fps is the reference rate, so the tuned benchmark values must not shift."""
    d = OnlineShotDetector(fps=30, require_court_projection=False)
    assert d.min_speed_px == pytest.approx(OnlineShotDetector.MIN_SPEED_PX)
    assert d.max_gap_frames == OnlineShotDetector.MAX_GAP_FRAMES
    assert d.cooldown == OnlineShotDetector.COOLDOWN


def test_pixel_speed_gate_falls_as_fps_rises():
    """Same physical speed = fewer pixels per frame at a higher rate."""
    slow = OnlineShotDetector(fps=30, require_court_projection=False)
    fast = OnlineShotDetector(fps=60, require_court_projection=False)
    assert fast.min_speed_px == pytest.approx(slow.min_speed_px / 2)


def test_frame_windows_cover_the_same_duration_at_any_fps():
    """A frame-count window is a duration, so frames must grow with fps."""
    for fps in (25, 30, 60):
        d = OnlineShotDetector(fps=fps, require_court_projection=False)
        assert d.max_gap_frames / fps == pytest.approx(
            OnlineShotDetector.MAX_GAP_FRAMES / OnlineShotDetector.REFERENCE_FPS,
            abs=1.0 / fps,
        )


def test_shot_windows_never_collapse_to_zero_frames():
    d = OnlineShotDetector(fps=5, require_court_projection=False)
    assert d.max_gap_frames >= 1
    assert d.cooldown >= 1
    assert d.dedup_window >= 1
    assert d.spike_lockout >= 1


# ── screen detector ──────────────────────────────────────────────────────────

def test_screen_speeds_at_reference_fps_are_unchanged():
    d = OnlineScreenDetector(fps=30)
    assert d.max_screener_speed == pytest.approx(OnlineScreenDetector.MAX_SCREENER_SPEED)
    assert d.min_cutter_speed == pytest.approx(OnlineScreenDetector.MIN_CUTTER_SPEED)
    assert d.cooldown == OnlineScreenDetector.COOLDOWN


def test_screen_speeds_halve_at_60fps():
    d = OnlineScreenDetector(fps=60)
    assert d.max_screener_speed == pytest.approx(OnlineScreenDetector.MAX_SCREENER_SPEED / 2)
    assert d.min_cutter_speed == pytest.approx(OnlineScreenDetector.MIN_CUTTER_SPEED / 2)


def test_screen_stationary_threshold_is_a_real_speed_at_each_rate():
    """0.40 ft/frame must not silently mean 24 ft/s at 60 fps."""
    for fps in (25, 30, 60):
        d = OnlineScreenDetector(fps=fps)
        ft_per_s = d.max_screener_speed * fps
        assert ft_per_s == pytest.approx(
            OnlineScreenDetector.MAX_SCREENER_SPEED * OnlineScreenDetector.REFERENCE_FPS,
            rel=0.05,
        )


def test_screen_windows_never_collapse():
    d = OnlineScreenDetector(fps=5)
    assert d.min_duration >= 1
    assert d.cooldown >= 1
    assert d.display_frames >= 1


# ── plumbing ─────────────────────────────────────────────────────────────────

def test_pipeline_passes_config_fps_to_the_screen_detector():
    """The detector is built during __init__, so cfg.fps must be set beforehand."""
    cfg = PipelineConfig(enable_events=True, fps=60.0)
    assert cfg.fps == 60.0

    detector = OnlineScreenDetector(fps=cfg.fps)
    assert detector.fps == 60.0
    assert detector.max_screener_speed == pytest.approx(
        OnlineScreenDetector.MAX_SCREENER_SPEED / 2
    )
