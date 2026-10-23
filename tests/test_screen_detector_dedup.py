"""Screen-detector regressions found while scoring the broadcast clip.

Both bugs are logic errors, not threshold choices, so they can be fixed and
asserted without any positive screen ground truth (we currently have none).
"""
from collections import deque

import pytest

from events.screen_detector_v2 import OnlineScreenDetector, track_speeds


def test_speed_is_normalised_by_the_frame_gap():
    """A track missing 2 of every 3 frames must not look 3x faster."""
    hist = {7: deque([(0, 0.0, 0.0), (3, 0.9, 0.0)], maxlen=12)}

    speed = track_speeds(hist)[7]

    assert speed == pytest.approx(0.3)
    # Unnormalised, the same motion reads as 0.9 ft/frame and would disqualify a
    # genuinely stationary screener.
    assert speed < OnlineScreenDetector.MAX_SCREENER_SPEED


def test_speed_of_dense_history_is_unchanged():
    hist = {1: deque([(0, 0.0, 0.0), (1, 0.3, 0.0), (2, 0.6, 0.0)], maxlen=12)}

    assert track_speeds(hist)[1] == pytest.approx(0.3)


def test_speed_is_zero_for_a_stationary_track():
    hist = {2: deque([(0, 5.0, 5.0), (1, 5.0, 5.0), (2, 5.0, 5.0)], maxlen=12)}

    assert track_speeds(hist)[2] == pytest.approx(0.0)


def test_pair_key_is_order_independent():
    assert OnlineScreenDetector._pair_key(3, 9) == OnlineScreenDetector._pair_key(9, 3)


def _approach(detector, frames, screener_id, cutter_id, start_sep=5.0):
    """Hold `screener_id` still while `cutter_id` walks straight at it."""
    events = []
    for f in frames:
        elapsed = f - frames[0]
        players = [
            (10.0, 10.0, screener_id),
            (10.0 + max(start_sep - 0.4 * elapsed, 0.0), 10.0, cutter_id),
        ]
        events += detector.update(f, players, None)
    return events


def test_a_screen_is_detected_for_a_clean_approach():
    detector = OnlineScreenDetector()
    events = _approach(detector, range(60), screener_id=5, cutter_id=6)
    assert len(events) == 1
    assert (events[0].screener_id, events[0].cutter_id) == (5, 6)


def test_the_same_collision_cannot_fire_with_roles_swapped():
    """Physically impossible: A cannot screen B while B screens A."""
    detector = OnlineScreenDetector()
    first = _approach(detector, range(60), screener_id=5, cutter_id=6)
    assert len(first) == 1

    # Now the roles invert: 6 stands still and 5 walks at it, well inside COOLDOWN.
    inverted = _approach(detector, range(60, 100), screener_id=6, cutter_id=5)
    assert inverted == []


def test_cooldown_uses_the_unordered_pair():
    detector = OnlineScreenDetector()
    _approach(detector, range(60), screener_id=5, cutter_id=6)

    assert detector._pair_key(5, 6) in detector._cooldown
    assert detector._pair_key(6, 5) in detector._cooldown


def test_reset_clears_cooldowns():
    detector = OnlineScreenDetector()
    _approach(detector, range(60), screener_id=5, cutter_id=6)
    assert detector._cooldown

    detector.reset()

    assert detector._cooldown == {}
    # A scene cut must let the same pair screen again.
    again = _approach(detector, range(60, 120), screener_id=5, cutter_id=6)
    assert len(again) == 1
