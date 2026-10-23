"""Motion-aware ball picking.

Argmax confidence has no memory, so with several candidates per frame a confident
false positive wins the frame. On a 150-frame window the picker covered 38% of
frames with a 90th-percentile jump of 543 px; scoring candidates against a
predicted position (last centre + last velocity) covered 67% with p90 114 px.
"""
from __future__ import annotations

import numpy as np
import pytest

from tracking.simple_ball_tracker import SimpleBallTracker


def _dets(*rows):
    """rows are (cx, cy, conf) for Ball-class detections."""
    return np.asarray(
        [[cx - 5, cy - 5, cx + 5, cy + 5, conf, 0] for cx, cy, conf in rows],
        dtype=np.float32,
    )


def test_falls_back_to_confidence_when_there_is_no_history():
    dets = _dets((100, 100, 0.4), (500, 500, 0.9))
    box = SimpleBallTracker.pick_motion_aware(dets)
    assert box is not None
    assert (box[0] + box[2]) / 2 == pytest.approx(500)


def test_prefers_the_candidate_nearest_the_prediction_over_the_confident_one():
    """The confident candidate is a false positive far from the ball."""
    dets = _dets((105, 102, 0.30), (900, 700, 0.95))
    box = SimpleBallTracker.pick_motion_aware(
        dets, last=(100.0, 100.0), velocity=(5.0, 2.0)
    )
    assert (box[0] + box[2]) / 2 == pytest.approx(105)
    assert box[4] == pytest.approx(0.30)


def test_uses_velocity_to_predict_not_just_the_last_point():
    """A ball moving fast should be matched at its predicted position."""
    dets = _dets((103, 100, 0.80), (140, 100, 0.55))
    # Predicted centre is 100 + 40 = 140, so the second candidate wins despite
    # being less confident than the one sitting on the last position.
    box = SimpleBallTracker.pick_motion_aware(
        dets, last=(100.0, 100.0), velocity=(40.0, 0.0)
    )
    assert (box[0] + box[2]) / 2 == pytest.approx(140)


def test_ignores_non_ball_classes():
    dets = np.asarray([[0, 0, 10, 10, 0.99, 4]], dtype=np.float32)  # class 4 = Player
    assert SimpleBallTracker.pick_motion_aware(dets) is None


def test_returns_none_on_empty_input():
    assert SimpleBallTracker.pick_motion_aware(np.empty((0, 6), np.float32)) is None
    assert SimpleBallTracker.pick_motion_aware(None) is None


def test_confidence_only_breaks_near_ties():
    """A tie on distance should go to the more confident candidate."""
    dets = _dets((100, 100, 0.40), (100, 100, 0.80))
    box = SimpleBallTracker.pick_motion_aware(dets, last=(100.0, 100.0), velocity=(0.0, 0.0))
    assert box[4] == pytest.approx(0.80)


def test_returns_the_full_box_not_just_a_point():
    dets = _dets((50, 60, 0.7))
    x1, y1, x2, y2, conf = SimpleBallTracker.pick_motion_aware(dets)
    assert (x1, y1, x2, y2) == (45.0, 55.0, 55.0, 65.0)
