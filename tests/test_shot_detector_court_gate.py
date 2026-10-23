import numpy as np

from events.shot_detector_v2 import OnlineShotDetector


def _feed(detector, frames, *, x=100.0, start_y=600.0, dy=12.0, h=1080):
    """Drive the detector with a ball moving upward and a player beneath it."""
    events = []
    for i in range(frames):
        tracks = np.array([[80, 400, 140, 700, 7, 0.9, 0, 0]], dtype=np.float32)
        ev = detector.update(
            frame_idx=i,
            ball_cx=x,
            ball_cy=start_y - dy * i,
            ball_tracked=True,
            ball_coasted=False,
            frame_h=h,
            tracks=tracks,
            H=None,
            pose_kps=None,
            pose_kpc=None,
            pose_boxes=None,
        )
        if ev is not None:
            events.append(ev)
    return events


def test_court_gate_blocks_detection_without_homography_by_default():
    detector = OnlineShotDetector(fps=30)
    assert detector.require_court_projection is True
    assert _feed(detector, 20) == []


def test_relaxed_gate_fires_without_homography():
    detector = OnlineShotDetector(fps=30, require_court_projection=False)
    events = _feed(detector, 20)
    assert events, "expected a shot event when the court gate is relaxed"
    assert events[0].ball_court_ft is None
    assert events[0].shot_distance_ft is None
    assert events[0].shot_type is None


def test_relaxed_gate_still_requires_a_nearby_player():
    detector = OnlineShotDetector(fps=30, require_court_projection=False)
    frames = []
    for i in range(20):
        ev = detector.update(
            frame_idx=i,
            ball_cx=100.0,
            ball_cy=600.0 - 12.0 * i,
            ball_tracked=True,
            ball_coasted=False,
            frame_h=1080,
            tracks=np.empty((0, 8), dtype=np.float32),
            H=None,
            pose_kps=None,
            pose_kpc=None,
            pose_boxes=None,
        )
        if ev is not None:
            frames.append(ev)
    assert frames == []
