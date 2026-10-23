import numpy as np

from events.shot_detector_v2 import OnlineShotDetector

TRACKS = np.array([[80, 400, 140, 900, 7, 0.9, 0, 0]], dtype=np.float32)


def _detector(**kwargs):
    defaults = {"require_court_projection": False}
    defaults.update(kwargs)
    return OnlineShotDetector(fps=30, **defaults)


def _feed(detector, samples):
    """samples: list of (frame_idx, cy). Ball x is constant."""
    events = []
    for frame_idx, cy in samples:
        ev = detector.update(
            frame_idx=frame_idx,
            ball_cx=500.0,
            ball_cy=cy,
            ball_tracked=True,
            ball_coasted=False,
            frame_h=1080,
            tracks=TRACKS,
            H=None,
        )
        if ev is not None:
            events.append(ev)
    return events


def test_velocity_is_normalized_by_frame_gap():
    """A 3-frame gap must not look like a 3x speed spike."""
    det = _detector()
    # Ball rises 30 px per real frame, but samples arrive every 3rd frame.
    samples = [(i * 3, 900.0 - 30.0 * i) for i in range(8)]

    # With normalization, per-frame speed is 30 px; the spike gate stays open.
    _feed(det, samples)
    assert det.reject_counts.get("implausible_speed", 0) == 0


def test_streak_survives_a_single_missed_detection():
    det = _detector()
    # Frame 5 is missing from the samples (detection dropped).
    samples = [(i, 900.0 - 25.0 * i) for i in range(10) if i != 5]
    _feed(det, samples)
    assert det.reject_counts.get("ball_not_tracked", 0) == 0
    assert det._consec_up > 0 or det.events, "streak should not be wiped by one gap"


def test_long_gap_drops_the_streak():
    det = _detector()
    _feed(det, [(0, 900.0), (1, 880.0)])
    _feed(det, [(20, 600.0)])  # 19-frame gap
    assert det.reject_counts.get("gap_too_long", 0) == 1
    assert det._consec_up == 0


def test_dribble_sized_rise_does_not_trigger():
    det = _detector()
    # Ball rises only 60 px: below MIN_RISE_PX, so no shot event.
    events = _feed(det, [(i, 900.0 - 10.0 * i) for i in range(8)])
    assert events == []
    assert det.reject_counts.get("insufficient_rise", 0) >= 3


def test_diagnostics_report_rejections():
    det = _detector()
    _feed(det, [(i, 900.0 - 10.0 * i) for i in range(8)])
    diag = det.diagnostics()
    assert diag["events"] == 0
    assert isinstance(diag["rejections"], dict)
    assert diag["rejections"]
