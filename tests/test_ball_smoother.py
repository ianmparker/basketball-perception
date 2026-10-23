import numpy as np

from analytics.ball_smoother import _fill_gaps, smooth_ball_track
from inference.pipeline import BallState, FrameResult


def test_short_gap_is_linear():
    values = np.array([0.0, 1.0, np.nan, np.nan, 4.0, 5.0])
    filled = _fill_gaps(values, linear_gap_max=8)
    assert np.allclose(filled[2:4], [2.0, 3.0])


def test_long_gap_uses_a_curve():
    """A ballistic arc must not be flattened into its chord."""
    t = np.arange(31, dtype=float)
    true_y = 0.05 * t ** 2
    values = true_y.copy()
    values[6:25] = np.nan  # 19-frame gap

    filled = _fill_gaps(values, linear_gap_max=8)
    gap = slice(6, 25)
    curve_error = np.max(np.abs(filled[gap] - true_y[gap]))

    chord = np.linspace(true_y[5], true_y[25], 21)[1:-1]
    linear_error = np.max(np.abs(chord - true_y[gap]))

    assert curve_error < 0.1
    assert curve_error < linear_error / 20


def test_leading_and_trailing_gaps_use_nearest_value():
    values = np.array([np.nan, np.nan, 3.0, np.nan, 7.0, np.nan])
    filled = _fill_gaps(values)
    assert np.allclose(filled, [3.0, 3.0, 3.0, 5.0, 7.0, 7.0])


def test_all_nan_is_unchanged():
    values = np.array([np.nan, np.nan])
    filled = _fill_gaps(values)
    assert np.isnan(filled).all()


# ── interpolated flag ───────────────────────────────────────────────────────
# The smoother used to stamp interpolated=True on every frame it wrote back, which
# made the flag constant. The renderer then labelled all 600 frames of a real clip
# "ball ~", and the JSON could not distinguish a detection from an invented position.

def _frames(balls):
    frames = []
    for i, b in enumerate(balls):
        f = FrameResult(frame_idx=i, H=None, court_kps=None, court_kpc=None, ball=b)
        frames.append(f)
    return frames


def _det(cx, cy):
    return BallState(tracked=True, coasted=False, cx=cx, cy=cy, coast_frames=0,
                     court_pos=None, det_box=[cx - 5, cy - 5, cx + 5, cy + 5],
                     det_conf=0.8)


def test_detection_backed_frames_are_not_marked_interpolated():
    frames = _frames([_det(100, 100), _det(102, 101), _det(104, 102)])

    smooth_ball_track(frames, max_jump_px=200, smooth_window=0)

    assert [f.ball.interpolated for f in frames] == [False, False, False]


def test_gap_filled_frames_are_marked_interpolated():
    frames = _frames([_det(100, 100), None, None, _det(160, 100)])

    smooth_ball_track(frames, max_jump_px=200, smooth_window=0)

    # The two middle frames have no detection, so they are the only invented ones.
    assert [f.ball.interpolated for f in frames] == [False, True, True, False]


def test_motion_gated_out_detection_counts_as_interpolated():
    # A 500 px jump is rejected by the gate, so that frame's position is not backed.
    frames = _frames([_det(100, 100), _det(600, 100), _det(110, 100)])

    smooth_ball_track(frames, max_jump_px=50, smooth_window=0)

    assert frames[1].ball.interpolated is True
    assert frames[0].ball.interpolated is False


def test_coasted_frames_are_not_treated_as_measurements():
    coasted = BallState(tracked=True, coasted=True, cx=105, cy=100,
                        coast_frames=2, court_pos=None, det_box=None, det_conf=0.0)
    frames = _frames([_det(100, 100), coasted, _det(110, 100)])

    smooth_ball_track(frames, max_jump_px=200, smooth_window=0)

    assert [f.ball.interpolated for f in frames] == [False, True, False]
