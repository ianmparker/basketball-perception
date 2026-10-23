"""Possession must not be inferred from ball positions that were never observed.

`update()` accepted a `ball_visible` argument documented as "whether ball was
*detected* this frame (vs purely coasted)" and then never read it. On the 60 fps
clip only 52.6% of frames had a real ball detection, so roughly half of all
possession decisions picked the "nearest player" to a coasted or interpolated
coordinate. That is the reported "possession is a bit off".
"""
from __future__ import annotations

import pytest

from analytics.possession import PossessionTracker


PLAYERS_COURT = [
    (1, (10.0, 10.0), 1),
    (2, (30.0, 20.0), 2),
]


def _tracker(**kw):
    return PossessionTracker(**kw)


def test_no_carrier_is_acquired_from_an_unobserved_ball():
    t = _tracker(min_hold_frames=2)
    # Ball sits on player 1's feet, but every frame is flagged not-visible.
    for _ in range(20):
        carrier, team = t.update(
            ball_court=(10.0, 10.0),
            ball_pixel=(100.0, 100.0),
            players_court=PLAYERS_COURT,
            players_pixel=[(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)],
            ball_visible=False,
        )
        assert carrier is None
        assert team is None


def test_carrier_is_acquired_from_an_observed_ball():
    t = _tracker(min_hold_frames=2)
    carrier = None
    for _ in range(5):
        carrier, team = t.update(
            ball_court=(10.0, 10.0),
            ball_pixel=(100.0, 100.0),
            players_court=PLAYERS_COURT,
            players_pixel=[(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)],
            ball_visible=True,
        )
    assert carrier == 1
    assert team == 1


def test_an_existing_carrier_is_held_through_unobserved_frames():
    t = _tracker(min_hold_frames=2)
    for _ in range(5):
        t.update((10.0, 10.0), (100.0, 100.0), PLAYERS_COURT,
                 [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], True)
    assert t.state.carrier_track_id == 1

    # Ball goes dark; the claim must not evaporate instantly.
    for _ in range(3):
        carrier, _team = t.update((10.0, 10.0), (100.0, 100.0), PLAYERS_COURT,
                                  [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], False)
        assert carrier == 1


def test_no_switch_happens_on_an_unobserved_frame():
    """A ghost position must not hand the ball to a different player."""
    t = _tracker(min_hold_frames=2, min_switch_frames=2)
    for _ in range(5):
        t.update((10.0, 10.0), (100.0, 100.0), PLAYERS_COURT,
                 [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], True)
    assert t.state.carrier_track_id == 1

    # Ball coordinate jumps onto player 2, but we did not actually see it.
    for _ in range(10):
        carrier, _ = t.update((30.0, 20.0), (300.0, 200.0), PLAYERS_COURT,
                              [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], False)
        assert carrier == 1


def test_unobserved_frames_clear_partial_candidate_streaks():
    """A streak built on guesses must not complete once the ball reappears."""
    t = _tracker(min_hold_frames=4)
    for _ in range(3):
        t.update((10.0, 10.0), (100.0, 100.0), PLAYERS_COURT,
                 [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], True)
    assert t.state.carrier_track_id is None       # streak of 3, needs 4

    t.update((10.0, 10.0), (100.0, 100.0), PLAYERS_COURT,
             [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], False)
    assert t.state.candidate_streak == 0

    # One more observed frame must not be enough, because the streak was reset.
    carrier, _ = t.update((10.0, 10.0), (100.0, 100.0), PLAYERS_COURT,
                          [(1, (100.0, 100.0), 1), (2, (300.0, 200.0), 2)], True)
    assert carrier is None
