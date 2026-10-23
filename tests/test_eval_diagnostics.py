import pytest

from eval.diagnostics import (
    bootstrap_mean_ci,
    resolvable_f1_delta,
    shuffle_sanity_check,
    summarize_by_group,
)


class _Report:
    def __init__(self, f1, game_id, footage_type="nba_broadcast"):
        self._f1 = f1
        self.game_id = game_id
        self.footage_type = footage_type

    def average_event_f1(self):
        return self._f1


def test_noise_floor_grows_as_events_shrink():
    small = resolvable_f1_delta(120, 0.8)
    large = resolvable_f1_delta(600, 0.8)
    assert small > large > 0
    # A 2-point difference is not measurable on 120 events.
    assert small > 0.02


def test_noise_floor_undefined_without_events():
    assert resolvable_f1_delta(0) is None


def test_noise_floor_does_not_claim_precision_from_an_empty_score():
    assert resolvable_f1_delta(1, 0.0) > 1.0


def test_shuffle_check_is_skipped_for_tiny_label_sets():
    result = shuffle_sanity_check([10, 20], [10, 20], tolerance_f=2)
    assert result.status == "skipped"


def test_shuffle_check_passes_when_signal_is_real():
    gt = [10, 50, 90, 130, 170, 210]
    pred = [10, 51, 90, 129, 171, 210]
    result = shuffle_sanity_check(pred, gt, tolerance_f=2, seed=0)
    assert result.status == "ok"
    assert result.shuffled_f1 < result.observed_f1


def test_bootstrap_ci_needs_two_clips():
    assert bootstrap_mean_ci([0.8]) == (None, None)
    lo, hi = bootstrap_mean_ci([0.8, 0.6])
    assert lo <= 0.7 <= hi


def test_summarize_by_group_uses_clips_not_frames():
    reports = {
        "a": _Report(0.8, "game_1"),
        "b": _Report(0.6, "game_1"),
        "c": _Report(0.9, "game_2"),
    }
    groups = summarize_by_group(reports, "game_id")
    assert [g.group for g in groups] == ["game_1", "game_2"]
    game1 = groups[0]
    assert game1.n_clips == 2
    assert game1.mean_f1 == pytest.approx(0.7)
    assert groups[1].ci_low is None
