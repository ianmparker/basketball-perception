import pytest

from eval.metrics.tracking import (
    iou,
    score_player_count,
    score_tracking,
    track_fragmentation,
)


def _box(x, y, w=10, h=20):
    return (x, y, x + w, y + h)


def test_iou_basics():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_perfect_tracking_scores_one():
    gt = {f: [(1, _box(f, 0)), (2, _box(f, 100))] for f in range(10)}
    pred = {f: [(1, _box(f, 0)), (2, _box(f, 100))] for f in range(10)}
    score = score_tracking(gt, pred)

    assert score.mota == pytest.approx(1.0)
    assert score.idf1 == pytest.approx(1.0)
    assert score.id_switches == 0
    assert score.tp == 20


def test_missed_detections_reduce_mota():
    gt = {f: [(1, _box(f, 0))] for f in range(10)}
    pred = {f: [(1, _box(f, 0))] for f in range(8)}
    score = score_tracking(gt, pred)

    assert score.fn == 2
    assert score.mota == pytest.approx(0.8)


def test_id_switch_is_penalized():
    gt = {f: [(1, _box(f, 0))] for f in range(10)}
    pred = {f: [(1 if f < 5 else 99, _box(f, 0))] for f in range(10)}
    score = score_tracking(gt, pred)

    assert score.id_switches == 1
    assert score.mota == pytest.approx(0.9)
    # One GT id split across two predicted ids: global ID matching keeps only
    # one of them, so IDTP is 5 of 10 detections.
    assert score.idf1 == pytest.approx(0.5)


def test_disjoint_ids_lower_idf1():
    gt = {f: [(1, _box(f, 0))] for f in range(10)}
    pred = {f: [(1, _box(f, 0))] for f in range(5)}
    score = score_tracking(gt, pred)
    assert score.idf1 == pytest.approx(2 * 5 / (2 * 5 + 0 + 5))
    assert score.idp == pytest.approx(1.0)
    assert score.idr == pytest.approx(0.5)


def test_player_count_checks():
    score = score_player_count(
        [(0, 10), (10, 10), (20, 9)],
        {0: 10, 10: 9, 20: 9},
    )
    assert score.checks == 3
    assert score.exact == 2
    assert score.mean_abs_error == pytest.approx(1 / 3)
    assert score.worst == 1


def test_player_count_skips_missing_frames():
    score = score_player_count([(0, 10), (99, 10)], {0: 10})
    assert score.checks == 1
    assert score.accuracy == 1.0


def test_player_count_records_direction_of_error():
    """An absolute error hides whether the tracker over- or under-counted."""
    score = score_player_count([(20, 10), (50, 10)], {20: 19, 50: 7})

    assert score.observations == [(20, 10, 19), (50, 10, 7)]
    assert score.mean_abs_error == pytest.approx(6.0)
    serialized = score.to_dict()
    assert serialized["observations"][0] == {
        "frame_idx": 20,
        "expected_players": 10,
        "observed_persons": 19,
    }


def test_track_fragmentation_separates_long_tracks_from_fragments():
    # Track 1 persists across all 20 frames; track 2 appears once then vanishes.
    per_frame = [[1, 2]] + [[1] for _ in range(19)]

    stats = track_fragmentation(per_frame)
    assert stats["unique_tracks"] == 2
    assert stats["long_tracks"] == 1
    assert stats["fragments"] == 1


def test_track_fragmentation_of_a_clean_track():
    stats = track_fragmentation([[7] for _ in range(10)])
    assert stats == {"unique_tracks": 1, "long_tracks": 1, "fragments": 0}
