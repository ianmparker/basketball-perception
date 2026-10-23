"""Smoke test for CI pipeline."""
import sys
import os

# Windows consoles default to cp1252 and cannot print the check marks below.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on the path so local modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_imports():
    """Test that all required packages can be imported."""
    try:
        import numpy
        import pandas
        import torch
        import torchvision
        import cv2
        import sklearn
        print("✓ All core imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_operations():
    """Test basic operations."""
    import numpy as np
    import torch

    # Test numpy
    arr = np.array([1, 2, 3])
    assert arr.sum() == 6, "NumPy operation failed"

    # Test torch
    tensor = torch.tensor([1, 2, 3])
    assert tensor.sum().item() == 6, "PyTorch operation failed"

    print("✓ Basic operations successful")
    return True


def test_homography_math():
    """Test pixel-to-court coordinate transformation (core pipeline math)."""
    import numpy as np
    from scripts.map_ball_to_court import pixel_to_court

    # Identity-like homography: scale pixels to feet (94ft court = 1280px wide)
    scale = 94.0 / 1280.0
    H = np.array([
        [scale, 0,     0],
        [0,     scale, 0],
        [0,     0,     1],
    ], dtype=float)

    court_pt = pixel_to_court([640, 360], H)
    expected_x = 640 * scale
    expected_y = 360 * scale

    assert abs(court_pt[0] - expected_x) < 1e-6, f"Homography x mismatch: {court_pt[0]} != {expected_x}"
    assert abs(court_pt[1] - expected_y) < 1e-6, f"Homography y mismatch: {court_pt[1]} != {expected_y}"

    print("✓ Homography pixel→court math correct")
    return True


def test_court_drawer_init():
    """Test that CourtDrawer initialises and has correct court dimensions."""
    from visualization.draw_court import CourtDrawer

    drawer = CourtDrawer(pixels_per_foot=5)
    assert drawer.COURT_LENGTH == 94.0, "Court length should be 94 ft"
    assert drawer.COURT_WIDTH == 50.0, "Court width should be 50 ft"

    print("✓ CourtDrawer initialised with correct NBA dimensions")
    return True


def test_learned_homography_output_shape():
    """Test that LearnedHomography outputs a valid (B, 3, 3) homography matrix."""
    import torch
    from homography.learned.model import LearnedHomography

    model = LearnedHomography(pretrained=False)
    model.eval()

    dummy_input = torch.zeros(2, 3, 224, 224)  # batch of 2 frames
    with torch.no_grad():
        H, kp_coords, kp_conf, heatmaps = model(dummy_input)

    assert H.shape == (2, 3, 3), f"Expected shape (2,3,3), got {H.shape}"
    assert not torch.isnan(H).any(), "Homography output contains NaN"
    assert kp_coords.shape == (2, 18, 2), f"Expected kp_coords (2,18,2), got {kp_coords.shape}"
    assert kp_conf.shape == (2, 18), f"Expected kp_conf (2,18), got {kp_conf.shape}"

    print("✓ LearnedHomography outputs valid (B, 3, 3) matrices")
    return True


def test_eval_event_scoring():
    """Phase 15: event P/R/F1 scoring at frame tolerance."""
    from eval.metrics.events import score_events

    # Exact match
    s = score_events([10, 50, 90], [10, 50, 90], tolerance_f=5)
    assert s.tp == 3 and s.fp == 0 and s.fn == 0
    assert abs(s.precision - 1.0) < 1e-9 and abs(s.recall - 1.0) < 1e-9

    # One miss, one extra
    s = score_events([10, 50, 90], [10, 55, 200], tolerance_f=5)
    assert s.tp == 2, f"tp={s.tp}"
    assert s.fp == 1 and s.fn == 1

    # Tolerance boundary: exactly == tolerance should match
    s = score_events([100], [115], tolerance_f=15)
    assert s.tp == 1 and s.fp == 0 and s.fn == 0

    # Tolerance + 1 does not match
    s = score_events([100], [116], tolerance_f=15)
    assert s.tp == 0 and s.fp == 1 and s.fn == 1

    # Greedy 1-1: two GTs within tol of one pred → only one TP
    s = score_events([10, 12], [11], tolerance_f=5)
    assert s.tp == 1 and s.fp == 0 and s.fn == 1

    # Empty GT, empty pred → zeros, f1=0 (no division by zero)
    s = score_events([], [], tolerance_f=5)
    assert s.tp == 0 and s.precision == 0.0 and s.f1 == 0.0

    print("✓ Event scoring correct on 6 cases")
    return True


def test_speed_distance_analytics():
    """Phase 12d: per-player speed and cumulative distance."""
    from analytics.speed_distance import compute_speed_and_distance
    from inference.pipeline import FrameResult, PlayerTrack

    # Track #1 moves +1 ft/frame in x at 30 fps → 30 ft/s.
    # Track #2 stays still. Track #3 has a frame with no position (gap).
    def mkp(tid, xy):
        return PlayerTrack(track_id=tid, box_xyxy=(0,0,1,1), conf=0.9, cls_id=0,
                           court_pos=xy)
    results = []
    for i in range(10):
        players = [mkp(1, (float(i), 0.0)), mkp(2, (50.0, 25.0))]
        if i != 4:  # gap at frame 4 for track 3
            players.append(mkp(3, (10.0 + i*0.5, 5.0)))  # 15 ft/s
        results.append(FrameResult(frame_idx=i, H=None, court_kps=None, court_kpc=None,
                                   players=players))

    n = compute_speed_and_distance(results, fps=30.0, window=5, max_speed_ft_s=35.0)
    assert n > 0

    # Track 1: after warm-up, speed ≈ 30 ft/s, total distance ≈ 9 ft (9 steps × 1 ft)
    p1_last = [p for p in results[-1].players if p.track_id == 1][0]
    assert abs(p1_last.speed_ft_per_s - 30.0) < 0.5, f"got {p1_last.speed_ft_per_s}"
    assert abs(p1_last.distance_traveled_ft - 9.0) < 0.01

    # Track 2: stationary → 0
    p2_last = [p for p in results[-1].players if p.track_id == 2][0]
    assert p2_last.speed_ft_per_s == 0.0
    assert p2_last.distance_traveled_ft == 0.0

    # Track 3: moving with one gap; total dist still non-zero & speed ≈ 15 ft/s
    p3_last = [p for p in results[-1].players if p.track_id == 3][0]
    assert p3_last.distance_traveled_ft > 0
    assert p3_last.speed_ft_per_s > 0

    # Clamp: inject a huge jump and ensure it gets capped, not accumulated
    results2 = [
        FrameResult(frame_idx=0, H=None, court_kps=None, court_kpc=None,
                    players=[mkp(99, (0.0, 0.0))]),
        FrameResult(frame_idx=1, H=None, court_kps=None, court_kpc=None,
                    players=[mkp(99, (500.0, 0.0))]),  # 15000 ft/s — absurd
    ]
    compute_speed_and_distance(results2, fps=30.0, window=5, max_speed_ft_s=35.0)
    p99 = results2[1].players[0]
    # Smoothed speed averages the clamp with prior zeros; what matters is:
    # (a) it never exceeds the clamp, and (b) the jittery step is NOT added to distance.
    assert p99.speed_ft_per_s <= 35.0, f"exceeded clamp: {p99.speed_ft_per_s}"
    assert p99.distance_traveled_ft == 0.0, "jitter step should not accumulate"

    print("✓ Speed & distance: rate, total, gap handling, clamping all correct")
    return True


def test_analytics_overlays():
    """Phase 13: cumulative team ball control + pass/int series."""
    from visualization.analytics_overlays import (
        compute_team_ball_control_series, compute_pass_int_series,
    )
    from inference.pipeline import FrameResult

    # 6 frames of possession: T1,T1,T2,None,T2,T1
    possessions = [1, 1, 2, None, 2, 1]
    results = [
        FrameResult(frame_idx=i, H=None, court_kps=None, court_kpc=None,
                    players=[], possession_team_id=t)
        for i, t in enumerate(possessions)
    ]
    tbc = compute_team_ball_control_series(results)
    assert tbc[-1] == (3, 2, 1), f"final TBC counts wrong: {tbc[-1]}"
    assert tbc[0]  == (1, 0, 0)
    assert tbc[3]  == (2, 1, 1)  # after the None frame

    # Simulate a PassInterceptionResult with 2 passes + 1 intercept
    class _E:
        def __init__(self, f, t): self.frame_idx, self.team_id = f, t
    class _Pir:
        passes        = [_E(1, 1), _E(4, 2)]
        interceptions = [_E(3, 2)]
    pi = compute_pass_int_series(6, _Pir())
    assert pi[0][1]["passes"] == 0
    assert pi[1][1]["passes"] == 1
    assert pi[3][2]["intercepts"] == 1
    assert pi[5][2]["passes"] == 1 and pi[5][2]["intercepts"] == 1
    assert pi[5][1]["passes"] == 1

    # None PassInterceptionResult → zeros
    pi0 = compute_pass_int_series(3, None)
    assert pi0[0] == {1: {"passes": 0, "intercepts": 0}, 2: {"passes": 0, "intercepts": 0}}

    print("✓ Analytics overlay series: TBC cumulative + pass/int cumulative correct")
    return True


def test_eval_possession_scoring():
    """Phase 15: team-level + jersey-level possession accuracy."""
    from eval.metrics.possession import (
        score_possession_team, score_possession_jersey,
    )

    # GT: frames 0-4 → team 1 (jersey 7), frames 5-9 → team 2 (jersey 23)
    gt = [
        {"start_f": 0, "end_f": 4, "team": 1, "player_jersey": 7},
        {"start_f": 5, "end_f": 9, "team": 2, "player_jersey": 23},
    ]
    # Predictions: frames 0-3 correct as T1, frame 4 wrong (T2), 5-7 correct,
    # 8 missing (None), 9 wrong (T1).
    pred_team = [1, 1, 1, 1, 2, 2, 2, 2, None, 1]

    s = score_possession_team(gt, pred_team)
    assert s.n_frames_scored == 10, s.n_frames_scored
    assert s.correct == 7, f"correct={s.correct}"
    assert s.wrong   == 2, f"wrong={s.wrong}"
    assert s.missing == 1, f"missing={s.missing}"
    assert abs(s.accuracy - 0.7) < 1e-9
    assert s.per_team[1]["n"] == 5 and s.per_team[1]["correct"] == 4
    assert s.per_team[2]["n"] == 5 and s.per_team[2]["correct"] == 3

    # Jersey-level: track 12 → jersey 7, tracks 3,5 → jersey 23
    jersey_map = {"7": [12], "23": [3, 5]}
    pred_track = [12, 12, 12, 12, 99, 3, 5, 3, None, 12]  # frame 4 unknown track, 9 wrong jersey
    sj = score_possession_jersey(gt, pred_track, jersey_map)
    assert sj.n_frames_scored == 10
    assert sj.correct == 7
    assert sj.wrong   == 2  # frame 4 (unknown→wrong) and frame 9 (jersey 7 != 23)
    assert sj.missing == 1

    # Empty jersey_map → everything wrong (no track resolves)
    sj2 = score_possession_jersey(gt, pred_track, {})
    assert sj2.correct == 0 and sj2.missing == 1 and sj2.wrong == 9

    print("✓ Possession scoring: team + jersey-level accuracy correct")
    return True


def test_eval_clipresult_load():
    """Phase 15a: pipeline output loads back via ClipResult.from_json."""
    from inference.schema import ClipResult, SCHEMA_VERSION
    from pathlib import Path

    p = Path("eval/results/broadcast_74.clip.json")
    if not p.exists():
        # CI may not have run eval/score.py; just verify the schema class loads
        print("✓ ClipResult class importable (no pipeline output to roundtrip)")
        return True
    clip = ClipResult.from_json(p)
    # Cached artifacts may predate the current schema; loading them must still
    # work (new fields are additive), but the major version must match.
    assert clip.schema_version.split(".")[0] == SCHEMA_VERSION.split(".")[0]
    assert clip.metadata.frame_count == len(clip.frames)
    print(f"✓ ClipResult loaded ({len(clip.frames)} frames, schema v{clip.schema_version})")
    return True


def main():
    """Run all smoke tests."""
    print("Running smoke tests...")

    tests = [
        test_imports,
        test_basic_operations,
        test_homography_math,
        test_court_drawer_init,
        test_learned_homography_output_shape,
        test_eval_event_scoring,
        test_speed_distance_analytics,
        test_analytics_overlays,
        test_eval_possession_scoring,
        test_eval_clipresult_load,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            results.append(False)

    if all(results):
        print("\n✓ All smoke tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some smoke tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
