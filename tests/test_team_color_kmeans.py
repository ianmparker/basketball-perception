import numpy as np

from analytics.team_color_kmeans import KMeansTeamAssigner


def test_hue_is_circular():
    """Hue 178 and hue 2 are neighbours, not opposites."""
    a = KMeansTeamAssigner._encode_hsv(np.array([178.0, 200.0, 150.0]))
    b = KMeansTeamAssigner._encode_hsv(np.array([2.0, 200.0, 150.0]))
    assert np.linalg.norm(a - b) < 0.2

    raw_a = np.array([178.0, 200.0, 150.0]) / 255.0
    raw_b = np.array([2.0, 200.0, 150.0]) / 255.0
    assert np.linalg.norm(raw_a - raw_b) > 0.6


def test_gray_ignores_hue():
    """When saturation is zero, hue carries no information."""
    a = KMeansTeamAssigner._encode_hsv(np.array([0.0, 0.0, 120.0]))
    b = KMeansTeamAssigner._encode_hsv(np.array([90.0, 0.0, 120.0]))
    assert np.allclose(a, b)


def test_hue_weight_scales_with_saturation():
    """A saturated crop should react more to hue than a washed-out one."""
    low_sat = KMeansTeamAssigner._encode_hsv(np.array([10.0, 10.0, 150.0]))
    low_sat_shift = KMeansTeamAssigner._encode_hsv(np.array([40.0, 10.0, 150.0]))
    high_sat = KMeansTeamAssigner._encode_hsv(np.array([10.0, 250.0, 150.0]))
    high_sat_shift = KMeansTeamAssigner._encode_hsv(np.array([40.0, 250.0, 150.0]))

    assert np.linalg.norm(high_sat_shift - high_sat) > 10 * np.linalg.norm(
        low_sat_shift - low_sat
    )


def test_margin_is_normalized_per_cluster():
    """A sparse cluster must not be starved by a threshold tuned on a dense one."""
    assigner = KMeansTeamAssigner()
    assigner._centroids = np.array(
        [[1.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    assigner._cluster_scale = np.array([1.0, 0.1], dtype=np.float32)
    assigner._light_centroid_idx = 0

    # 0.2 away from the sparse centroid, far from the dense one.
    team, margin = assigner._assign_color(np.array([0.0, 0.0, 0.0]))
    assert team == 1
    assert margin > 0.9


def test_finalize_reports_failed_warmup():
    assigner = KMeansTeamAssigner(warmup_frames=40)
    assigner._warmup_samples = [
        np.array([10.0, 200.0, 150.0], dtype=np.float32),
        np.array([170.0, 200.0, 150.0], dtype=np.float32),
    ]
    assert assigner.finalize() is False
    assert assigner.team_assignment_available is False
    assert assigner.diagnostics()["samples_collected"] == 2


def test_finalize_fits_when_enough_samples_exist():
    assigner = KMeansTeamAssigner(warmup_frames=40)
    assigner._warmup_samples = [
        np.array([10.0, 200.0, 150.0], dtype=np.float32),
        np.array([12.0, 190.0, 150.0], dtype=np.float32),
        np.array([170.0, 200.0, 150.0], dtype=np.float32),
        np.array([172.0, 190.0, 150.0], dtype=np.float32),
    ]
    assert assigner.finalize() is True
    assert assigner.team_assignment_available is True
