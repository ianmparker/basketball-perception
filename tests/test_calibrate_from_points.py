"""Non-interactive court calibration from pixel points.

homography/manual_homography.py needs GUI clicks, so it cannot run headless or from
an agent. calibrate_from_points.py does the same solve from coordinates, which makes
calibration a two-minute scriptable step and keeps it out of a mouse-only path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.pipeline import COURT_KPS_FT, load_homography
from scripts.calibrate_from_points import parse_matches, solve

# A plausible pixel->court matrix (court feet), used as ground truth.
H_REF = np.array(
    [[1.4, 0.2, -120.0], [0.05, 1.1, -40.0], [0.0004, 0.0001, 1.0]], dtype=np.float64
)


def _pixels_for(indices, H=H_REF):
    """Project court feet back to pixels via the inverse mapping."""
    H_inv = np.linalg.inv(H)
    out = []
    for i in indices:
        fx, fy = COURT_KPS_FT[i]
        v = H_inv @ np.array([fx, fy, 1.0])
        out.append((v[0] / v[2], v[1] / v[2]))
    return np.asarray(out, dtype=np.float32)


def test_parse_matches_maps_indices_to_court_feet():
    pixels, court = parse_matches(["0:100,200", "5:300,400"])

    assert pixels.shape == (2, 2)
    np.testing.assert_allclose(pixels[0], [100, 200])
    np.testing.assert_allclose(court[0], COURT_KPS_FT[0])
    np.testing.assert_allclose(court[1], COURT_KPS_FT[5])


def test_parse_matches_rejects_bad_syntax():
    with pytest.raises(SystemExit):
        parse_matches(["nonsense"])
    with pytest.raises(SystemExit):
        parse_matches(["5:1,2,3"])


def test_parse_matches_rejects_out_of_range_index():
    with pytest.raises(SystemExit):
        parse_matches([f"{len(COURT_KPS_FT)}:1,2"])


def test_solve_recovers_a_known_homography():
    idx = [0, 3, 5, 8, 10, 12, 15]
    pixels = _pixels_for(idx)
    court = np.asarray([COURT_KPS_FT[i] for i in idx])
    H, rms, inliers = solve(pixels, court)

    assert rms < 1e-3
    assert inliers == len(idx)
    # Homographies are scale-invariant, so compare the mapping rather than the matrix.
    mapped = (H @ np.hstack([pixels, np.ones((len(pixels), 1))]).T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    np.testing.assert_allclose(mapped, court, atol=1e-3)


def test_solve_requires_four_points():
    with pytest.raises(SystemExit, match="at least 4"):
        solve(np.zeros((3, 2), np.float32), np.zeros((3, 2), np.float32))


def test_solve_output_is_loadable_by_the_pipeline(tmp_path):
    """The whole point: a calibration produced here must feed run_video."""
    idx = [0, 3, 5, 10]
    H, _, _ = solve(_pixels_for(idx), np.asarray([COURT_KPS_FT[i] for i in idx]))

    path = tmp_path / "cal.json"
    import json

    path.write_text(json.dumps({"homography_matrix": H.tolist()}), encoding="utf-8")
    loaded = load_homography(path)
    np.testing.assert_allclose(loaded, H)
