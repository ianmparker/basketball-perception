import cv2
import numpy as np
import pytest

from scripts.click_calibrate import LANDMARKS, solve


def test_solve_recovers_a_known_homography():
    court = [LANDMARKS[i][2] for i in (0, 4, 5, 6, 7, 9)]
    # court (ft) -> pixel
    H_cp = np.array([[12.0, 0.0, 200.0], [0.0, -12.0, 900.0], [1e-5, 0.0, 1.0]])
    pts = np.asarray(court, dtype=np.float64).reshape(-1, 1, 2)
    pixels = cv2.perspectiveTransform(pts, H_cp).reshape(-1, 2)
    H, Hinv, rms, inliers, mask = solve(pixels.tolist(), court)
    assert inliers == len(court)
    assert rms < 0.01
    assert H.shape == (3, 3)
    assert Hinv.shape == (3, 3)
    assert mask.sum() == len(court)


def test_solve_requires_four_points():
    with pytest.raises(ValueError):
        solve([(0, 0), (1, 1), (2, 2)], [(0, 0), (1, 1), (2, 2)])
