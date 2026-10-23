"""Fixed (manual) court homography.

The learned court-keypoint model only fires on NBA-style broadcast. For a static
camera — the case the business actually targets — clicking 4-8 court points once
gives an exact calibration that needs no model at all. That path only works if
the pipeline can be handed the resulting matrix, which is what these tests cover.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from inference.pipeline import PipelineConfig, load_homography

H_SAMPLE = [[1.4, 0.2, -120.0], [0.05, 1.1, -40.0], [0.0004, 0.0001, 1.0]]


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_loads_manual_homography_tool_output(tmp_path):
    """Format written by homography/manual_homography.py."""
    path = _write(
        tmp_path / "cal.json",
        {
            "video_path": "clip.mp4",
            "homography_matrix": H_SAMPLE,
            "homography_matrix_inverse": np.linalg.inv(H_SAMPLE).tolist(),
            "num_points": 8,
        },
    )
    H = load_homography(path)
    assert H.shape == (3, 3)
    np.testing.assert_allclose(H, np.asarray(H_SAMPLE))


def test_loads_a_bare_matrix(tmp_path):
    H = load_homography(_write(tmp_path / "cal.json", H_SAMPLE))
    np.testing.assert_allclose(H, np.asarray(H_SAMPLE))


def test_rejects_missing_key(tmp_path):
    with pytest.raises(ValueError, match="no 'homography_matrix'"):
        load_homography(_write(tmp_path / "cal.json", {"foo": 1}))


def test_rejects_wrong_shape(tmp_path):
    with pytest.raises(ValueError, match="3x3"):
        load_homography(_write(tmp_path / "cal.json", {"homography_matrix": [[1, 0], [0, 1]]}))


def test_rejects_singular_matrix(tmp_path):
    singular = [[1, 2, 3], [2, 4, 6], [1, 1, 1]]
    with pytest.raises(ValueError, match="singular"):
        load_homography(_write(tmp_path / "cal.json", {"homography_matrix": singular}))


def test_config_has_no_fixed_homography_by_default():
    assert PipelineConfig().fixed_homography is None


def test_config_accepts_a_matrix():
    cfg = PipelineConfig(fixed_homography=np.asarray(H_SAMPLE))
    assert cfg.fixed_homography is not None
