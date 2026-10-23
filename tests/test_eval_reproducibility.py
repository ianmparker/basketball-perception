"""Reproducibility of the eval driver.

Two defects made the committed scorecard impossible to rebuild: the pipeline was
invoked with defaults rather than the flags the labels were scored against, and
ingested clips (which live under data/raw/<source>/<date>/ with .ogv/.webm
extensions) could not be located by stem at all.
"""
from __future__ import annotations

import json
from pathlib import Path

from eval.score import EVAL_PIPELINE_ARGS, _find_video


def _label(path: Path, video_path: str) -> Path:
    path.write_text(json.dumps({"video_path": video_path, "shots": []}), encoding="utf-8")
    return path


def test_eval_flags_are_explicit():
    """The benchmark was produced with explicit flags; they must be passed."""
    assert "--ball-conf" in EVAL_PIPELINE_ARGS
    assert "0.25" in EVAL_PIPELINE_ARGS
    assert "--shot-no-court-required" in EVAL_PIPELINE_ARGS


def test_eval_flags_include_offline_events():
    """Without this the shot events vanish: online detection runs before the ball
    track is smoothed, so it fires nothing on the gym clips."""
    assert "--offline-events" in EVAL_PIPELINE_ARGS


def test_video_is_found_from_the_label(tmp_path):
    video = tmp_path / "data" / "raw" / "src" / "2026-10-20" / "clip_s001.ogv"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    label = _label(tmp_path / "clip_s001.json", "data/raw/src/2026-10-20/clip_s001.ogv")

    assert _find_video("clip_s001", label, repo_root=tmp_path) == video


def test_video_falls_back_to_raw_search_by_extension(tmp_path):
    video = tmp_path / "data" / "raw" / "src" / "2026-10-20" / "clip.webm"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")
    label = _label(tmp_path / "clip.json", "data/raw/does-not-exist.webm")

    assert _find_video("clip", label, repo_root=tmp_path) == video


def test_video_found_in_sample_videos(tmp_path):
    video = tmp_path / "data" / "sample" / "videos" / "broadcast_74.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"x")

    assert _find_video("broadcast_74", None, repo_root=tmp_path) == video


def test_missing_video_returns_none(tmp_path):
    (tmp_path / "data").mkdir()
    assert _find_video("nope", None, repo_root=tmp_path) is None
