"""The court-view gate: report nothing when the court cannot be located.

Broadcast footage is full of crowd shots, bench close-ups and replay graphics.
Without a gate the detector reports spectators as players and assigns them a
team, which made a demo render look broken: frames of two men in the crowd got a
skeleton and a "T1 #62 71%" team label.
"""
from __future__ import annotations

import numpy as np

from inference.pipeline import FrameResult, PipelineConfig, is_court_view


def test_gate_is_off_by_default():
    """The labelled clips are close-up gym footage with no homography at all, so
    gating unconditionally would suppress every event and zero the scorecard."""
    assert PipelineConfig().court_view_gate is False


def test_gate_off_reports_every_frame():
    assert is_court_view(False, None) is True
    assert is_court_view(False, np.eye(3)) is True


def test_gate_rejects_frames_without_a_homography():
    assert is_court_view(True, None) is False


def test_gate_keeps_frames_with_a_homography():
    assert is_court_view(True, np.eye(3)) is True


def test_frame_result_defaults_to_court_view():
    fr = FrameResult(frame_idx=0, H=None, court_kps=None, court_kpc=None)
    assert fr.court_view is True
    assert fr.to_dict()["court_view"] is True


def test_frame_result_serializes_the_gate_decision():
    fr = FrameResult(frame_idx=3, H=None, court_kps=None, court_kpc=None, court_view=False)
    assert fr.to_dict()["court_view"] is False
