from inference.schema import (
    ClipMetadata,
    ClipResult,
    FrameResultModel,
    HomographyQuality,
)


def test_frame_accepts_health_fields():
    frame = FrameResultModel.model_validate(
        {
            "frame_idx": 0,
            "homography_quality": {
                "source": "current",
                "inlier_ratio": 0.75,
                "rmse_ft": 1.2,
            },
            "possession_distance_mode": "court",
        }
    )
    assert isinstance(frame.homography_quality, HomographyQuality)
    assert frame.homography_quality.inlier_ratio == 0.75
    assert frame.possession_distance_mode == "court"


def test_clip_can_report_unavailable_team_assignment():
    clip = ClipResult(
        metadata=ClipMetadata(
            video="clip.mp4", fps=30.0, width=1920, height=1080, frame_count=0
        ),
        team_assignment_available=False,
    )
    assert clip.team_assignment_available is False
