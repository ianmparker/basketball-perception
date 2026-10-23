"""Default ball detector profile is the target-domain replay-v3 model."""

from inference.pipeline import PipelineConfig


def test_replay_v3_is_the_default_ball_profile():
    cfg = PipelineConfig()

    assert cfg.ball_model_path == "models/ball_detector_ebard_v3.pt"
    assert cfg.ball_conf == 0.25
    assert cfg.ball_imgsz == 1280
    assert cfg.ball_class_id == 0


def test_legacy_detector_can_be_selected_with_config_overrides():
    cfg = PipelineConfig(
        ball_model_path="models/ball_detector_model.pt",
        ball_conf=0.15,
        ball_imgsz=640,
    )

    assert cfg.ball_model_path == "models/ball_detector_model.pt"
    assert cfg.ball_conf == 0.15
    assert cfg.ball_imgsz == 640
