"""Download the public backbone weights used by the pipeline.

The task-specific ball, player, and court models were trained on research-only
footage and are not distributed. See ``models/README.md`` for the expected paths.

Usage:
    python scripts/download_models.py
"""
from __future__ import annotations


def main() -> int:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Run: pip install -r requirements.txt")
        return 1

    for name in ("yolov8n.pt", "yolov8x-pose.pt"):
        print(f"[download] {name}")
        YOLO(name)  # Ultralytics resolves and downloads the public checkpoint.

    print(
        "\nPublic backbones are ready. The task-specific weights are not included:\n"
        "  models/ball_detector_ebard_v3.pt\n"
        "  runs/detect/phase6_yolo11m/weights/best.pt\n"
        "  runs/pose/court_kp_v18_resume/weights/best.pt\n"
        "See models/README.md for details.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
