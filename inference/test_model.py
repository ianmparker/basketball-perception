"""
Test trained YOLOv8 model on sample images.

Usage:
    python inference/test_model.py --model runs/detect/train/weights/best.pt
    python inference/test_model.py --model runs/detect/train/weights/best.pt --source data/raw/basketball-players/valid
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def test_inference(
    model_path: str,
    source: str = None,
    conf: float = 0.25,
    save: bool = True,
):
    """
    Run inference with trained model.
    
    Args:
        model_path: Path to trained model (.pt file)
        source: Path to image/video/directory, or webcam index
        conf: Confidence threshold for detections
        save: Save annotated results
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"\n{'=' * 60}")
    print(f"YOLOV8 INFERENCE TEST")
    print(f"{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"Source: {source or 'default test images'}")
    print(f"Confidence: {conf}")
    print(f"{'=' * 60}\n")
    
    # Load model
    model = YOLO(model_path)
    
    # Default source: sample validation images
    if source is None:
        source = Path(__file__).parent.parent / "data" / "raw" / "basketball-players" / "valid"
        if not source.exists():
            print(f"⚠ Warning: Default source not found: {source}")
            return
    
    # Run inference
    results = model.predict(
        source=str(source),
        conf=conf,
        save=save,
        project='runs/inference',
        name='test',
        exist_ok=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
    )
    
    # Print summary
    print(f"\n{'=' * 60}")
    print(f"✓ INFERENCE COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Processed {len(results)} images")
    if save:
        print(f"Results saved to: runs/inference/test")
    print(f"{'=' * 60}\n")
    
    # Print per-class detection counts
    class_counts = {}
    total_detections = 0
    for result in results:
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                total_detections += 1
    
    if total_detections > 0:
        print(f"Total Detections: {total_detections}")
        print(f"\nDetections by Class:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections) * 100
            print(f"  {cls_name:20s}: {count:4d} ({percentage:.1f}%)")
    else:
        print("⚠ No detections found. Consider lowering confidence threshold.")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test YOLOv8 model inference")
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    parser.add_argument(
        '--source', '-s',
        type=str,
        default=None,
        help='Path to image/video/directory. Default: validation images'
    )
    parser.add_argument(
        '--conf', '-c',
        type=float,
        default=0.25,
        help='Confidence threshold (0-1)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save annotated results'
    )
    
    args = parser.parse_args()
    
    try:
        test_inference(
            model_path=args.model,
            source=args.source,
            conf=args.conf,
            save=not args.no_save,
        )
    except Exception as e:
        print(f"\n❌ Error during inference: {e}")
        raise


if __name__ == "__main__":
    main()
