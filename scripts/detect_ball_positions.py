"""
Extract ball positions from video using trained YOLOv8 model.
Saves ball pixel coordinates to JSON for each frame.
"""

import json
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def detect_ball_in_video(video_path, model_path, conf_threshold=0.25, output_json=None):
    """
    Run ball detection on video and extract ball positions.
    
    Args:
        video_path: Path to input video
        model_path: Path to trained YOLO model (.pt file)
        conf_threshold: Confidence threshold for detections (default 0.25)
        output_json: Path to output JSON file (default: video_name_ball_positions.json)
    
    Returns:
        Dictionary with frame numbers as keys and ball positions as values
    """
    # Load model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    print(f"Confidence threshold: {conf_threshold}")
    
    # Run inference on video
    results = model.predict(
        source=str(video_path),
        conf=conf_threshold,
        verbose=False,
        stream=True  # Stream results for memory efficiency
    )
    
    ball_positions = {}
    frames_with_detection = 0
    
    for frame_idx, result in enumerate(results):
        detections = []
        
        # Extract ball detections from this frame
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Calculate center point of bounding box
                center_x = float((x1 + x2) / 2)
                center_y = float((y1 + y2) / 2)
                
                # Calculate box dimensions
                width = float(x2 - x1)
                height = float(y2 - y1)
                
                detections.append({
                    'center': [center_x, center_y],
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'width': width,
                    'height': height,
                    'confidence': confidence
                })
            
            frames_with_detection += 1
        
        # Store detections for this frame
        ball_positions[frame_idx] = detections
        
        # Progress update
        if (frame_idx + 1) % 50 == 0:
            print(f"Processed {frame_idx + 1}/{total_frames} frames...")
    
    cap.release()
    
    # Summary statistics
    detection_rate = (frames_with_detection / total_frames) * 100
    print(f"\nDetection Summary:")
    print(f"  Frames with ball detected: {frames_with_detection}/{total_frames} ({detection_rate:.1f}%)")
    
    # Count frames with multiple detections
    multi_detection_frames = sum(1 for dets in ball_positions.values() if len(dets) > 1)
    if multi_detection_frames > 0:
        print(f"  Frames with multiple detections: {multi_detection_frames}")
    
    # Create output JSON
    if output_json is None:
        video_name = Path(video_path).stem
        output_json = f"{video_name}_ball_positions.json"
    
    output_data = {
        'video_path': str(video_path),
        'model_path': str(model_path),
        'conf_threshold': conf_threshold,
        'total_frames': total_frames,
        'fps': fps,
        'frames_with_detection': frames_with_detection,
        'detection_rate': detection_rate,
        'ball_positions': ball_positions
    }
    
    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nBall positions saved to: {output_json}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(description='Extract ball positions from video using YOLOv8')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='best.pt', help='Path to trained YOLO model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold (default: 0.25)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    
    args = parser.parse_args()
    
    detect_ball_in_video(
        video_path=args.video,
        model_path=args.model,
        conf_threshold=args.conf,
        output_json=args.output
    )


if __name__ == '__main__':
    main()
