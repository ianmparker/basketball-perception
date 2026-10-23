"""
Video tracking script using YOLOv8 detection + ByteTrack for multi-object tracking.
Maintains persistent player IDs across frames with visualization.
"""

import os
import sys
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from tracking.byte_track import ByteTrack


# Color palette for different track IDs (BGR format for OpenCV)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (192, 192, 192), (128, 128, 128)
]


def get_color(track_id):
    """Get consistent color for track ID."""
    return COLORS[track_id % len(COLORS)]


def draw_tracks(frame, tracks, class_names):
    """
    Draw tracked objects on frame with IDs and trails.
    
    Args:
        frame: Input frame (H, W, 3)
        tracks: List of STrack objects
        class_names: Dictionary mapping class IDs to names
    
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        if not track.is_activated:
            continue
        
        # Get bounding box
        x1, y1, x2, y2 = track.tlbr.astype(int)
        track_id = track.track_id
        class_id = track.class_id
        score = track.score
        
        # Get color for this track
        color = get_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label text
        class_name = class_names.get(class_id, f"Class {class_id}")
        label = f"ID:{track_id} {class_name} {score:.2f}"
        
        # Calculate text size and position
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw label background
        cv2.rectangle(annotated, 
                     (x1, y1 - text_h - baseline - 5),
                     (x1 + text_w, y1),
                     color, -1)
        
        # Draw label text
        cv2.putText(annotated, label,
                   (x1, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (255, 255, 255), 2)
        
        # Draw center point
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(annotated, (cx, cy), 4, color, -1)
    
    return annotated


def track_video(model_path, video_path, output_path=None, conf_thresh=0.5, 
                track_thresh=0.5, track_buffer=30, match_thresh=0.8,
                save_video=True, display=False):
    """
    Run detection and tracking on video.
    
    Args:
        model_path: Path to YOLO model weights
        video_path: Path to input video
        output_path: Path to save output video (optional)
        conf_thresh: Detection confidence threshold
        track_thresh: Tracking confidence threshold
        track_buffer: Number of frames to keep lost tracks
        match_thresh: IoU threshold for matching
        save_video: Whether to save output video
        display: Whether to display video in real-time
    
    Returns:
        Dictionary with tracking statistics
    """
    print("=" * 60)
    print("BYTETRACK VIDEO TRACKING")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Video: {video_path}")
    print(f"Confidence Threshold: {conf_thresh}")
    print(f"Track Threshold: {track_thresh}")
    print(f"Track Buffer: {track_buffer} frames")
    print(f"Match Threshold: {match_thresh}")
    print("=" * 60)
    print()
    
    # Load YOLO model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    class_names = model.names
    print(f"✓ Model loaded with {len(class_names)} classes")
    print(f"  Classes: {list(class_names.values())}")
    print()
    
    # Open video
    print("Opening video...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Video opened: {frame_width}x{frame_height} @ {fps} FPS")
    print(f"  Total frames: {total_frames}")
    print()
    
    # Setup output video writer
    if save_video:
        if output_path is None:
            output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_tracked.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output will be saved to: {output_path}")
        print()
    
    # Initialize ByteTrack tracker
    tracker = ByteTrack(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh
    )
    
    # Tracking statistics
    stats = {
        'total_frames': 0,
        'total_detections': 0,
        'total_tracks': 0,
        'active_tracks_per_frame': [],
        'class_counts': {},
        'processing_times': []
    }
    
    frame_idx = 0
    start_time = time.time()
    
    print("Processing video...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            # Run detection
            results = model(frame, conf=conf_thresh, verbose=False)[0]
            
            # Extract detections
            detections = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes = results.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                scores = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy()
                
                # Format: [x1, y1, x2, y2, score, class_id]
                for box, score, cls in zip(boxes, scores, classes):
                    detections.append([*box, score, cls])
                
                detections = np.array(detections)
            else:
                detections = np.array([])
            
            # Update tracker
            if len(detections) > 0:
                tracks = tracker.update(detections)
            else:
                tracks = tracker.update([])
            
            # Draw tracks
            annotated = draw_tracks(frame, tracks, class_names)
            
            # Add frame info
            info_text = f"Frame: {frame_idx + 1}/{total_frames} | Active Tracks: {len(tracks)} | Detections: {len(detections)}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update statistics
            stats['total_frames'] += 1
            stats['total_detections'] += len(detections)
            stats['active_tracks_per_frame'].append(len(tracks))
            stats['processing_times'].append(time.time() - frame_start)
            
            for det in detections:
                cls_id = int(det[5])
                cls_name = class_names.get(cls_id, f"Class {cls_id}")
                stats['class_counts'][cls_name] = stats['class_counts'].get(cls_name, 0) + 1
            
            # Track total unique IDs
            if tracks:
                max_id = max(track.track_id for track in tracks)
                stats['total_tracks'] = max(stats['total_tracks'], max_id)
            
            # Save frame
            if save_video:
                out.write(annotated)
            
            # Display frame
            if display:
                cv2.imshow('Tracking', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress
            frame_idx += 1
            if frame_idx % 50 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_idx / elapsed
                eta = (total_frames - frame_idx) / fps_current if fps_current > 0 else 0
                print(f"  Processed {frame_idx}/{total_frames} frames "
                      f"({frame_idx/total_frames*100:.1f}%) | "
                      f"FPS: {fps_current:.1f} | "
                      f"ETA: {eta:.1f}s")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if save_video:
            out.release()
        if display:
            cv2.destroyAllWindows()
    
    # Calculate final statistics
    elapsed_time = time.time() - start_time
    avg_fps = stats['total_frames'] / elapsed_time if elapsed_time > 0 else 0
    avg_tracks = np.mean(stats['active_tracks_per_frame']) if stats['active_tracks_per_frame'] else 0
    avg_proc_time = np.mean(stats['processing_times']) * 1000 if stats['processing_times'] else 0
    
    # Print results
    print()
    print("=" * 60)
    print("✓ TRACKING COMPLETE!")
    print("=" * 60)
    print(f"Processed: {stats['total_frames']} frames")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average FPS: {avg_fps:.2f}")
    print(f"Average processing time: {avg_proc_time:.1f}ms per frame")
    print()
    print(f"Total Detections: {stats['total_detections']}")
    print(f"Total Unique Tracks: {stats['total_tracks']}")
    print(f"Average Active Tracks: {avg_tracks:.1f} per frame")
    print()
    print("Detections by Class:")
    for cls_name, count in sorted(stats['class_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls_name:20s}: {count:5d} ({count/stats['total_detections']*100:.1f}%)")
    
    if save_video:
        print()
        print(f"Output saved to: {output_path}")
    
    print("=" * 60)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Track objects in video using YOLO + ByteTrack")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to YOLO model weights")
    parser.add_argument("--video", type=str, required=True,
                       help="Path to input video")
    parser.add_argument("--output", type=str, default=None,
                       help="Path to output video (default: <video>_tracked.mp4)")
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--track-thresh", type=float, default=0.5,
                       help="Tracking confidence threshold (default: 0.5)")
    parser.add_argument("--track-buffer", type=int, default=30,
                       help="Frames to keep lost tracks (default: 30)")
    parser.add_argument("--match-thresh", type=float, default=0.8,
                       help="IoU threshold for matching (default: 0.8)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output video")
    parser.add_argument("--display", action="store_true",
                       help="Display video in real-time")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")
    
    # Run tracking
    stats = track_video(
        model_path=args.model,
        video_path=args.video,
        output_path=args.output,
        conf_thresh=args.conf,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        save_video=not args.no_save,
        display=args.display
    )
    
    return stats


if __name__ == "__main__":
    main()
