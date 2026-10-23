"""
Transform pixel trajectories to real-world court coordinates using homography.

This module takes ByteTrack output (pixel coordinates) and transforms them to
NBA court coordinates (feet) using the calibrated homography matrix. It then
calculates real-world metrics like speed, distance, and acceleration.

Mathematical Foundation (see docs/MATHEMATICAL_FOUNDATIONS.md):
- Homography transformation: p_court = H × p_pixel
- Distance: Euclidean distance in feet
- Speed: distance / time (converted to mph)
- Acceleration: Δspeed / Δtime (ft/s²)
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse


class TrajectoryTransformer:
    """Transform pixel trajectories to court coordinates and calculate metrics."""
    
    def __init__(self, homography_path: str, fps: float = 29.0):
        """
        Initialize trajectory transformer.
        
        Args:
            homography_path: Path to homography_matrix.json
            fps: Video frame rate (for speed calculations)
        """
        self.fps = fps
        self.frame_time = 1.0 / fps  # seconds per frame
        
        # Load homography matrix
        with open(homography_path, 'r') as f:
            data = json.load(f)
        
        self.homography = np.array(data['homography_matrix'])
        self.video_path = data['video_path']
        self.frame_width = data['frame_width']
        self.frame_height = data['frame_height']
        self.court_length = data['court_dimensions']['length_feet']
        self.court_width = data['court_dimensions']['width_feet']
        
        print(f"✅ Loaded homography matrix from: {homography_path}")
        print(f"   Video: {self.video_path}")
        print(f"   Resolution: {self.frame_width}x{self.frame_height}")
        print(f"   Court: {self.court_length}ft × {self.court_width}ft")
        print(f"   FPS: {self.fps}")
    
    def transform_point(self, pixel_point: np.ndarray) -> np.ndarray:
        """
        Transform a single pixel point to court coordinates.
        
        Args:
            pixel_point: [x, y] in pixels
            
        Returns:
            [x, y] in feet on court
        """
        # Convert to homogeneous coordinates
        point_h = np.array([pixel_point[0], pixel_point[1], 1.0])
        
        # Apply homography: p' = H × p
        court_h = self.homography @ point_h
        
        # Convert back from homogeneous (divide by w)
        court_point = court_h[:2] / court_h[2]
        
        return court_point
    
    def transform_trajectory(self, pixel_trajectory: List[Tuple[float, float]]) -> np.ndarray:
        """
        Transform entire trajectory from pixels to court coordinates.
        
        Args:
            pixel_trajectory: List of (x, y) pixel coordinates
            
        Returns:
            np.ndarray of shape (N, 2) with court coordinates in feet
        """
        court_trajectory = []
        for pixel_point in pixel_trajectory:
            court_point = self.transform_point(np.array(pixel_point))
            court_trajectory.append(court_point)
        
        return np.array(court_trajectory)
    
    def calculate_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two court points.
        
        Args:
            point1: [x, y] in feet
            point2: [x, y] in feet
            
        Returns:
            Distance in feet
        """
        return np.linalg.norm(point2 - point1)
    
    def calculate_speed(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Calculate instantaneous speed at each point in trajectory.
        
        Speed = distance / time
        Converted to miles per hour (mph)
        
        Args:
            trajectory: np.ndarray of shape (N, 2) in feet
            
        Returns:
            np.ndarray of shape (N,) with speeds in mph
        """
        if len(trajectory) < 2:
            return np.array([0.0])
        
        # Calculate distances between consecutive points
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)  # feet
        
        # Convert to speed: feet/frame → feet/second → mph
        speeds_fps = distances / self.frame_time  # feet per second
        speeds_mph = speeds_fps * 0.681818  # convert ft/s to mph (1 ft/s = 0.681818 mph)
        
        # Prepend 0 for first point (no previous point to calculate from)
        speeds = np.concatenate([[0.0], speeds_mph])
        
        return speeds
    
    def calculate_acceleration(self, speeds: np.ndarray) -> np.ndarray:
        """
        Calculate acceleration from speed profile.
        
        Acceleration = Δspeed / Δtime
        
        Args:
            speeds: np.ndarray of shape (N,) in mph
            
        Returns:
            np.ndarray of shape (N,) with acceleration in ft/s²
        """
        if len(speeds) < 2:
            return np.array([0.0])
        
        # Convert mph back to ft/s for acceleration calculation
        speeds_fps = speeds / 0.681818
        
        # Calculate change in speed
        delta_speeds = np.diff(speeds_fps)  # ft/s
        
        # Calculate acceleration: (ft/s) / time = ft/s²
        accelerations = delta_speeds / self.frame_time
        
        # Prepend 0 for first point
        accelerations = np.concatenate([[0.0], accelerations])
        
        return accelerations
    
    def calculate_total_distance(self, trajectory: np.ndarray) -> float:
        """
        Calculate total distance traveled along trajectory.
        
        Args:
            trajectory: np.ndarray of shape (N, 2) in feet
            
        Returns:
            Total distance in feet
        """
        if len(trajectory) < 2:
            return 0.0
        
        distances = np.linalg.norm(np.diff(trajectory, axis=0), axis=1)
        return np.sum(distances)
    
    def smooth_trajectory(self, trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply moving average smoothing to trajectory.
        
        Reduces noise from detection/tracking jitter.
        
        Args:
            trajectory: np.ndarray of shape (N, 2)
            window_size: Size of moving average window (must be odd)
            
        Returns:
            Smoothed trajectory of same shape
        """
        if len(trajectory) < window_size:
            return trajectory
        
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        # Apply moving average to x and y separately
        smoothed = np.copy(trajectory)
        half_window = window_size // 2
        
        for i in range(half_window, len(trajectory) - half_window):
            smoothed[i] = np.mean(trajectory[i - half_window:i + half_window + 1], axis=0)
        
        return smoothed
    
    def process_track(self, pixel_trajectory: List[Tuple[float, float]], 
                      track_id: int, 
                      smooth: bool = True) -> Dict:
        """
        Process a single track: transform, smooth, calculate metrics.
        
        Args:
            pixel_trajectory: List of (x, y) pixel coordinates
            track_id: Unique track ID
            smooth: Whether to apply smoothing
            
        Returns:
            Dictionary with:
                - track_id: int
                - pixel_trajectory: List[(x,y)]
                - court_trajectory: np.ndarray (N, 2)
                - speeds: np.ndarray (N,) in mph
                - accelerations: np.ndarray (N,) in ft/s²
                - total_distance: float in feet
                - avg_speed: float in mph
                - max_speed: float in mph
                - duration: float in seconds
        """
        # Transform to court coordinates
        court_traj = self.transform_trajectory(pixel_trajectory)
        
        # Apply smoothing if requested
        if smooth and len(court_traj) > 5:
            court_traj = self.smooth_trajectory(court_traj, window_size=5)
        
        # Calculate metrics
        speeds = self.calculate_speed(court_traj)
        accelerations = self.calculate_acceleration(speeds)
        total_distance = self.calculate_total_distance(court_traj)
        duration = len(pixel_trajectory) * self.frame_time
        
        # Summary statistics
        avg_speed = np.mean(speeds[1:]) if len(speeds) > 1 else 0.0
        max_speed = np.max(speeds)
        
        return {
            'track_id': track_id,
            'num_frames': len(pixel_trajectory),
            'pixel_trajectory': pixel_trajectory,
            'court_trajectory': court_traj.tolist(),  # Convert to list for JSON
            'speeds_mph': speeds.tolist(),
            'accelerations_fps2': accelerations.tolist(),
            'total_distance_feet': float(total_distance),
            'avg_speed_mph': float(avg_speed),
            'max_speed_mph': float(max_speed),
            'duration_seconds': float(duration)
        }


def extract_tracks_from_video(video_path: str, model_path: str, 
                               conf_threshold: float = 0.5) -> Dict[int, List[Tuple[float, float]]]:
    """
    Extract tracked trajectories from video using YOLOv8 + ByteTrack.
    
    This function re-runs detection and tracking to extract clean trajectory data.
    
    Args:
        video_path: Path to input video
        model_path: Path to YOLOv8 model
        conf_threshold: Detection confidence threshold
        
    Returns:
        Dictionary mapping track_id → list of (center_x, center_y) pixel coordinates
    """
    from ultralytics import YOLO
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from tracking.byte_track import ByteTrack
    
    print(f"\n{'='*70}")
    print("EXTRACTING TRAJECTORIES FROM VIDEO")
    print(f"{'='*70}\n")
    
    # Load model
    model = YOLO(model_path)
    print(f"✅ Loaded YOLOv8 model: {model_path}")
    
    # Initialize tracker
    tracker = ByteTrack()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"✅ Video: {video_path}")
    print(f"   FPS: {fps}, Total frames: {total_frames}")
    
    # Storage for trajectories
    tracks_dict = {}  # track_id → [(x, y), ...]
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Extract detections for tracker
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            
            # Track all objects (will filter by class later if needed)
            detections.append([x1, y1, x2, y2, conf, cls])
        
        # Update tracker
        if detections:
            detections_np = np.array(detections)
            tracks = tracker.update(detections_np)
            
            # Store center points
            for track in tracks:
                track_id = track.track_id
                bbox = track.tlbr  # [x1, y1, x2, y2]
                
                # Calculate center point
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                if track_id not in tracks_dict:
                    tracks_dict[track_id] = []
                
                tracks_dict[track_id].append((float(center_x), float(center_y)))
        
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"   Processed {frame_idx}/{total_frames} frames...")
    
    cap.release()
    
    print(f"\n✅ Extracted {len(tracks_dict)} unique tracks")
    
    # Filter out very short tracks (noise)
    min_track_length = 10
    filtered_tracks = {tid: traj for tid, traj in tracks_dict.items() 
                      if len(traj) >= min_track_length}
    
    print(f"✅ Filtered to {len(filtered_tracks)} tracks (≥{min_track_length} frames)")
    
    return filtered_tracks


def main():
    """Main function to transform trajectories and calculate metrics."""
    parser = argparse.ArgumentParser(description='Transform pixel trajectories to court coordinates')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--model', required=True, help='Path to YOLOv8 model')
    parser.add_argument('--homography', required=True, help='Path to homography_matrix.json')
    parser.add_argument('--output', default='trajectories.json', help='Output JSON file')
    parser.add_argument('--conf', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--fps', type=float, default=29.0, help='Video FPS')
    parser.add_argument('--smooth', action='store_true', help='Apply trajectory smoothing')
    
    args = parser.parse_args()
    
    # Extract trajectories from video
    tracks = extract_tracks_from_video(args.video, args.model, args.conf)
    
    # Initialize transformer
    transformer = TrajectoryTransformer(args.homography, fps=args.fps)
    
    # Process each track
    print(f"\n{'='*70}")
    print("TRANSFORMING TRAJECTORIES")
    print(f"{'='*70}\n")
    
    processed_tracks = []
    for track_id, pixel_traj in tracks.items():
        result = transformer.process_track(pixel_traj, track_id, smooth=args.smooth)
        processed_tracks.append(result)
        
        # Print summary
        print(f"Track {track_id:3d}: "
              f"{result['num_frames']:3d} frames, "
              f"{result['total_distance_feet']:6.1f} ft, "
              f"avg {result['avg_speed_mph']:4.1f} mph, "
              f"max {result['max_speed_mph']:4.1f} mph")
    
    # Save to JSON
    output_data = {
        'video_path': args.video,
        'model_path': args.model,
        'homography_path': args.homography,
        'fps': args.fps,
        'conf_threshold': args.conf,
        'smoothing_enabled': args.smooth,
        'num_tracks': len(processed_tracks),
        'tracks': processed_tracks
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✅ Saved trajectories to: {output_path}")
    
    # Print overall statistics
    if processed_tracks:
        total_distance = sum(t['total_distance_feet'] for t in processed_tracks)
        avg_speeds = [t['avg_speed_mph'] for t in processed_tracks if t['avg_speed_mph'] > 0]
        max_speeds = [t['max_speed_mph'] for t in processed_tracks]
        
        print(f"\n{'='*70}")
        print("OVERALL STATISTICS")
        print(f"{'='*70}")
        print(f"Total tracks: {len(processed_tracks)}")
        print(f"Total distance traveled: {total_distance:.1f} feet")
        if avg_speeds:
            print(f"Average speed (all tracks): {np.mean(avg_speeds):.1f} mph")
        if max_speeds:
            print(f"Max speed observed: {np.max(max_speeds):.1f} mph")
        print(f"{'='*70}\n")
    else:
        print(f"\n⚠️  No tracks found. Check detection confidence threshold or video content.\n")


if __name__ == '__main__':
    main()
