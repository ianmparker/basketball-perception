"""
Shot detection and classification using ball trajectory analysis.

Detects:
- Shot attempts (ball release with upward trajectory)
- Shot outcomes (made/missed based on ball-hoop proximity)
- Shot types (layup, mid-range, 3-pointer based on court position)
- Shot distance (using homography for real-world measurements)
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import cv2


@dataclass
class ShotEvent:
    """Shot event data structure."""
    event_id: int
    frame_start: int
    frame_release: int
    frame_end: int
    shooter_track_id: Optional[int] = None
    release_position_pixels: Optional[Tuple[float, float]] = None
    release_position_court: Optional[Tuple[float, float]] = None
    shot_type: Optional[str] = None  # 'layup', 'mid_range', '3_pointer'
    shot_distance_feet: Optional[float] = None
    outcome: Optional[str] = None  # 'made', 'missed', 'unknown'
    confidence: float = 0.0


class ShotDetector:
    """Detect shot events using ball trajectory analysis."""
    
    def __init__(self, 
                 min_upward_velocity: float = 5.0,
                 min_shot_frames: int = 10,
                 max_shot_frames: int = 120,
                 hoop_proximity_threshold: float = 50.0,
                 interpolate_missing: bool = True,
                 max_interpolation_gap: int = 15):
        """
        Initialize shot detector.
        
        Args:
            min_upward_velocity: Minimum upward velocity (pixels/frame) to detect release
            min_shot_frames: Minimum frames for valid shot
            max_shot_frames: Maximum frames to track shot
            hoop_proximity_threshold: Distance (pixels) for made shot detection
            interpolate_missing: Interpolate missing ball detections
            max_interpolation_gap: Maximum frames to interpolate across
        """
        self.min_upward_velocity = min_upward_velocity
        self.min_shot_frames = min_shot_frames
        self.max_shot_frames = max_shot_frames
        self.hoop_proximity_threshold = hoop_proximity_threshold
        self.interpolate_missing = interpolate_missing
        self.max_interpolation_gap = max_interpolation_gap
        
        print(f"✅ Shot detector initialized")
        print(f"   Min upward velocity: {min_upward_velocity} px/frame")
        print(f"   Shot frame range: {min_shot_frames}-{max_shot_frames}")
        print(f"   Hoop proximity: {hoop_proximity_threshold} px")
        print(f"   Interpolation: {'Enabled (max gap: ' + str(max_interpolation_gap) + ' frames)' if interpolate_missing else 'Disabled'}")
    
    def calculate_ball_velocity(self, ball_positions: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Calculate ball velocity from consecutive positions.
        
        Args:
            ball_positions: List of (x, y) positions
            
        Returns:
            List of (vx, vy) velocities
        """
        velocities = []
        for i in range(1, len(ball_positions)):
            prev_x, prev_y = ball_positions[i-1]
            curr_x, curr_y = ball_positions[i]
            vx = curr_x - prev_x
            vy = curr_y - prev_y
            velocities.append((vx, vy))
        return velocities
    
    def detect_release_point(self, ball_trajectory: List[Dict]) -> Optional[int]:
        """
        Detect shot release frame (ball starts moving upward significantly).
        
        Args:
            ball_trajectory: List of ball detections with frame_id and position
            
        Returns:
            Frame index of release, or None
        """
        if len(ball_trajectory) < 3:
            return None
        
        positions = [(ball['x'], ball['y']) for ball in ball_trajectory]
        velocities = self.calculate_ball_velocity(positions)
        
        # Look for significant upward velocity (negative vy = upward)
        for i, (vx, vy) in enumerate(velocities):
            if vy < -self.min_upward_velocity:  # Moving up significantly
                return i + 1  # Return frame index (offset by 1 from velocity index)
        
        return None
    
    def detect_apex(self, ball_trajectory: List[Dict], release_idx: int) -> Optional[int]:
        """
        Detect apex (highest point) of shot trajectory.
        
        Args:
            ball_trajectory: List of ball detections
            release_idx: Release frame index
            
        Returns:
            Frame index of apex, or None
        """
        if release_idx >= len(ball_trajectory) - 1:
            return None
        
        # Find minimum y-value after release (y decreases upward)
        min_y = float('inf')
        apex_idx = None
        
        for i in range(release_idx, min(len(ball_trajectory), release_idx + self.max_shot_frames)):
            y = ball_trajectory[i]['y']
            if y < min_y:
                min_y = y
                apex_idx = i
        
        return apex_idx
    
    def classify_shot_outcome(self, ball_trajectory: List[Dict], 
                             hoop_positions: List[Dict],
                             apex_idx: int) -> str:
        """
        Classify shot as made or missed based on ball-hoop proximity.
        
        Args:
            ball_trajectory: Ball detections
            hoop_positions: Hoop detections with frame_id and position
            apex_idx: Apex frame index
            
        Returns:
            'made', 'missed', or 'unknown'
        """
        if apex_idx >= len(ball_trajectory):
            return 'unknown'
        
        # Create hoop position lookup
        hoop_lookup = {hoop['frame_id']: hoop for hoop in hoop_positions}
        
        # Check ball-hoop distance around apex and descent
        for i in range(apex_idx, min(len(ball_trajectory), apex_idx + 30)):
            ball = ball_trajectory[i]
            frame_id = ball['frame_id']
            
            if frame_id in hoop_lookup:
                hoop = hoop_lookup[frame_id]
                ball_x, ball_y = ball['x'], ball['y']
                hoop_x, hoop_y = hoop['x'], hoop['y']
                
                distance = np.sqrt((ball_x - hoop_x)**2 + (ball_y - hoop_y)**2)
                
                if distance < self.hoop_proximity_threshold:
                    return 'made'
        
        return 'missed'
    
    def classify_shot_type(self, release_position_court: Tuple[float, float], 
                          hoop_position_court: Tuple[float, float] = (5.25, 25.0)) -> Tuple[str, float]:
        """
        Classify shot type and calculate distance.
        
        Args:
            release_position_court: (x, y) in feet on court
            hoop_position_court: Hoop position (default: 5.25 ft from baseline, 25 ft from sideline)
            
        Returns:
            (shot_type, distance_feet)
        """
        x, y = release_position_court
        hoop_x, hoop_y = hoop_position_court
        
        # Calculate distance to hoop
        distance = np.sqrt((x - hoop_x)**2 + (y - hoop_y)**2)
        
        # Classify by distance
        if distance < 8:
            shot_type = 'layup'
        elif distance < 23.75:
            shot_type = 'mid_range'
        else:
            shot_type = '3_pointer'
        
        return shot_type, float(distance)
    
    def extract_ball_trajectory(self, detections: List[Dict]) -> List[Dict]:
        """
        Extract ball detections from frame detections with interpolation for missing frames.
        
        Args:
            detections: List of frame detections with all objects
            
        Returns:
            List of ball detections with frame_id and center position (including interpolated)
        """
        ball_detections = []
        
        for frame_data in detections:
            frame_id = frame_data['frame_id']
            
            # Look for ball detections (class_id=1)
            if 'detections' not in frame_data:
                ball_detections.append({'frame_id': frame_id, 'x': None, 'y': None, 'confidence': 0.0})
                continue
            
            found_ball = False
            for det in frame_data['detections']:
                if det.get('class_id') == 1:  # Ball
                    bbox = det['bbox']
                    # Calculate center
                    center_x = (bbox['x1'] + bbox['x2']) / 2
                    center_y = (bbox['y1'] + bbox['y2']) / 2
                    
                    ball_detections.append({
                        'frame_id': frame_id,
                        'x': center_x,
                        'y': center_y,
                        'confidence': det.get('confidence', 0.0),
                        'interpolated': False
                    })
                    found_ball = True
                    break
            
            if not found_ball:
                ball_detections.append({
                    'frame_id': frame_id, 
                    'x': None, 
                    'y': None, 
                    'confidence': 0.0,
                    'interpolated': False
                })
        
        # Interpolate missing detections if enabled
        if self.interpolate_missing:
            ball_detections = self._interpolate_trajectory(ball_detections)
        
        # Filter out None values
        ball_trajectory = [b for b in ball_detections if b['x'] is not None]
        
        return ball_trajectory
    
    def _interpolate_trajectory(self, detections: List[Dict]) -> List[Dict]:
        """
        Interpolate missing ball positions using linear interpolation.
        
        Args:
            detections: Ball detections with some None positions
            
        Returns:
            Detections with interpolated positions
        """
        interpolated = detections.copy()
        n = len(interpolated)
        
        i = 0
        while i < n:
            # Find a detected ball
            if interpolated[i]['x'] is not None:
                # Look for next detected ball
                j = i + 1
                while j < n and interpolated[j]['x'] is None:
                    j += 1
                
                # If found and gap is small enough, interpolate
                if j < n and (j - i - 1) <= self.max_interpolation_gap:
                    gap_size = j - i
                    start_x, start_y = interpolated[i]['x'], interpolated[i]['y']
                    end_x, end_y = interpolated[j]['x'], interpolated[j]['y']
                    
                    # Linear interpolation
                    for k in range(i + 1, j):
                        alpha = (k - i) / gap_size
                        interpolated[k]['x'] = start_x + alpha * (end_x - start_x)
                        interpolated[k]['y'] = start_y + alpha * (end_y - start_y)
                        interpolated[k]['confidence'] = 0.5  # Mark as interpolated
                        interpolated[k]['interpolated'] = True
                
                i = j
            else:
                i += 1
        
        return interpolated
    
    def extract_hoop_positions(self, detections: List[Dict]) -> List[Dict]:
        """
        Extract hoop detections from frame detections.
        
        Args:
            detections: List of frame detections
            
        Returns:
            List of hoop positions with frame_id and center
        """
        hoop_positions = []
        
        for frame_data in detections:
            frame_id = frame_data['frame_id']
            
            if 'detections' not in frame_data:
                continue
            
            for det in frame_data['detections']:
                if det.get('class_id') == 2:  # Hoop
                    bbox = det['bbox']
                    center_x = (bbox['x1'] + bbox['x2']) / 2
                    center_y = (bbox['y1'] + bbox['y2']) / 2
                    
                    hoop_positions.append({
                        'frame_id': frame_id,
                        'x': center_x,
                        'y': center_y,
                        'confidence': det.get('confidence', 0.0)
                    })
                    break
        
        return hoop_positions
    
    def detect_shots(self, detections_json: str, 
                    homography_json: Optional[str] = None) -> List[ShotEvent]:
        """
        Detect all shots in video detections.
        
        Args:
            detections_json: Path to detection results JSON
            homography_json: Optional path to homography matrix
            
        Returns:
            List of detected shot events
        """
        # Load detections
        with open(detections_json, 'r') as f:
            data = json.load(f)
        
        detections = data.get('frames', [])
        
        # Load homography if provided
        homography_matrix = None
        if homography_json and Path(homography_json).exists():
            with open(homography_json, 'r') as f:
                homography_data = json.load(f)
                homography_matrix = np.array(homography_data['homography_matrix'])
        
        print(f"\n{'='*70}")
        print("SHOT DETECTION")
        print(f"{'='*70}\n")
        print(f"Frames: {len(detections)}")
        print(f"Homography: {'✅ Loaded' if homography_matrix is not None else '❌ Not available'}\n")
        
        # Extract ball trajectory
        ball_trajectory = self.extract_ball_trajectory(detections)
        print(f"Ball detections: {len(ball_trajectory)}")
        raw_detections = [b for b in ball_trajectory if not b.get('interpolated', False)]
        interpolated_count = len(ball_trajectory) - len(raw_detections)
        print(f"  Raw detections: {len(raw_detections)}")
        print(f"  Interpolated: {interpolated_count}")
        
        # Extract hoop positions
        hoop_positions = self.extract_hoop_positions(detections)
        print(f"Hoop detections: {len(hoop_positions)}\n")
        
        if len(ball_trajectory) < self.min_shot_frames:
            print("⚠️  Insufficient ball detections for shot analysis")
            return []
        
        # Detect shots
        shots = []
        event_id = 1
        i = 0
        
        while i < len(ball_trajectory):
            # Detect release point
            remaining_trajectory = ball_trajectory[i:]
            release_idx = self.detect_release_point(remaining_trajectory)
            
            if release_idx is None:
                i += 1
                continue
            
            # Adjust to absolute index
            abs_release_idx = i + release_idx
            
            # Detect apex
            apex_idx = self.detect_apex(ball_trajectory, abs_release_idx)
            
            if apex_idx is None:
                i = abs_release_idx + 1
                continue
            
            # Validate shot duration
            shot_duration = apex_idx - abs_release_idx
            if shot_duration < self.min_shot_frames or shot_duration > self.max_shot_frames:
                i = abs_release_idx + 1
                continue
            
            # Classify outcome
            outcome = self.classify_shot_outcome(ball_trajectory, hoop_positions, apex_idx)
            
            # Get release position
            release_ball = ball_trajectory[abs_release_idx]
            release_pos_pixels = (release_ball['x'], release_ball['y'])
            
            # Transform to court coordinates if homography available
            release_pos_court = None
            shot_type = None
            shot_distance = None
            
            if homography_matrix is not None:
                pixel_point = np.array([release_pos_pixels[0], release_pos_pixels[1], 1.0])
                court_point = homography_matrix @ pixel_point
                if court_point[2] != 0:
                    release_pos_court = (
                        float(court_point[0] / court_point[2]),
                        float(court_point[1] / court_point[2])
                    )
                    shot_type, shot_distance = self.classify_shot_type(release_pos_court)
            
            # Create shot event
            shot = ShotEvent(
                event_id=event_id,
                frame_start=ball_trajectory[abs_release_idx]['frame_id'],
                frame_release=ball_trajectory[abs_release_idx]['frame_id'],
                frame_end=ball_trajectory[min(apex_idx + 20, len(ball_trajectory) - 1)]['frame_id'],
                release_position_pixels=release_pos_pixels,
                release_position_court=release_pos_court,
                shot_type=shot_type,
                shot_distance_feet=shot_distance,
                outcome=outcome,
                confidence=release_ball['confidence']
            )
            
            shots.append(shot)
            event_id += 1
            
            # Move past this shot
            i = apex_idx + 20
        
        print(f"{'='*70}")
        print(f"DETECTED {len(shots)} SHOTS")
        print(f"{'='*70}\n")
        
        for shot in shots:
            print(f"Shot {shot.event_id}:")
            print(f"  Frame: {shot.frame_release}")
            print(f"  Type: {shot.shot_type if shot.shot_type else 'unknown'}")
            print(f"  Distance: {shot.shot_distance_feet:.1f} ft" if shot.shot_distance_feet else "  Distance: unknown")
            print(f"  Outcome: {shot.outcome}")
            print(f"  Confidence: {shot.confidence:.2f}\n")
        
        return shots
    
    def export_shots(self, shots: List[ShotEvent], output_json: str):
        """
        Export shot events to JSON.
        
        Args:
            shots: List of shot events
            output_json: Output path
        """
        output_data = {
            'total_shots': len(shots),
            'made': len([s for s in shots if s.outcome == 'made']),
            'missed': len([s for s in shots if s.outcome == 'missed']),
            'unknown': len([s for s in shots if s.outcome == 'unknown']),
            'layups': len([s for s in shots if s.shot_type == 'layup']),
            'mid_range': len([s for s in shots if s.shot_type == 'mid_range']),
            'three_pointers': len([s for s in shots if s.shot_type == '3_pointer']),
            'shots': [asdict(shot) for shot in shots]
        }
        
        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ Saved {len(shots)} shots to: {output_json}")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect shots in basketball video')
    parser.add_argument('--detections', required=True, help='Detection results JSON')
    parser.add_argument('--homography', default=None, help='Homography matrix JSON')
    parser.add_argument('--output', required=True, help='Output shots JSON')
    parser.add_argument('--min-velocity', type=float, default=5.0, help='Min upward velocity')
    parser.add_argument('--hoop-proximity', type=float, default=50.0, help='Hoop proximity threshold')
    
    args = parser.parse_args()
    
    detector = ShotDetector(
        min_upward_velocity=args.min_velocity,
        hoop_proximity_threshold=args.hoop_proximity
    )
    
    shots = detector.detect_shots(args.detections, args.homography)
    detector.export_shots(shots, args.output)


if __name__ == '__main__':
    main()
