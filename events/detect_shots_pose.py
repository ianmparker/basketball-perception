"""
Shot detection using pose estimation data.

More reliable than ball detection because:
- Player poses detected in 98.7% of frames (vs 6% for ball)
- Shooting motion has clear biomechanical signatures (wrist elevation, arm extension)
- Ball often occluded by player body in broadcast footage
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ShotEvent:
    """Shot event detected from pose data."""
    event_id: int
    shooter_track_id: int
    frame_start: int
    frame_peak: int
    frame_end: int
    release_height_pixels: float
    elbow_angle_at_release: float
    shot_type: Optional[str] = None
    shot_distance_feet: Optional[float] = None
    release_position_court: Optional[Tuple[float, float]] = None
    form_score: Optional[float] = None
    confidence: float = 0.0


class PoseBasedShotDetector:
    """Detect shots using player pose data (wrist elevation and arm extension)."""
    
    def __init__(self,
                 max_wrist_y: float = 600.0,
                 min_elbow_extension: float = 120.0,
                 min_shot_duration: int = 3,
                 max_shot_duration: int = 20,
                 require_upward_motion: bool = True,
                 require_follow_through: bool = True):
        """
        Initialize pose-based shot detector.
        
        Args:
            max_wrist_y: Maximum Y coordinate for wrist (lower = higher in image)
            min_elbow_extension: Minimum elbow angle for release (degrees)
            min_shot_duration: Minimum frames for valid shooting motion
            max_shot_duration: Maximum frames to track shooting motion
            require_upward_motion: Require wrist to move upward before peak
            require_follow_through: Require arm to stay extended after peak
        """
        self.max_wrist_y = max_wrist_y
        self.min_elbow_extension = min_elbow_extension
        self.min_shot_duration = min_shot_duration
        self.max_shot_duration = max_shot_duration
        self.require_upward_motion = require_upward_motion
        self.require_follow_through = require_follow_through
        
        # COCO keypoint indices
        self.KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        print(f"✅ Pose-based shot detector initialized")
        print(f"   Max wrist Y: {max_wrist_y} px (lower = higher)")
        print(f"   Min elbow extension: {min_elbow_extension}°")
        print(f"   Shot duration: {min_shot_duration}-{max_shot_duration} frames")
        print(f"   Filters: upward_motion={require_upward_motion}, follow_through={require_follow_through}")
    
    def calculate_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """
        Calculate angle at p2 formed by p1-p2-p3.
        
        Args:
            p1, p2, p3: Points as [x, y] arrays
            
        Returns:
            Angle in degrees
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        
        return angle
    
    def detect_shooting_motion(self, track_poses: List[Dict]) -> List[Dict]:
        """
        Detect shooting motion sequences within a track.
        Returns list of separate shooting motions (can be multiple per track).
        
        Args:
            track_poses: List of pose data for a single track
            
        Returns:
            List of shooting motion sequences
        """
        if len(track_poses) < self.min_shot_duration:
            return []
        
        # Find all frames with elevated extended arms
        shooting_frames = []
        
        for pose_data in track_poses:
            if 'pose' not in pose_data or pose_data.get('pose') is None:
                continue
            
            if not pose_data.get('has_pose', False):
                continue
            
            keypoints = pose_data['pose']['keypoints']
            
            # Check both arms
            for side in ['left', 'right']:
                shoulder_idx = 5 if side == 'left' else 6
                elbow_idx = 7 if side == 'left' else 8
                wrist_idx = 9 if side == 'left' else 10
                
                # Check if keypoints are visible
                if (keypoints[shoulder_idx]['confidence'] < 0.5 or 
                    keypoints[elbow_idx]['confidence'] < 0.5 or 
                    keypoints[wrist_idx]['confidence'] < 0.5):
                    continue
                
                shoulder = np.array([keypoints[shoulder_idx]['x'], keypoints[shoulder_idx]['y']])
                elbow = np.array([keypoints[elbow_idx]['x'], keypoints[elbow_idx]['y']])
                wrist = np.array([keypoints[wrist_idx]['x'], keypoints[wrist_idx]['y']])
                
                # Check high wrist position (low Y value = high in image)
                wrist_y = wrist[1]
                
                if wrist_y > self.max_wrist_y:
                    continue
                
                # Calculate elbow angle
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                
                # Extended arm indicates shooting motion
                if elbow_angle >= self.min_elbow_extension:
                    shooting_frames.append({
                        'frame_id': pose_data['frame_id'],
                        'arm': side,
                        'wrist_y': wrist_y,
                        'elbow_angle': elbow_angle,
                        'wrist_position': wrist.tolist(),
                        'confidence': min(keypoints[shoulder_idx]['confidence'], 
                                         keypoints[elbow_idx]['confidence'], 
                                         keypoints[wrist_idx]['confidence'])
                    })
                    break  # Only count once per frame
        
        if len(shooting_frames) == 0:
            return []
        
        # Group consecutive shooting frames into separate shot sequences
        shot_sequences = []
        current_sequence = [shooting_frames[0]]
        
        for i in range(1, len(shooting_frames)):
            frame_gap = shooting_frames[i]['frame_id'] - shooting_frames[i-1]['frame_id']
            
            # If gap is too large, start new sequence
            if frame_gap > 5:  # Max 5 frame gap within a shot
                if len(current_sequence) >= self.min_shot_duration:
                    shot_sequences.append(current_sequence)
                current_sequence = [shooting_frames[i]]
            else:
                current_sequence.append(shooting_frames[i])
                
                # If sequence too long, end it
                if len(current_sequence) > self.max_shot_duration:
                    shot_sequences.append(current_sequence[:self.max_shot_duration])
                    current_sequence = []
        
        # Add final sequence
        if len(current_sequence) >= self.min_shot_duration:
            shot_sequences.append(current_sequence)
        
        # Validate each sequence
        validated_sequences = []
        for sequence in shot_sequences:
            validated = self._validate_shot_sequence(sequence)
            if validated:
                validated_sequences.append(validated)
        
        return validated_sequences
    
    def _validate_shot_sequence(self, shooting_frames: List[Dict]) -> Optional[Dict]:
        """Validate a shooting frame sequence."""
        if len(shooting_frames) < self.min_shot_duration:
            return None
        
        # Find peak of shooting motion (highest wrist = lowest Y value)
        peak_idx = min(range(len(shooting_frames)), key=lambda i: shooting_frames[i]['wrist_y'])
        peak_frame = shooting_frames[peak_idx]
        
        # Filter out passes by checking for upward motion before peak
        if self.require_upward_motion and peak_idx > 0:
            # Check if wrist moved upward (Y decreased) in frames before peak
            frames_before_peak = shooting_frames[:peak_idx]
            if len(frames_before_peak) >= 2:
                # Calculate upward velocity
                early_y = frames_before_peak[0]['wrist_y']
                late_y = frames_before_peak[-1]['wrist_y']
                upward_movement = early_y - late_y  # Positive = moved up
                
                if upward_movement < 20:  # Minimum upward movement (pixels)
                    return None  # Likely a pass or static overhead position
        
        # Filter out passes by checking for follow-through after peak
        if self.require_follow_through and peak_idx < len(shooting_frames) - 1:
            # Shots have follow-through (arm stays extended after peak)
            # Passes snap back down quickly
            frames_after_peak = shooting_frames[peak_idx + 1:]
            if len(frames_after_peak) < 2:
                return None  # No follow-through, likely a pass
        
        return {
            'shooting_frames': shooting_frames,
            'peak_frame': peak_frame,
            'peak_idx': peak_idx,
            'duration': len(shooting_frames)
        }
    
    def group_poses_by_track(self, tracked_poses: List[Dict]) -> Dict[int, List[Dict]]:
        """
        Group poses by track ID.
        
        Args:
            tracked_poses: List of poses with track_id
            
        Returns:
            Dict mapping track_id to list of poses
        """
        tracks = {}
        
        for pose in tracked_poses:
            track_id = pose.get('track_id')
            if track_id is None:
                continue
            
            if track_id not in tracks:
                tracks[track_id] = []
            
            tracks[track_id].append(pose)
        
        # Sort each track by frame_id
        for track_id in tracks:
            tracks[track_id].sort(key=lambda x: x['frame_id'])
        
        return tracks
    
    def detect_shots(self, tracked_poses_json: str, 
                    homography_json: Optional[str] = None,
                    min_shot_spacing: int = 30) -> List[ShotEvent]:
        """
        Detect all shots in tracked pose data.
        
        Args:
            tracked_poses_json: Path to tracked poses JSON
            homography_json: Optional homography matrix for court coordinates
            min_shot_spacing: Minimum frames between shots (filters clustered false positives)
            
        Returns:
            List of detected shot events
        """
        # Load tracked poses
        with open(tracked_poses_json, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if 'frames' in data:
            # Extract poses from frame-based structure
            tracked_poses = []
            for frame_data in data['frames']:
                frame_id = frame_data['frame_id']
                for pose in frame_data.get('tracked_poses', frame_data.get('poses', [])):
                    pose['frame_id'] = frame_id
                    tracked_poses.append(pose)
        else:
            tracked_poses = data.get('tracked_poses', data.get('poses', []))
        
        # Load homography if provided
        homography_matrix = None
        if homography_json and Path(homography_json).exists():
            with open(homography_json, 'r') as f:
                homography_data = json.load(f)
                homography_matrix = np.array(homography_data['homography_matrix'])
        
        print(f"\n{'='*70}")
        print("POSE-BASED SHOT DETECTION")
        print(f"{'='*70}\n")
        print(f"Tracked poses: {len(tracked_poses)}")
        print(f"Homography: {'✅ Loaded' if homography_matrix is not None else '❌ Not available'}\n")
        
        # Group poses by track
        tracks = self.group_poses_by_track(tracked_poses)
        print(f"Unique tracks: {len(tracks)}\n")
        
        # Detect shooting motions for each track
        shots = []
        event_id = 1
        
        print("Analyzing tracks for shooting motions...")
        total_elevated_wrists = 0
        total_extended_elbows = 0
        
        for track_id, track_poses in tracks.items():
            shooting_motions = self.detect_shooting_motion(track_poses)
            
            for shooting_motion in shooting_motions:
                shooting_frames = shooting_motion['shooting_frames']
                peak_frame_data = shooting_motion['peak_frame']
                
                total_elevated_wrists += len(shooting_frames)
                total_extended_elbows += sum(1 for f in shooting_frames if f['elbow_angle'] >= self.min_elbow_extension)
                
                # Create shot event
                shot = ShotEvent(
                    event_id=event_id,
                    shooter_track_id=track_id,
                    frame_start=shooting_frames[0]['frame_id'],
                    frame_peak=peak_frame_data['frame_id'],
                    frame_end=shooting_frames[-1]['frame_id'],
                    release_height_pixels=peak_frame_data['wrist_y'],
                    elbow_angle_at_release=peak_frame_data['elbow_angle'],
                    confidence=peak_frame_data['confidence']
                )
            
                # Transform to court coordinates if available
                if homography_matrix is not None:
                    wrist_pos = peak_frame_data['wrist_position']
                    pixel_point = np.array([wrist_pos[0], wrist_pos[1], 1.0])
                    court_point = homography_matrix @ pixel_point
                    
                    if court_point[2] != 0:
                        shot.release_position_court = (
                            float(court_point[0] / court_point[2]),
                            float(court_point[1] / court_point[2])
                        )
                        
                        # Calculate shot distance and type
                        shot_type, distance = self.classify_shot_type(shot.release_position_court)
                        shot.shot_type = shot_type
                        shot.shot_distance_feet = distance
                
                shots.append(shot)
                event_id += 1
        
        print(f"\nDiagnostics:")
        print(f"  Frames with elevated wrists: {total_elevated_wrists}")
        print(f"  Frames with extended elbows: {total_extended_elbows}")
        print(f"  Initial shots detected: {len(shots)}")
        
        # Filter shots that are too close together (keep highest confidence)
        if min_shot_spacing > 0 and len(shots) > 1:
            filtered_shots = self._filter_clustered_shots(shots, min_shot_spacing)
            print(f"  After temporal filtering: {len(filtered_shots)}")
            shots = filtered_shots
        
        print(f"\n{'='*70}")
        print(f"DETECTED {len(shots)} SHOTS")
        print(f"{'='*70}\n")
        
        for shot in shots:
            print(f"Shot {shot.event_id} (Track {shot.shooter_track_id}):")
            print(f"  Frames: {shot.frame_start}-{shot.frame_peak}-{shot.frame_end}")
            print(f"  Wrist Y position: {shot.release_height_pixels:.1f} px (peak)")
            print(f"  Elbow angle: {shot.elbow_angle_at_release:.1f}°")
            if shot.shot_type:
                print(f"  Type: {shot.shot_type}")
                print(f"  Distance: {shot.shot_distance_feet:.1f} ft")
            print(f"  Confidence: {shot.confidence:.2f}\n")
        
        return shots
    
    def classify_shot_type(self, release_position_court: Tuple[float, float], 
                          hoop_position_court: Tuple[float, float] = (5.25, 25.0)) -> Tuple[str, float]:
        """
        Classify shot type by distance to hoop.
        
        Args:
            release_position_court: (x, y) in feet
            hoop_position_court: Hoop position (default: 5.25 ft from baseline, 25 ft from sideline)
            
        Returns:
            (shot_type, distance_feet)
        """
        x, y = release_position_court
        hoop_x, hoop_y = hoop_position_court
        
        distance = np.sqrt((x - hoop_x)**2 + (y - hoop_y)**2)
        
        if distance < 8:
            shot_type = 'layup'
        elif distance < 23.75:
            shot_type = 'mid_range'
        else:
            shot_type = '3_pointer'
        
        return shot_type, float(distance)
    
    def _filter_clustered_shots(self, shots: List[ShotEvent], min_spacing: int) -> List[ShotEvent]:
        """
        Filter out shots that are too close together temporally.
        Keeps the highest confidence shot in each cluster.
        
        Args:
            shots: List of shot events
            min_spacing: Minimum frames between shot peaks
            
        Returns:
            Filtered list of shots
        """
        if len(shots) <= 1:
            return shots
        
        # Sort by frame peak
        sorted_shots = sorted(shots, key=lambda x: x.frame_peak)
        
        filtered = [sorted_shots[0]]
        
        for shot in sorted_shots[1:]:
            # Check if this shot is far enough from the last kept shot
            last_shot = filtered[-1]
            frame_gap = shot.frame_peak - last_shot.frame_peak
            
            if frame_gap >= min_spacing:
                # Far enough, keep it
                filtered.append(shot)
            else:
                # Too close, keep the one with higher confidence
                if shot.confidence > last_shot.confidence:
                    filtered[-1] = shot
        
        return filtered
    
    def export_shots(self, shots: List[ShotEvent], output_json: str):
        """Export shot events to JSON."""
        output_data = {
            'total_shots': len(shots),
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
    
    parser = argparse.ArgumentParser(description='Detect shots using pose estimation')
    parser.add_argument('--tracked-poses', required=True, help='Tracked poses JSON')
    parser.add_argument('--homography', default=None, help='Homography matrix JSON')
    parser.add_argument('--output', required=True, help='Output shots JSON')
    parser.add_argument('--max-wrist-y', type=float, default=700.0, help='Max wrist Y coordinate (px)')
    parser.add_argument('--min-extension', type=float, default=120.0, help='Min elbow angle (degrees)')
    parser.add_argument('--no-upward-filter', action='store_true', help='Disable upward motion filter')
    parser.add_argument('--no-followthrough-filter', action='store_true', help='Disable follow-through filter')
    parser.add_argument('--min-spacing', type=int, default=30, help='Min frames between shots')
    
    args = parser.parse_args()
    
    detector = PoseBasedShotDetector(
        max_wrist_y=args.max_wrist_y,
        min_elbow_extension=args.min_extension,
        require_upward_motion=not args.no_upward_filter,
        require_follow_through=not args.no_followthrough_filter
    )
    
    shots = detector.detect_shots(args.tracked_poses, args.homography, 
                                  min_shot_spacing=args.min_spacing)
    detector.export_shots(shots, args.output)


if __name__ == '__main__':
    main()
