"""
Screen Detection Module

Detects screens (picks) in basketball using player positioning, movement patterns,
and proximity changes. Screens are identified when:
1. An offensive player is stationary or slow-moving (potential screener)
2. A defender's movement is impeded (collision or path obstruction)
3. A ball handler is nearby (for on-ball screens)

References:
- Basketball analytics: Screen detection using tracking data
- Movement analysis: Velocity and acceleration patterns
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class Screen:
    """Represents a detected screen event"""
    screen_id: int
    frame_id: int
    timestamp: float  # in seconds
    screener_track_id: int
    defender_track_id: int
    ball_handler_track_id: Optional[int]
    screen_type: str  # 'on_ball', 'off_ball'
    screener_position: Tuple[float, float]  # (x, y) in pixels
    screener_position_court: Optional[Tuple[float, float]]  # (x, y) in feet
    defender_velocity_before: float  # pixels per frame
    defender_velocity_after: float
    proximity_distance: float  # distance between screener and defender in pixels
    confidence: float


class ScreenDetector:
    """
    Detects screens using player trajectories and pose data
    """
    
    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        max_screener_velocity: float = 2.0,  # pixels per frame
        min_defender_velocity_change: float = 3.0,  # pixels per frame
        min_proximity_distance: float = 50.0,  # pixels
        max_proximity_distance: float = 150.0,  # pixels
        min_screen_duration: int = 3,  # frames
        ball_handler_proximity: float = 300.0,  # pixels for on-ball screens
    ):
        """
        Initialize screen detector
        
        Args:
            homography_matrix: 3x3 matrix for pixel to court coordinate transformation
            max_screener_velocity: Maximum velocity for player to be considered stationary
            min_defender_velocity_change: Minimum velocity change to indicate obstruction
            min_proximity_distance: Minimum distance for screen contact
            max_proximity_distance: Maximum distance for screen detection
            min_screen_duration: Minimum frames a screen must last
            ball_handler_proximity: Distance to ball handler for on-ball screen classification
        """
        self.homography_matrix = homography_matrix
        self.max_screener_velocity = max_screener_velocity
        self.min_defender_velocity_change = min_defender_velocity_change
        self.min_proximity_distance = min_proximity_distance
        self.max_proximity_distance = max_proximity_distance
        self.min_screen_duration = min_screen_duration
        self.ball_handler_proximity = ball_handler_proximity
    
    def load_tracked_poses(self, filepath: str) -> Dict:
        """Load tracked poses from JSON file"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def load_homography(self, filepath: str):
        """Load homography matrix from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.homography_matrix = np.array(data['homography_matrix'])
    
    def transform_to_court_coords(self, x: float, y: float) -> Optional[Tuple[float, float]]:
        """
        Transform pixel coordinates to court coordinates using homography
        
        Args:
            x, y: Pixel coordinates
            
        Returns:
            (x, y) in feet, or None if homography not available
        """
        if self.homography_matrix is None:
            return None
        
        # Apply homography transformation
        point = np.array([x, y, 1.0])
        transformed = self.homography_matrix @ point
        transformed = transformed / transformed[2]
        
        return (float(transformed[0]), float(transformed[1]))
    
    def calculate_velocity(
        self,
        positions: List[Tuple[float, float]],
        frame_indices: List[int]
    ) -> List[float]:
        """
        Calculate velocity magnitude for each frame
        
        Args:
            positions: List of (x, y) positions
            frame_indices: List of frame numbers
            
        Returns:
            List of velocity magnitudes (pixels per frame)
        """
        velocities = []
        
        for i in range(len(positions)):
            if i == 0:
                velocities.append(0.0)
            else:
                # Calculate displacement
                dx = positions[i][0] - positions[i-1][0]
                dy = positions[i][1] - positions[i-1][1]
                
                # Calculate time difference (should be 1 for consecutive frames)
                dt = frame_indices[i] - frame_indices[i-1]
                if dt == 0:
                    velocities.append(0.0)
                else:
                    velocity = np.sqrt(dx**2 + dy**2) / dt
                    velocities.append(velocity)
        
        return velocities
    
    def get_player_position(self, tracked_pose: Dict) -> Tuple[float, float]:
        """
        Get player position from tracked pose (center of bounding box)
        
        Args:
            tracked_pose: Tracked pose dictionary
            
        Returns:
            (x, y) center position in pixels
        """
        bbox = tracked_pose['bbox']
        x = (bbox['x1'] + bbox['x2']) / 2
        y = (bbox['y1'] + bbox['y2']) / 2
        return (x, y)
    
    def get_wrist_position(self, tracked_pose: Dict) -> Optional[Tuple[float, float]]:
        """
        Get average wrist position from pose keypoints (for ball handling detection)
        
        Args:
            tracked_pose: Tracked pose dictionary
            
        Returns:
            (x, y) average wrist position, or None if not available
        """
        if not tracked_pose.get('has_pose'):
            return None
        
        keypoints = tracked_pose['pose']['keypoints']
        wrist_points = []
        
        for kp in keypoints:
            if kp['name'] in ['left_wrist', 'right_wrist'] and kp['confidence'] > 0.3:
                wrist_points.append((kp['x'], kp['y']))
        
        if not wrist_points:
            return None
        
        # Return average of detected wrists
        avg_x = sum(p[0] for p in wrist_points) / len(wrist_points)
        avg_y = sum(p[1] for p in wrist_points) / len(wrist_points)
        return (avg_x, avg_y)
    
    def identify_ball_handler(
        self,
        frame_data: Dict,
        ball_position: Optional[Tuple[float, float]]
    ) -> Optional[int]:
        """
        Identify the ball handler in a frame using pose-based heuristics
        
        Strategy (since ball detection is sparse - 6% in broadcast footage):
        1. Primary: Use pose heuristics (wrist position, proximity, body orientation)
        2. Fallback: Use ball proximity if ball detected
        
        Basketball analytics literature supports pose-based ball handler identification
        when ball tracking is unreliable (Sports Analytics papers 2018-2024)
        
        Args:
            frame_data: Frame data with tracked poses
            ball_position: Ball position (x, y) if available
            
        Returns:
            Track ID of most likely ball handler, or None
        """
        candidates = []
        
        # Method 1: Pose-based heuristics (works without ball detection)
        for tracked_pose in frame_data['tracked_poses']:
            if not tracked_pose.get('has_pose'):
                continue
            
            ball_handler_score = self._calculate_ball_handler_score(tracked_pose)
            if ball_handler_score > 0.3:  # Threshold for ball handling likelihood
                candidates.append((tracked_pose['track_id'], ball_handler_score))
        
        # Method 2: Ball proximity (if ball detected - rare)
        if ball_position is not None:
            for tracked_pose in frame_data['tracked_poses']:
                wrist_pos = self.get_wrist_position(tracked_pose)
                if wrist_pos is not None:
                    distance = np.sqrt(
                        (wrist_pos[0] - ball_position[0])**2 +
                        (wrist_pos[1] - ball_position[1])**2
                    )
                    if distance < 100:  # Within 100 pixels
                        # Very high confidence - ball detected near wrists
                        candidates.append((tracked_pose['track_id'], 0.95))
        
        if not candidates:
            return None
        
        # Return candidate with highest confidence
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _calculate_ball_handler_score(self, tracked_pose: Dict) -> float:
        """
        Calculate likelihood that player is ball handler based on pose
        
        Heuristics from basketball analytics:
        1. Low wrists (below hips) = dribbling stance
        2. Wrists close together = holding/controlling ball
        3. Wrists in mid-range height (not too low, not overhead)
        
        Args:
            tracked_pose: Tracked pose dictionary
            
        Returns:
            Score [0, 1] indicating ball handling likelihood
        """
        if not tracked_pose.get('has_pose'):
            return 0.0
        
        keypoints = tracked_pose['pose']['keypoints']
        
        # Extract relevant keypoints
        kp_dict = {kp['name']: kp for kp in keypoints if kp['confidence'] > 0.3}
        
        # Need at least wrists
        if 'left_wrist' not in kp_dict or 'right_wrist' not in kp_dict:
            return 0.0
        
        score = 0.0
        factors = 0
        
        left_wrist = kp_dict['left_wrist']
        right_wrist = kp_dict['right_wrist']
        
        avg_wrist_y = (left_wrist['y'] + right_wrist['y']) / 2
        
        # Factor 1: Wrist height (ball handlers have wrists in mid-range)
        # Too high (< 400) = overhead movements
        # Good range (400-900) = dribbling/holding
        # Too low (> 900) = hands at sides
        if 400 < avg_wrist_y < 900:
            # Optimal around 600-700 (waist-chest height)
            if 600 <= avg_wrist_y <= 750:
                height_score = 1.0
            elif 500 <= avg_wrist_y < 600:
                height_score = 0.8
            elif 750 < avg_wrist_y <= 850:
                height_score = 0.8
            else:
                height_score = 0.5
            
            score += height_score
            factors += 1
        
        # Factor 2: Wrist proximity (hands close together = holding ball)
        wrist_distance = np.sqrt(
            (left_wrist['x'] - right_wrist['x'])**2 +
            (left_wrist['y'] - right_wrist['y'])**2
        )
        
        # Closer hands = higher score (< 200 pixels is ball-handling range)
        if wrist_distance < 300:
            proximity_score = 1.0 - (wrist_distance / 300)
            score += proximity_score
            factors += 1
        
        # Factor 3: Compare to hips (if available)
        # Ball handlers have wrists below or near hips (not overhead)
        if 'left_hip' in kp_dict and 'right_hip' in kp_dict:
            avg_hip_y = (kp_dict['left_hip']['y'] + kp_dict['right_hip']['y']) / 2
            
            # Wrists near or below hips = dribbling
            if avg_wrist_y >= avg_hip_y - 50:  # Allow slight margin
                hip_score = 1.0
                score += hip_score
                factors += 1
        
        if factors == 0:
            return 0.0
        
        return score / factors
    
    def group_poses_by_track(self, tracked_poses_data: Dict) -> Dict[int, List[Dict]]:
        """
        Group tracked poses by track_id
        
        Args:
            tracked_poses_data: Full tracked poses JSON data
            
        Returns:
            Dictionary mapping track_id to list of frame data
        """
        tracks = {}
        
        for frame in tracked_poses_data['frames']:
            for tracked_pose in frame['tracked_poses']:
                track_id = tracked_pose['track_id']
                
                if track_id not in tracks:
                    tracks[track_id] = []
                
                tracks[track_id].append({
                    'frame_id': frame['frame_id'],
                    'tracked_pose': tracked_pose
                })
        
        return tracks
    
    def detect_potential_screeners(
        self,
        tracks: Dict[int, List[Dict]],
        frame_window: int = 5
    ) -> Dict[int, List[int]]:
        """
        Detect players who are stationary (potential screeners)
        
        Args:
            tracks: Dictionary of track_id to frame data
            frame_window: Number of frames to check for stationary behavior
            
        Returns:
            Dictionary mapping frame_id to list of potential screener track_ids
        """
        potential_screeners = {}
        
        for track_id, frames in tracks.items():
            if len(frames) < frame_window:
                continue
            
            # Calculate velocities
            positions = [self.get_player_position(f['tracked_pose']) for f in frames]
            frame_indices = [f['frame_id'] for f in frames]
            velocities = self.calculate_velocity(positions, frame_indices)
            
            # Check for stationary periods
            for i in range(len(frames) - frame_window + 1):
                window_velocities = velocities[i:i+frame_window]
                avg_velocity = np.mean(window_velocities)
                
                if avg_velocity <= self.max_screener_velocity:
                    frame_id = frames[i + frame_window // 2]['frame_id']
                    
                    if frame_id not in potential_screeners:
                        potential_screeners[frame_id] = []
                    
                    potential_screeners[frame_id].append(track_id)
        
        return potential_screeners
    
    def detect_defender_obstruction(
        self,
        defender_track: List[Dict],
        screen_frame_idx: int,
        window: int = 3
    ) -> Tuple[bool, float, float]:
        """
        Detect if a defender's movement was obstructed
        
        Args:
            defender_track: List of frame data for defender
            screen_frame_idx: Index in defender_track where screen might occur
            window: Frames before/after to check velocity change
            
        Returns:
            (is_obstructed, velocity_before, velocity_after)
        """
        if screen_frame_idx < window or screen_frame_idx >= len(defender_track) - window:
            return (False, 0.0, 0.0)
        
        # Get positions and velocities
        positions = [self.get_player_position(f['tracked_pose']) for f in defender_track]
        frame_indices = [f['frame_id'] for f in defender_track]
        velocities = self.calculate_velocity(positions, frame_indices)
        
        # Calculate average velocity before and after screen
        velocity_before = np.mean(velocities[screen_frame_idx-window:screen_frame_idx])
        velocity_after = np.mean(velocities[screen_frame_idx+1:screen_frame_idx+window+1])
        
        # Check if velocity decreased significantly (defender slowed down)
        velocity_change = velocity_before - velocity_after
        is_obstructed = velocity_change >= self.min_defender_velocity_change
        
        return (is_obstructed, velocity_before, velocity_after)
    
    def detect_screens(
        self,
        tracked_poses_filepath: str,
        ball_trajectory_filepath: Optional[str] = None
    ) -> List[Screen]:
        """
        Detect all screens in the video
        
        Args:
            tracked_poses_filepath: Path to tracked poses JSON
            ball_trajectory_filepath: Optional path to ball trajectory JSON
            
        Returns:
            List of detected screens
        """
        # Load data
        tracked_poses_data = self.load_tracked_poses(tracked_poses_filepath)
        fps = tracked_poses_data['video_info']['fps']
        
        # Load ball trajectory if available
        ball_positions = {}
        if ball_trajectory_filepath:
            try:
                with open(ball_trajectory_filepath, 'r') as f:
                    ball_data = json.load(f)
                    # Map frame_id to ball position
                    for item in ball_data.get('trajectory', []):
                        frame_id = item.get('frame_id')
                        if frame_id is not None and 'x' in item and 'y' in item:
                            ball_positions[frame_id] = (item['x'], item['y'])
            except:
                pass
        
        # Group poses by track
        tracks = self.group_poses_by_track(tracked_poses_data)
        
        # Detect potential screeners (stationary players)
        potential_screeners = self.detect_potential_screeners(tracks)
        
        screens = []
        screen_id = 0
        
        # For each frame with potential screeners
        for frame_id, screener_ids in potential_screeners.items():
            # Get frame data
            frame_data = next(
                (f for f in tracked_poses_data['frames'] if f['frame_id'] == frame_id),
                None
            )
            if frame_data is None:
                continue
            
            # Identify ball handler
            ball_position = ball_positions.get(frame_id)
            ball_handler_id = self.identify_ball_handler(frame_data, ball_position)
            
            # Get screener positions
            screener_positions = {}
            for tracked_pose in frame_data['tracked_poses']:
                if tracked_pose['track_id'] in screener_ids:
                    screener_positions[tracked_pose['track_id']] = self.get_player_position(tracked_pose)
            
            # Check all other players as potential defenders
            for tracked_pose in frame_data['tracked_poses']:
                defender_id = tracked_pose['track_id']
                
                # Skip if this is a screener or ball handler
                if defender_id in screener_ids or defender_id == ball_handler_id:
                    continue
                
                defender_position = self.get_player_position(tracked_pose)
                
                # Check proximity to each screener
                for screener_id, screener_position in screener_positions.items():
                    distance = np.sqrt(
                        (defender_position[0] - screener_position[0])**2 +
                        (defender_position[1] - screener_position[1])**2
                    )
                    
                    # Check if defender is in screen proximity range
                    if self.min_proximity_distance <= distance <= self.max_proximity_distance:
                        # Check if defender was obstructed
                        defender_track = tracks.get(defender_id, [])
                        if not defender_track:
                            continue
                        
                        # Find index of current frame in defender track
                        defender_frame_idx = next(
                            (i for i, f in enumerate(defender_track) if f['frame_id'] == frame_id),
                            None
                        )
                        if defender_frame_idx is None:
                            continue
                        
                        is_obstructed, vel_before, vel_after = self.detect_defender_obstruction(
                            defender_track,
                            defender_frame_idx
                        )
                        
                        if is_obstructed:
                            # Determine screen type
                            screen_type = 'on_ball'
                            if ball_handler_id is not None:
                                ball_handler_position = None
                                for tp in frame_data['tracked_poses']:
                                    if tp['track_id'] == ball_handler_id:
                                        ball_handler_position = self.get_player_position(tp)
                                        break
                                
                                if ball_handler_position is not None:
                                    distance_to_ball_handler = np.sqrt(
                                        (screener_position[0] - ball_handler_position[0])**2 +
                                        (screener_position[1] - ball_handler_position[1])**2
                                    )
                                    
                                    if distance_to_ball_handler > self.ball_handler_proximity:
                                        screen_type = 'off_ball'
                            else:
                                screen_type = 'off_ball'
                            
                            # Calculate confidence based on velocity change
                            confidence = min(1.0, vel_before / (vel_before + 0.1))
                            
                            # Transform to court coordinates
                            court_coords = self.transform_to_court_coords(*screener_position)
                            
                            screen = Screen(
                                screen_id=screen_id,
                                frame_id=frame_id,
                                timestamp=frame_id / fps,
                                screener_track_id=screener_id,
                                defender_track_id=defender_id,
                                ball_handler_track_id=ball_handler_id,
                                screen_type=screen_type,
                                screener_position=screener_position,
                                screener_position_court=court_coords,
                                defender_velocity_before=vel_before,
                                defender_velocity_after=vel_after,
                                proximity_distance=distance,
                                confidence=confidence
                            )
                            
                            screens.append(screen)
                            screen_id += 1
        
        # Filter by minimum duration (group nearby screens)
        filtered_screens = self._filter_by_duration(screens)
        
        return filtered_screens
    
    def _filter_by_duration(self, screens: List[Screen]) -> List[Screen]:
        """
        Filter screens by minimum duration (remove isolated single-frame detections)
        
        Args:
            screens: List of all detected screens
            
        Returns:
            Filtered list of screens
        """
        if not screens:
            return []
        
        # Group screens by screener and defender pair
        screen_groups = {}
        for screen in screens:
            key = (screen.screener_track_id, screen.defender_track_id)
            if key not in screen_groups:
                screen_groups[key] = []
            screen_groups[key].append(screen)
        
        # Filter each group
        filtered_screens = []
        for key, group in screen_groups.items():
            # Sort by frame_id
            group.sort(key=lambda s: s.frame_id)
            
            # Find continuous sequences
            sequences = []
            current_sequence = [group[0]]
            
            for i in range(1, len(group)):
                if group[i].frame_id - group[i-1].frame_id <= 2:  # Allow 1 frame gap
                    current_sequence.append(group[i])
                else:
                    sequences.append(current_sequence)
                    current_sequence = [group[i]]
            sequences.append(current_sequence)
            
            # Keep sequences that meet minimum duration
            for sequence in sequences:
                if len(sequence) >= self.min_screen_duration:
                    # Keep the screen with highest confidence in sequence
                    best_screen = max(sequence, key=lambda s: s.confidence)
                    filtered_screens.append(best_screen)
        
        return filtered_screens
    
    def export_screens(self, screens: List[Screen], output_path: str):
        """
        Export detected screens to JSON file
        
        Args:
            screens: List of detected screens
            output_path: Path to output JSON file
        """
        # Convert screens to dictionaries
        screens_list = [asdict(screen) for screen in screens]
        
        # Create output structure
        output = {
            'num_screens': len(screens),
            'on_ball_screens': len([s for s in screens if s.screen_type == 'on_ball']),
            'off_ball_screens': len([s for s in screens if s.screen_type == 'off_ball']),
            'screens': screens_list
        }
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"Exported {len(screens)} screens to {output_path}")
        print(f"  On-ball screens: {output['on_ball_screens']}")
        print(f"  Off-ball screens: {output['off_ball_screens']}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect screens (picks) in basketball video')
    parser.add_argument('--tracked-poses', type=str, required=True,
                       help='Path to tracked poses JSON file')
    parser.add_argument('--ball-trajectory', type=str, default=None,
                       help='Optional path to ball trajectory JSON file')
    parser.add_argument('--homography', type=str, default=None,
                       help='Optional path to homography matrix JSON file')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output JSON file')
    parser.add_argument('--max-screener-velocity', type=float, default=2.0,
                       help='Maximum velocity for stationary screener (pixels/frame)')
    parser.add_argument('--min-velocity-change', type=float, default=3.0,
                       help='Minimum defender velocity change to indicate obstruction')
    parser.add_argument('--min-proximity', type=float, default=50.0,
                       help='Minimum proximity distance (pixels)')
    parser.add_argument('--max-proximity', type=float, default=150.0,
                       help='Maximum proximity distance (pixels)')
    parser.add_argument('--min-duration', type=int, default=3,
                       help='Minimum screen duration (frames)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = ScreenDetector(
        max_screener_velocity=args.max_screener_velocity,
        min_defender_velocity_change=args.min_velocity_change,
        min_proximity_distance=args.min_proximity,
        max_proximity_distance=args.max_proximity,
        min_screen_duration=args.min_duration
    )
    
    # Load homography if provided
    if args.homography:
        detector.load_homography(args.homography)
    
    # Detect screens
    print("Detecting screens...")
    screens = detector.detect_screens(
        args.tracked_poses,
        args.ball_trajectory
    )
    
    # Export results
    detector.export_screens(screens, args.output)


if __name__ == '__main__':
    main()
