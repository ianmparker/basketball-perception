"""
Ball-centric screen detection (Phase 3.7)

User directive: "if we based everything on the location of the ball we would better be able to narrow the scope"

Strategy:
1. Find ball position (from detections or interpolation)
2. Identify player closest to ball = ball handler
3. Find stationary player near ball handler = screener candidate
4. Find defender whose movement is obstructed = defender
5. Verify contact at specific frames (e.g., frame 149)
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class BallCentricScreenDetector:
    """Detect screens using ball position first to identify relevant players"""
    
    def __init__(
        self,
        tracked_poses_file: str,
        ball_detections_file: str,
        ball_search_radius: int = 10,
        max_ball_distance: float = 500.0
    ):
        """
        Initialize detector with ball-centric approach
        
        Args:
            tracked_poses_file: Path to tracked poses JSON
            ball_detections_file: Path to ball detections JSON
            ball_search_radius: How many frames to search for ball
            max_ball_distance: Maximum distance from ball to be considered ball handler (px)
        """
        self.ball_search_radius = ball_search_radius
        self.max_ball_distance = max_ball_distance
        
        # Load data
        with open(tracked_poses_file, 'r') as f:
            self.tracked_poses = json.load(f)
        
        with open(ball_detections_file, 'r') as f:
            self.ball_detections = json.load(f)
    
    def find_ball_position(self, frame_id: int) -> Optional[Tuple[float, float]]:
        """
        Find ball position at or near given frame
        
        Strategy: Search current frame and nearby frames (±search_radius)
        Use interpolation if ball found in adjacent frames
        
        Args:
            frame_id: Frame to search
            
        Returns:
            (x, y) ball position or None
        """
        ball_positions = {}
        
        # Search nearby frames
        for offset in range(-self.ball_search_radius, self.ball_search_radius + 1):
            check_frame = frame_id + offset
            if check_frame < 0:
                continue
            
            # Find ball in this frame
            for frame_data in self.ball_detections.get('frames', []):
                if frame_data['frame_id'] == check_frame:
                    balls = [d for d in frame_data['detections'] if d['class_id'] == 1]
                    if balls:
                        # Use most confident detection
                        best_ball = max(balls, key=lambda b: b['confidence'])
                        bbox = best_ball['bbox']
                        ball_x = (bbox['x1'] + bbox['x2']) / 2
                        ball_y = (bbox['y1'] + bbox['y2']) / 2
                        ball_positions[check_frame] = (ball_x, ball_y, best_ball['confidence'])
        
        if not ball_positions:
            return None
        
        # If ball found in exact frame, use it
        if frame_id in ball_positions:
            return ball_positions[frame_id][:2]
        
        # Interpolate from nearest frames
        frames = sorted(ball_positions.keys())
        
        # Find frames before and after target
        before = [f for f in frames if f < frame_id]
        after = [f for f in frames if f > frame_id]
        
        if before and after:
            # Linear interpolation
            f1 = before[-1]
            f2 = after[0]
            x1, y1, _ = ball_positions[f1]
            x2, y2, _ = ball_positions[f2]
            
            t = (frame_id - f1) / (f2 - f1)
            ball_x = x1 + t * (x2 - x1)
            ball_y = y1 + t * (y2 - y1)
            return (ball_x, ball_y)
        
        # Use nearest frame
        nearest_frame = min(frames, key=lambda f: abs(f - frame_id))
        return ball_positions[nearest_frame][:2]
    
    def identify_ball_handler(self, frame_id: int, ball_pos: Tuple[float, float]) -> Optional[int]:
        """
        Identify player closest to ball = ball handler
        
        Args:
            frame_id: Frame number
            ball_pos: Ball position (x, y)
            
        Returns:
            Track ID of ball handler or None
        """
        frame_key = str(frame_id)
        if frame_key not in self.tracked_poses:
            return None
        
        frame_data = self.tracked_poses[frame_key]
        
        min_dist = float('inf')
        ball_handler_id = None
        
        for track_id, data in frame_data.items():
            bbox = data['bbox']
            player_x = (bbox['x1'] + bbox['x2']) / 2
            player_y = (bbox['y1'] + bbox['y2']) / 2
            
            dist = np.sqrt((player_x - ball_pos[0])**2 + (player_y - ball_pos[1])**2)
            
            if dist < min_dist:
                min_dist = dist
                ball_handler_id = int(track_id)
        
        # Only return if player reasonably close to ball
        if ball_handler_id and min_dist < self.max_ball_distance:
            return ball_handler_id
        
        return None
    
    def detect_screens(
        self,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        min_proximity: float = 50.0,
        max_proximity: float = 150.0,
        max_screener_velocity: float = 2.0,
        min_defender_velocity_change: float = 3.0
    ) -> List[Dict]:
        """
        Detect screens using ball-centric approach
        
        Algorithm:
        1. For each frame, find ball position
        2. Identify player closest to ball = ball handler
        3. Find stationary players near ball handler = screener candidates
        4. Find defenders whose path is obstructed
        5. Verify contact timing
        
        Args:
            start_frame: Starting frame
            end_frame: Ending frame (None = all frames)
            min_proximity: Minimum distance for screen (px)
            max_proximity: Maximum distance for screen (px)
            max_screener_velocity: Max velocity for stationary screener (px/frame)
            min_defender_velocity_change: Min velocity change for obstruction (px/frame)
            
        Returns:
            List of detected screens with metadata
        """
        screens = []
        
        # Get all frames
        all_frames = sorted([int(f) for f in self.tracked_poses.keys() if f.isdigit()])
        if end_frame is None:
            end_frame = all_frames[-1]
        
        frames_to_check = [f for f in all_frames if start_frame <= f <= end_frame]
        
        # Calculate velocities for all tracks
        velocities = self._calculate_velocities()
        
        for frame_id in frames_to_check:
            # Step 1: Find ball position
            ball_pos = self.find_ball_position(frame_id)
            if not ball_pos:
                continue
            
            # Step 2: Identify ball handler
            ball_handler_id = self.identify_ball_handler(frame_id, ball_pos)
            if not ball_handler_id:
                continue
            
            # Step 3: Find potential screeners (stationary near ball handler)
            screeners = self._find_potential_screeners(
                frame_id,
                ball_handler_id,
                velocities,
                max_screener_velocity,
                min_proximity,
                max_proximity
            )
            
            # Step 4: For each screener, check for defender obstruction
            for screener_id in screeners:
                defenders = self._find_obstructed_defenders(
                    frame_id,
                    ball_handler_id,
                    screener_id,
                    velocities,
                    min_defender_velocity_change,
                    min_proximity,
                    max_proximity
                )
                
                for defender_id in defenders:
                    # Verify this is a valid screen
                    screen = self._create_screen_event(
                        frame_id,
                        ball_handler_id,
                        screener_id,
                        defender_id,
                        ball_pos,
                        velocities
                    )
                    
                    if screen:
                        screens.append(screen)
        
        # Remove duplicates and merge nearby detections
        screens = self._merge_duplicate_screens(screens)
        
        return screens
    
    def _calculate_velocities(self) -> Dict[int, Dict[int, float]]:
        """
        Calculate velocity for each track across all frames
        
        Returns:
            Dict mapping frame_id -> track_id -> velocity (px/frame)
        """
        velocities = {}
        
        all_frames = sorted([int(f) for f in self.tracked_poses.keys() if f.isdigit()])
        
        for i in range(len(all_frames) - 1):
            frame1 = all_frames[i]
            frame2 = all_frames[i + 1]
            
            frame1_data = self.tracked_poses[str(frame1)]
            frame2_data = self.tracked_poses[str(frame2)]
            
            velocities[frame2] = {}
            
            # Calculate velocity for each track
            for track_id in frame1_data:
                if track_id in frame2_data:
                    bbox1 = frame1_data[track_id]['bbox']
                    bbox2 = frame2_data[track_id]['bbox']
                    
                    x1 = (bbox1['x1'] + bbox1['x2']) / 2
                    y1 = (bbox1['y1'] + bbox1['y2']) / 2
                    x2 = (bbox2['x1'] + bbox2['x2']) / 2
                    y2 = (bbox2['y1'] + bbox2['y2']) / 2
                    
                    velocity = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    velocities[frame2][int(track_id)] = velocity
        
        return velocities
    
    def _find_potential_screeners(
        self,
        frame_id: int,
        ball_handler_id: int,
        velocities: Dict,
        max_velocity: float,
        min_proximity: float,
        max_proximity: float
    ) -> List[int]:
        """Find stationary players near ball handler"""
        frame_key = str(frame_id)
        if frame_key not in self.tracked_poses:
            return []
        
        frame_data = self.tracked_poses[frame_key]
        handler_bbox = frame_data[str(ball_handler_id)]['bbox']
        handler_x = (handler_bbox['x1'] + handler_bbox['x2']) / 2
        handler_y = (handler_bbox['y1'] + handler_bbox['y2']) / 2
        
        screeners = []
        
        for track_id, data in frame_data.items():
            track_id = int(track_id)
            
            # Skip ball handler
            if track_id == ball_handler_id:
                continue
            
            # Check velocity (must be stationary)
            if frame_id in velocities and track_id in velocities[frame_id]:
                if velocities[frame_id][track_id] > max_velocity:
                    continue
            
            # Check proximity to ball handler
            bbox = data['bbox']
            player_x = (bbox['x1'] + bbox['x2']) / 2
            player_y = (bbox['y1'] + bbox['y2']) / 2
            
            dist = np.sqrt((player_x - handler_x)**2 + (player_y - handler_y)**2)
            
            if min_proximity <= dist <= max_proximity:
                screeners.append(track_id)
        
        return screeners
    
    def _find_obstructed_defenders(
        self,
        frame_id: int,
        ball_handler_id: int,
        screener_id: int,
        velocities: Dict,
        min_velocity_change: float,
        min_proximity: float,
        max_proximity: float
    ) -> List[int]:
        """Find defenders whose movement is obstructed by screen"""
        frame_key = str(frame_id)
        if frame_key not in self.tracked_poses:
            return []
        
        frame_data = self.tracked_poses[frame_key]
        screener_bbox = frame_data[str(screener_id)]['bbox']
        screener_x = (screener_bbox['x1'] + screener_bbox['x2']) / 2
        screener_y = (screener_bbox['y1'] + screener_bbox['y2']) / 2
        
        defenders = []
        
        # Check previous frames for velocity change
        all_frames = sorted([int(f) for f in self.tracked_poses.keys() if f.isdigit()])
        frame_idx = all_frames.index(frame_id)
        
        if frame_idx < 3:
            return []
        
        for track_id, data in frame_data.items():
            track_id = int(track_id)
            
            # Skip ball handler and screener
            if track_id in [ball_handler_id, screener_id]:
                continue
            
            # Check proximity to screener
            bbox = data['bbox']
            player_x = (bbox['x1'] + bbox['x2']) / 2
            player_y = (bbox['y1'] + bbox['y2']) / 2
            
            dist = np.sqrt((player_x - screener_x)**2 + (player_y - screener_y)**2)
            
            if dist > max_proximity:
                continue
            
            # Check velocity change (defender slowed down)
            prev_frame = all_frames[frame_idx - 3]
            curr_frame = frame_id
            
            if prev_frame in velocities and curr_frame in velocities:
                if track_id in velocities[prev_frame] and track_id in velocities[curr_frame]:
                    prev_vel = velocities[prev_frame][track_id]
                    curr_vel = velocities[curr_frame][track_id]
                    
                    velocity_change = prev_vel - curr_vel
                    
                    if velocity_change >= min_velocity_change:
                        defenders.append(track_id)
        
        return defenders
    
    def _create_screen_event(
        self,
        frame_id: int,
        ball_handler_id: int,
        screener_id: int,
        defender_id: int,
        ball_pos: Tuple[float, float],
        velocities: Dict
    ) -> Optional[Dict]:
        """Create screen event with metadata"""
        frame_key = str(frame_id)
        if frame_key not in self.tracked_poses:
            return None
        
        frame_data = self.tracked_poses[frame_key]
        
        # Get positions
        handler_bbox = frame_data[str(ball_handler_id)]['bbox']
        screener_bbox = frame_data[str(screener_id)]['bbox']
        defender_bbox = frame_data[str(defender_id)]['bbox']
        
        handler_x = (handler_bbox['x1'] + handler_bbox['x2']) / 2
        handler_y = (handler_bbox['y1'] + handler_bbox['y2']) / 2
        
        screener_x = (screener_bbox['x1'] + screener_bbox['x2']) / 2
        screener_y = (screener_bbox['y1'] + screener_bbox['y2']) / 2
        
        defender_x = (defender_bbox['x1'] + defender_bbox['x2']) / 2
        defender_y = (defender_bbox['y1'] + defender_bbox['y2']) / 2
        
        # Calculate metrics
        screener_defender_dist = np.sqrt(
            (screener_x - defender_x)**2 + (screener_y - defender_y)**2
        )
        
        ball_handler_dist = np.sqrt(
            (ball_pos[0] - handler_x)**2 + (ball_pos[1] - handler_y)**2
        )
        
        # Get velocities
        prev_vel = velocities.get(frame_id - 3, {}).get(defender_id, 0)
        curr_vel = velocities.get(frame_id, {}).get(defender_id, 0)
        
        return {
            'frame_id': frame_id,
            'screen_type': 'on-ball',
            'ball_handler': ball_handler_id,
            'screener': screener_id,
            'defender': defender_id,
            'ball_position': {'x': ball_pos[0], 'y': ball_pos[1]},
            'ball_handler_distance': ball_handler_dist,
            'screener_defender_proximity': screener_defender_dist,
            'defender_velocity_before': prev_vel,
            'defender_velocity_after': curr_vel,
            'confidence': 0.8 if ball_handler_dist < 200 else 0.6
        }
    
    def _merge_duplicate_screens(self, screens: List[Dict], frame_threshold: int = 10) -> List[Dict]:
        """Merge screens detected in nearby frames"""
        if not screens:
            return []
        
        # Sort by frame
        screens = sorted(screens, key=lambda s: s['frame_id'])
        
        merged = []
        current_group = [screens[0]]
        
        for screen in screens[1:]:
            # If within threshold and same players, merge
            if (screen['frame_id'] - current_group[-1]['frame_id'] <= frame_threshold and
                screen['screener'] == current_group[0]['screener'] and
                screen['defender'] == current_group[0]['defender']):
                current_group.append(screen)
            else:
                # Save group and start new
                merged.append(self._average_screen_group(current_group))
                current_group = [screen]
        
        # Add last group
        if current_group:
            merged.append(self._average_screen_group(current_group))
        
        return merged
    
    def _average_screen_group(self, group: List[Dict]) -> Dict:
        """Average metrics for screens in same group"""
        # Use frame with highest confidence
        best = max(group, key=lambda s: s['confidence'])
        return best


def main():
    """Test ball-centric screen detection"""
    detector = BallCentricScreenDetector(
        tracked_poses_file='luka_3pt_tracked_poses.json',
        ball_detections_file='luka_3pt_detections_enhanced.json',
        ball_search_radius=10,
        max_ball_distance=500.0
    )
    
    # Detect screens
    screens = detector.detect_screens(
        start_frame=120,
        end_frame=160,
        min_proximity=50.0,
        max_proximity=150.0,
        max_screener_velocity=2.0,
        min_defender_velocity_change=3.0
    )
    
    print(f"\n{'='*80}")
    print(f"Ball-Centric Screen Detection Results")
    print(f"{'='*80}\n")
    
    print(f"Detected {len(screens)} screens\n")
    
    for i, screen in enumerate(screens, 1):
        print(f"Screen {i}:")
        print(f"  Frame: {screen['frame_id']}")
        print(f"  Type: {screen['screen_type']}")
        print(f"  Ball Handler: Track #{screen['ball_handler']}")
        print(f"  Screener: Track #{screen['screener']}")
        print(f"  Defender: Track #{screen['defender']}")
        print(f"  Ball Position: ({screen['ball_position']['x']:.1f}, {screen['ball_position']['y']:.1f})")
        print(f"  Ball-Handler Distance: {screen['ball_handler_distance']:.1f}px")
        print(f"  Screener-Defender Proximity: {screen['screener_defender_proximity']:.1f}px")
        print(f"  Defender Velocity: {screen['defender_velocity_before']:.1f} → {screen['defender_velocity_after']:.1f}")
        print(f"  Confidence: {screen['confidence']:.3f}")
        print()
    
    # Save results
    output_file = 'luka_3pt_screens_v3_ball_centric.json'
    with open(output_file, 'w') as f:
        json.dump({'screens': screens}, f, indent=2)
    
    print(f"✅ Saved to {output_file}")


if __name__ == '__main__':
    main()
