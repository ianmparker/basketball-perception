"""
Visualize detected screens on video with annotations

Shows:
- Screen indicators with colored borders
- Screener, defender, ball handler labels
- Screen type (on-ball/off-ball)
- Timeline showing screen events
"""

import cv2
import json
import numpy as np
from typing import Dict, List, Tuple
from tqdm import tqdm


class ScreenVisualizer:
    """Visualize detected screens on video"""
    
    def __init__(self, screens_path: str):
        """
        Initialize visualizer
        
        Args:
            screens_path: Path to screens JSON file
        """
        with open(screens_path, 'r') as f:
            self.screens_data = json.load(f)
        
        self.screens = self.screens_data['screens']
        
        # Create lookup by frame for fast access
        self.screens_by_frame = {}
        for screen in self.screens:
            frame_id = screen['frame_id']
            if frame_id not in self.screens_by_frame:
                self.screens_by_frame[frame_id] = []
            self.screens_by_frame[frame_id].append(screen)
        
        # Colors
        self.colors = {
            'screen': (0, 255, 0),      # Green for screen event
            'screener': (255, 165, 0),  # Orange for screener
            'defender': (0, 0, 255),    # Red for defender
            'ball_handler': (255, 255, 0)  # Cyan for ball handler
        }
    
    def draw_screen_indicator(
        self,
        frame: np.ndarray,
        screen: Dict,
        tracked_poses_frame: Dict
    ) -> np.ndarray:
        """
        Draw screen indicator on frame
        
        Args:
            frame: BGR image
            screen: Screen detection dictionary
            tracked_poses_frame: Frame data with tracked poses
            
        Returns:
            Annotated frame
        """
        frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Get track IDs
        screener_id = screen['screener_track_id']
        defender_id = screen['defender_track_id']
        ball_handler_id = screen['ball_handler_track_id']
        
        # Find bounding boxes
        screener_bbox = None
        defender_bbox = None
        ball_handler_bbox = None
        
        for tracked_pose in tracked_poses_frame.get('tracked_poses', []):
            track_id = tracked_pose['track_id']
            bbox = tracked_pose['bbox']
            
            if track_id == screener_id:
                screener_bbox = bbox
            elif track_id == defender_id:
                defender_bbox = bbox
            elif track_id == ball_handler_id:
                ball_handler_bbox = bbox
        
        # Draw bounding boxes with labels
        if screener_bbox:
            x1, y1 = int(screener_bbox['x1']), int(screener_bbox['y1'])
            x2, y2 = int(screener_bbox['x2']), int(screener_bbox['y2'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['screener'], 3)
            
            # Label
            label = f"SCREENER #{screener_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['screener'], 2)
        
        if defender_bbox:
            x1, y1 = int(defender_bbox['x1']), int(defender_bbox['y1'])
            x2, y2 = int(defender_bbox['x2']), int(defender_bbox['y2'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['defender'], 3)
            
            label = f"DEFENDER #{defender_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['defender'], 2)
        
        if ball_handler_bbox:
            x1, y1 = int(ball_handler_bbox['x1']), int(ball_handler_bbox['y1'])
            x2, y2 = int(ball_handler_bbox['x2']), int(ball_handler_bbox['y2'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['ball_handler'], 3)
            
            label = f"BALL HANDLER #{ball_handler_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['ball_handler'], 2)
        
        # Draw screen event indicator
        screen_type = screen['screen_type'].upper().replace('_', '-')
        screen_text = f"SCREEN: {screen_type}"
        
        # Big green text at top
        text_size = cv2.getTextSize(screen_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (w - text_size[0]) // 2
        text_y = 60
        
        # Background rectangle
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, screen_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.colors['screen'], 3)
        
        # Draw screen details
        details_y = 120
        details = [
            f"Frame: {screen['frame_id']} (t={screen['timestamp']:.2f}s)",
            f"Confidence: {screen['confidence']:.3f}",
            f"Defender velocity: {screen['defender_velocity_before']:.1f} -> {screen['defender_velocity_after']:.1f} px/frame",
            f"Proximity: {screen['proximity_distance']:.1f} pixels"
        ]
        
        for detail in details:
            cv2.putText(frame, detail, (20, details_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            details_y += 30
        
        return frame
    
    def draw_timeline(self, frame: np.ndarray, current_frame: int, total_frames: int) -> np.ndarray:
        """
        Draw timeline bar at bottom with screen markers
        
        Args:
            frame: BGR image
            current_frame: Current frame number
            total_frames: Total frames in video
            
        Returns:
            Frame with timeline
        """
        h, w = frame.shape[:2]
        
        # Timeline bar dimensions
        bar_height = 40
        bar_y = h - bar_height - 10
        bar_x_start = 50
        bar_x_end = w - 50
        bar_width = bar_x_end - bar_x_start
        
        # Draw background
        cv2.rectangle(frame, (bar_x_start, bar_y), (bar_x_end, bar_y + bar_height),
                     (50, 50, 50), -1)
        
        # Draw screen markers
        for screen in self.screens:
            screen_frame = screen['frame_id']
            marker_x = bar_x_start + int((screen_frame / total_frames) * bar_width)
            
            # Draw vertical line
            cv2.line(frame, (marker_x, bar_y), (marker_x, bar_y + bar_height),
                    self.colors['screen'], 3)
            
            # Draw circle
            cv2.circle(frame, (marker_x, bar_y + bar_height // 2), 8,
                      self.colors['screen'], -1)
        
        # Draw current position
        current_x = bar_x_start + int((current_frame / total_frames) * bar_width)
        cv2.line(frame, (current_x, bar_y), (current_x, bar_y + bar_height),
                (255, 255, 255), 2)
        
        # Frame counter
        frame_text = f"{current_frame}/{total_frames}"
        cv2.putText(frame, frame_text, (bar_x_start, bar_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def visualize(
        self,
        video_path: str,
        tracked_poses_path: str,
        output_path: str,
        show_all_frames: bool = False,
        context_frames: int = 30
    ):
        """
        Create annotated video showing screens
        
        Args:
            video_path: Path to input video
            tracked_poses_path: Path to tracked poses JSON
            output_path: Path to output video
            show_all_frames: If True, annotate all frames. If False, only frames near screens
            context_frames: Number of frames before/after screen to show
        """
        # Load tracked poses
        with open(tracked_poses_path, 'r') as f:
            tracked_poses_data = json.load(f)
        
        # Create frame lookup
        poses_by_frame = {
            frame['frame_id']: frame 
            for frame in tracked_poses_data['frames']
        }
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine which frames to annotate
        if show_all_frames:
            annotate_frames = set(range(total_frames))
        else:
            annotate_frames = set()
            for screen in self.screens:
                screen_frame = screen['frame_id']
                for offset in range(-context_frames, context_frames + 1):
                    frame_id = screen_frame + offset
                    if 0 <= frame_id < total_frames:
                        annotate_frames.add(frame_id)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Creating annotated video...")
        print(f"  Input: {video_path}")
        print(f"  Output: {output_path}")
        print(f"  Screens: {len(self.screens)}")
        print(f"  Annotating: {len(annotate_frames)} / {total_frames} frames")
        
        frame_id = 0
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only annotate selected frames
            if frame_id in annotate_frames:
                # Check if this frame has a screen
                if frame_id in self.screens_by_frame:
                    for screen in self.screens_by_frame[frame_id]:
                        tracked_poses_frame = poses_by_frame.get(frame_id, {})
                        frame = self.draw_screen_indicator(frame, screen, tracked_poses_frame)
                
                # Always draw timeline on annotated frames
                frame = self.draw_timeline(frame, frame_id, total_frames)
            
            out.write(frame)
            frame_id += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"\nCreated: {output_path}")


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize detected screens on video')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--screens', type=str, required=True, help='Path to screens JSON file')
    parser.add_argument('--tracked-poses', type=str, required=True, help='Path to tracked poses JSON')
    parser.add_argument('--output', type=str, required=True, help='Path to output video')
    parser.add_argument('--all-frames', action='store_true', 
                       help='Annotate all frames (default: only frames near screens)')
    parser.add_argument('--context-frames', type=int, default=30,
                       help='Number of frames before/after screen to annotate (default: 30)')
    
    args = parser.parse_args()
    
    visualizer = ScreenVisualizer(args.screens)
    visualizer.visualize(
        args.video,
        args.tracked_poses,
        args.output,
        show_all_frames=args.all_frames,
        context_frames=args.context_frames
    )


if __name__ == '__main__':
    main()
