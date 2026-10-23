"""
Visualize detected shots on video with highlights.

Shows:
- Shot event markers when shooting motion detected
- Shot type and distance
- Track ID of shooter
- Frame timeline
"""

import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


class ShotVisualizer:
    """Visualize detected shots on video."""
    
    def __init__(self, shots_json: str):
        """
        Initialize visualizer.
        
        Args:
            shots_json: Path to shots JSON file
        """
        with open(shots_json, 'r') as f:
            data = json.load(f)
        
        self.shots = data['shots']
        
        # Create shot lookup by frame for fast access
        self.shot_by_frame = {}
        for shot in self.shots:
            for frame_id in range(shot['frame_start'], shot['frame_end'] + 1):
                if frame_id not in self.shot_by_frame:
                    self.shot_by_frame[frame_id] = []
                self.shot_by_frame[frame_id].append(shot)
        
        print(f"✅ Loaded {len(self.shots)} shots")
        print(f"   Frames with shots: {len(self.shot_by_frame)}")
    
    def draw_shot_indicator(self, frame: np.ndarray, shot: dict, is_peak: bool = False):
        """
        Draw shot indicator on frame.
        
        Args:
            frame: Video frame
            shot: Shot data
            is_peak: Whether this is the peak frame
        """
        h, w = frame.shape[:2]
        
        # Color coding
        if is_peak:
            color = (0, 255, 0)  # Green for peak
            thickness = 4
            label = "🏀 SHOT!"
        else:
            color = (0, 165, 255)  # Orange for shooting motion
            thickness = 2
            label = "Shooting..."
        
        # Draw border around frame
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, thickness)
        
        # Draw shot info box
        info_y = 60
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.0
        
        # Shot label
        cv2.putText(frame, label, (30, info_y), font, font_scale, color, 2)
        
        # Shot details
        info_y += 40
        shot_type = shot.get('shot_type', 'unknown').replace('_', ' ').title()
        distance = shot.get('shot_distance_feet', 0)
        track_id = shot.get('shooter_track_id', '?')
        
        details = f"Track {track_id} | {shot_type}"
        if distance > 0:
            details += f" | {distance:.1f} ft"
        
        cv2.putText(frame, details, (30, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Frame info
        info_y += 40
        frame_info = f"Frames {shot['frame_start']}-{shot['frame_end']} (Peak: {shot['frame_peak']})"
        cv2.putText(frame, frame_info, (30, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Confidence
        info_y += 30
        conf_text = f"Confidence: {shot['confidence']:.2f}"
        cv2.putText(frame, conf_text, (30, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return frame
    
    def draw_timeline(self, frame: np.ndarray, current_frame: int, total_frames: int):
        """
        Draw timeline showing shot events.
        
        Args:
            frame: Video frame
            current_frame: Current frame number
            total_frames: Total frames in video
        """
        h, w = frame.shape[:2]
        
        # Timeline bar
        bar_height = 40
        bar_y = h - bar_height - 20
        bar_x_start = 50
        bar_x_end = w - 50
        bar_width = bar_x_end - bar_x_start
        
        # Background
        cv2.rectangle(frame, (bar_x_start, bar_y), 
                     (bar_x_end, bar_y + bar_height), (50, 50, 50), -1)
        
        # Shot markers
        for shot in self.shots:
            peak_frame = shot['frame_peak']
            peak_x = bar_x_start + int((peak_frame / total_frames) * bar_width)
            
            # Draw shot marker
            cv2.circle(frame, (peak_x, bar_y + bar_height // 2), 8, (0, 255, 0), -1)
            cv2.circle(frame, (peak_x, bar_y + bar_height // 2), 10, (255, 255, 255), 2)
        
        # Current position
        current_x = bar_x_start + int((current_frame / total_frames) * bar_width)
        cv2.line(frame, (current_x, bar_y), 
                (current_x, bar_y + bar_height), (255, 255, 255), 3)
        
        # Frame number
        frame_text = f"Frame {current_frame}/{total_frames}"
        cv2.putText(frame, frame_text, (bar_x_start, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def visualize(self, video_path: str, output_path: str):
        """
        Create visualization video.
        
        Args:
            video_path: Input video path
            output_path: Output video path
        """
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"\n{'='*70}")
        print("CREATING SHOT VISUALIZATION")
        print(f"{'='*70}\n")
        print(f"Input: {video_path}")
        print(f"Output: {output_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}\n")
        
        frame_id = 0
        
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if current frame has shot
                if frame_id in self.shot_by_frame:
                    shots_in_frame = self.shot_by_frame[frame_id]
                    
                    # Draw indicator for each shot (usually just one)
                    for shot in shots_in_frame:
                        is_peak = (frame_id == shot['frame_peak'])
                        frame = self.draw_shot_indicator(frame, shot, is_peak)
                
                # Draw timeline
                frame = self.draw_timeline(frame, frame_id, total_frames)
                
                out.write(frame)
                frame_id += 1
                pbar.update(1)
        
        cap.release()
        out.release()
        
        print(f"\n✅ Saved visualization: {output_path}")
        
        # Print shot summary
        print(f"\n{'='*70}")
        print("SHOT SUMMARY")
        print(f"{'='*70}\n")
        
        # Sort by confidence
        sorted_shots = sorted(self.shots, key=lambda x: x['confidence'], reverse=True)
        
        for i, shot in enumerate(sorted_shots[:10], 1):
            shot_type = shot.get('shot_type', 'unknown').replace('_', ' ').title()
            distance = shot.get('shot_distance_feet', 0)
            print(f"{i}. Frame {shot['frame_peak']:3d} | Track {shot['shooter_track_id']:3d} | "
                  f"{shot_type:12s} | {distance:5.1f}ft | Conf: {shot['confidence']:.2f}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize detected shots')
    parser.add_argument('--video', required=True, help='Input video path')
    parser.add_argument('--shots', required=True, help='Shots JSON path')
    parser.add_argument('--output', required=True, help='Output video path')
    
    args = parser.parse_args()
    
    visualizer = ShotVisualizer(args.shots)
    visualizer.visualize(args.video, args.output)


if __name__ == '__main__':
    main()
