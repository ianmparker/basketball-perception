"""
Draw NBA basketball court template with trajectories and overlays.

This module creates visualizations of player trajectories on a properly scaled
NBA court template (94ft × 50ft) with all standard court markings.

Court dimensions (NBA regulation):
- Full court: 94 feet × 50 feet
- 3-point line: 23.75 feet from hoop (corners: 22 feet)
- Key (paint): 16 feet × 19 feet
- Free throw line: 15 feet from backboard (19 feet from baseline)
- Center circle: 12 feet diameter
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
import colorsys


class CourtDrawer:
    """Draw NBA court with trajectories and annotations."""
    
    # NBA court dimensions in feet
    COURT_LENGTH = 94.0
    COURT_WIDTH = 50.0
    
    # Key dimensions
    KEY_WIDTH = 16.0
    KEY_LENGTH = 19.0
    
    # 3-point line
    THREE_POINT_SIDE = 23.75  # feet from hoop
    THREE_POINT_CORNER = 22.0  # feet from baseline in corners
    
    # Free throw
    FREE_THROW_LINE = 19.0  # feet from baseline
    FREE_THROW_CIRCLE_RADIUS = 6.0  # feet
    
    # Center circle
    CENTER_CIRCLE_RADIUS = 6.0  # feet
    
    # Hoop position
    HOOP_FROM_BASELINE = 5.25  # feet (4 feet from backboard, 1.25ft to center)
    
    def __init__(self, pixels_per_foot: int = 10, background_color: Tuple[int, int, int] = (20, 50, 20)):
        """
        Initialize court drawer.
        
        Args:
            pixels_per_foot: Scale factor (higher = larger image)
            background_color: RGB color for court background
        """
        self.ppf = pixels_per_foot
        self.bg_color = background_color
        
        # Calculate image dimensions
        self.width_px = int(self.COURT_LENGTH * self.ppf)
        self.height_px = int(self.COURT_WIDTH * self.ppf)
        
        # Create base court image
        self.court_img = self.draw_court()
        
        print(f"✅ Court template created: {self.width_px}×{self.height_px} pixels")
        print(f"   Scale: {self.ppf} pixels/foot")
    
    def feet_to_pixels(self, x_feet: float, y_feet: float) -> Tuple[int, int]:
        """
        Convert court coordinates (feet) to pixel coordinates.
        
        Args:
            x_feet: X coordinate in feet (0 = baseline)
            y_feet: Y coordinate in feet (0 = left sideline)
            
        Returns:
            (x_px, y_px) pixel coordinates
        """
        x_px = int(x_feet * self.ppf)
        y_px = int(y_feet * self.ppf)
        return (x_px, y_px)
    
    def draw_court(self) -> np.ndarray:
        """
        Draw NBA court template with all markings.
        
        Returns:
            Court image (H, W, 3) in BGR format
        """
        # Create blank court
        court = np.full((self.height_px, self.width_px, 3), self.bg_color, dtype=np.uint8)
        
        line_color = (255, 255, 255)  # White lines
        line_thickness = 2
        
        # Court perimeter
        cv2.rectangle(court, (0, 0), (self.width_px - 1, self.height_px - 1), 
                     line_color, line_thickness)
        
        # Halfcourt line
        half_x = self.width_px // 2
        cv2.line(court, (half_x, 0), (half_x, self.height_px), line_color, line_thickness)
        
        # Center circle
        center_x, center_y = half_x, self.height_px // 2
        radius = int(self.CENTER_CIRCLE_RADIUS * self.ppf)
        cv2.circle(court, (center_x, center_y), radius, line_color, line_thickness)
        
        # Draw both ends (left and right)
        for baseline_x in [0, self.width_px]:
            flip = 1 if baseline_x == 0 else -1
            self._draw_half_court(court, baseline_x, flip, line_color, line_thickness)
        
        return court
    
    def _draw_half_court(self, court: np.ndarray, baseline_x: int, flip: int,
                         line_color: Tuple[int, int, int], line_thickness: int):
        """Draw one half of the court (key, 3-point, free throw)."""
        
        # Key (paint)
        key_width_px = int(self.KEY_WIDTH * self.ppf)
        key_length_px = int(self.KEY_LENGTH * self.ppf)
        key_top = (self.height_px - key_width_px) // 2
        
        if flip == 1:  # Left side
            key_rect = [(baseline_x, key_top), 
                       (baseline_x + key_length_px, key_top + key_width_px)]
        else:  # Right side
            key_rect = [(baseline_x - key_length_px, key_top),
                       (baseline_x, key_top + key_width_px)]
        
        cv2.rectangle(court, key_rect[0], key_rect[1], line_color, line_thickness)
        
        # Free throw line
        ft_line_x = baseline_x + flip * int(self.FREE_THROW_LINE * self.ppf)
        cv2.line(court, (ft_line_x, key_top), 
                (ft_line_x, key_top + key_width_px), line_color, line_thickness)
        
        # Free throw circle
        ft_circle_radius = int(self.FREE_THROW_CIRCLE_RADIUS * self.ppf)
        cv2.circle(court, (ft_line_x, self.height_px // 2), 
                  ft_circle_radius, line_color, line_thickness)
        
        # 3-point line (simplified as straight lines and arc)
        three_pt_corner_x = baseline_x + flip * int(self.THREE_POINT_CORNER * self.ppf)
        three_pt_side_x = baseline_x + flip * int(self.THREE_POINT_SIDE * self.ppf)
        
        # Corner 3-point line (vertical from baseline to where arc starts)
        corner_to_sideline = 3 * self.ppf  # 3 feet from sideline
        cv2.line(court, (three_pt_corner_x, int(corner_to_sideline)),
                (three_pt_corner_x, int(corner_to_sideline + 10)), line_color, line_thickness)
        cv2.line(court, (three_pt_corner_x, self.height_px - int(corner_to_sideline)),
                (three_pt_corner_x, self.height_px - int(corner_to_sideline + 10)), 
                line_color, line_thickness)
        
        # 3-point arc (simplified)
        hoop_x = baseline_x + flip * int(self.HOOP_FROM_BASELINE * self.ppf)
        arc_radius = int(self.THREE_POINT_SIDE * self.ppf)
        
        # Draw arc from top to bottom
        if flip == 1:
            start_angle = -90
            end_angle = 90
        else:
            start_angle = 90
            end_angle = 270
        
        cv2.ellipse(court, (hoop_x, self.height_px // 2), 
                   (arc_radius, arc_radius), 0, start_angle, end_angle, 
                   line_color, line_thickness)
        
        # Hoop
        cv2.circle(court, (hoop_x, self.height_px // 2), 3, (255, 100, 0), -1)
    
    def draw_trajectory(self, court: np.ndarray, trajectory: List[Tuple[float, float]],
                       color: Tuple[int, int, int] = (0, 255, 255),
                       thickness: int = 2, show_points: bool = True) -> np.ndarray:
        """
        Draw trajectory on court.
        
        Args:
            court: Court image to draw on
            trajectory: List of (x_feet, y_feet) coordinates
            color: BGR color for trajectory
            thickness: Line thickness
            show_points: Whether to draw points at each position
            
        Returns:
            Court image with trajectory
        """
        if len(trajectory) < 2:
            return court
        
        court_copy = court.copy()
        
        # Convert to pixel coordinates
        points_px = [self.feet_to_pixels(x, y) for x, y in trajectory]
        
        # Draw lines between consecutive points
        for i in range(len(points_px) - 1):
            cv2.line(court_copy, points_px[i], points_px[i + 1], color, thickness)
        
        # Draw points
        if show_points:
            for point in points_px:
                cv2.circle(court_copy, point, 2, color, -1)
        
        # Draw start and end markers
        cv2.circle(court_copy, points_px[0], 5, (0, 255, 0), -1)  # Green start
        cv2.circle(court_copy, points_px[-1], 5, (0, 0, 255), -1)  # Red end
        
        return court_copy
    
    def draw_all_trajectories(self, trajectories_data: Dict, 
                             max_tracks: Optional[int] = None,
                             min_distance: float = 20.0) -> np.ndarray:
        """
        Draw multiple trajectories with different colors.
        
        Args:
            trajectories_data: Dict from trajectories.json
            max_tracks: Maximum number of tracks to draw (None = all)
            min_distance: Minimum total distance (feet) to include track
            
        Returns:
            Court image with all trajectories
        """
        court = self.court_img.copy()
        
        tracks = trajectories_data['tracks']
        
        # Filter tracks by distance
        tracks = [t for t in tracks if t['total_distance_feet'] >= min_distance]
        
        # Sort by distance (draw longest tracks first so they're in background)
        tracks = sorted(tracks, key=lambda t: t['total_distance_feet'], reverse=True)
        
        if max_tracks:
            tracks = tracks[:max_tracks]
        
        print(f"\n📊 Drawing {len(tracks)} trajectories on court...")
        
        # Generate distinct colors
        colors = self._generate_colors(len(tracks))
        
        for i, track in enumerate(tracks):
            trajectory = track['court_trajectory']
            color = colors[i]
            
            # Draw trajectory
            court = self.draw_trajectory(court, trajectory, color, 
                                        thickness=2, show_points=False)
            
            # Add label at end point
            if trajectory:
                end_point = self.feet_to_pixels(trajectory[-1][0], trajectory[-1][1])
                label = f"ID:{track['track_id']}"
                cv2.putText(court, label, (end_point[0] + 5, end_point[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return court
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """Generate n distinct colors in BGR format."""
        colors = []
        for i in range(n):
            hue = i / n
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr)
        return colors
    
    def add_statistics_overlay(self, court: np.ndarray, 
                               trajectories_data: Dict) -> np.ndarray:
        """
        Add statistics overlay to court image.
        
        Args:
            court: Court image
            trajectories_data: Dict from trajectories.json
            
        Returns:
            Court with statistics overlay
        """
        court_copy = court.copy()
        
        tracks = trajectories_data['tracks']
        
        # Calculate statistics
        total_tracks = len(tracks)
        total_distance = sum(t['total_distance_feet'] for t in tracks)
        avg_speed = np.mean([t['avg_speed_mph'] for t in tracks 
                            if t['avg_speed_mph'] > 0])
        max_speed = max([t['max_speed_mph'] for t in tracks])
        
        # Create statistics text
        stats = [
            f"Tracks: {total_tracks}",
            f"Total Distance: {total_distance:.0f} ft",
            f"Avg Speed: {avg_speed:.1f} mph",
            f"Max Speed: {max_speed:.1f} mph"
        ]
        
        # Add semi-transparent background
        overlay = court_copy.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, court_copy, 0.3, 0, court_copy)
        
        # Add text
        y_offset = 30
        for stat in stats:
            cv2.putText(court_copy, stat, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return court_copy


def main():
    """Main function to visualize trajectories on court."""
    parser = argparse.ArgumentParser(description='Visualize trajectories on NBA court')
    parser.add_argument('--trajectories', required=True, help='Path to trajectories.json')
    parser.add_argument('--output', default='court_trajectories.png', help='Output image')
    parser.add_argument('--scale', type=int, default=15, help='Pixels per foot')
    parser.add_argument('--max-tracks', type=int, default=None, help='Max tracks to draw')
    parser.add_argument('--min-distance', type=float, default=20.0, 
                       help='Min distance (ft) to include track')
    parser.add_argument('--stats', action='store_true', help='Add statistics overlay')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("NBA COURT TRAJECTORY VISUALIZATION")
    print(f"{'='*70}\n")
    
    # Load trajectories
    with open(args.trajectories, 'r') as f:
        trajectories_data = json.load(f)
    
    print(f"✅ Loaded {len(trajectories_data['tracks'])} tracks from {args.trajectories}")
    
    # Create court drawer
    drawer = CourtDrawer(pixels_per_foot=args.scale)
    
    # Draw trajectories
    court_with_tracks = drawer.draw_all_trajectories(
        trajectories_data, 
        max_tracks=args.max_tracks,
        min_distance=args.min_distance
    )
    
    # Add statistics if requested
    if args.stats:
        court_with_tracks = drawer.add_statistics_overlay(court_with_tracks, trajectories_data)
    
    # Save output
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), court_with_tracks)
    
    print(f"\n✅ Saved court visualization to: {output_path}")
    print(f"   Image size: {court_with_tracks.shape[1]}×{court_with_tracks.shape[0]} pixels")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
