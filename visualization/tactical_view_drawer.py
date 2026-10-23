"""
Phase 13a: Tactical View (Court-From-Above) Drawer

Provides bird's-eye court visualization for understanding player positioning,
spacing, and defensive sets. Reuses existing homography + projection code.

This module encapsulates the court drawing logic that was previously embedded
in run_video.py's draw_minimap(), making it reusable and modular for:
- Broadcast overlay rendering
- Standalone tactical analysis
- PDF report generation (Product C)
- Real-time coaching display
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Union


def _track_colour(track_id: int) -> Tuple[int, int, int]:
    """Generate consistent color for a track ID."""
    hue = (track_id * 137.5) % 360
    saturation = 0.7
    value = 0.9
    c = colorsys.hsv_to_rgb(hue / 360.0, saturation, value)
    return tuple(int(255 * x) for x in c)


import colorsys


class TacticalViewDrawer:
    """
    Renders bird's-eye court visualization with player and ball positions.
    
    NBA court dimensions: 94 ft × 50 ft
    Supports customizable size, colors, and rendering options.
    """
    
    # NBA court dimensions (feet)
    COURT_LENGTH_FT = 94.0
    COURT_WIDTH_FT = 50.0
    
    def __init__(
        self,
        court_width_px: int = 800,  # Higher resolution (was 500)
        court_height_px: int = 450,  # Higher resolution (was 280)
        team1_color: Tuple[int, int, int] = (0, 0, 255),      # Blue (BGR)
        team2_color: Tuple[int, int, int] = (255, 0, 0),      # Red (BGR)
        bg_color: Tuple[int, int, int] = (30, 30, 30),        # Dark gray
        line_color: Tuple[int, int, int] = (100, 100, 100),   # Gray
        bright_line_color: Tuple[int, int, int] = (0, 140, 220),  # Orange (court perimeter)
        home_team_name: str = "Home",  # Phase 13a: Team 1 label (home team)
        away_team_name: str = "Away",  # Phase 13a: Team 2 label (away team)
    ):
        """
        Initialize tactical view drawer.
        
        Args:
            court_width_px: Width of court in pixels
            court_height_px: Height of court in pixels
            team1_color: BGR color for team 1 players
            team2_color: BGR color for team 2 players
            bg_color: Background color (BGR)
            line_color: Court line color (BGR)
            bright_line_color: Court perimeter color (BGR)
            home_team_name: Label for team 1 (home team)
            away_team_name: Label for team 2 (away team)
        """
        self.court_w_px = court_width_px
        self.court_h_px = court_height_px
        self.team1_color = team1_color
        self.team2_color = team2_color
        self.bg_color = bg_color
        self.line_color = line_color
        self.bright_line_color = bright_line_color
        self.home_team_name = home_team_name  # Phase 13a: Team 1 (home) label
        self.away_team_name = away_team_name  # Phase 13a: Team 2 (away) label
        self.pad = 15
        
        # Precompute scaled court dimensions for faster rendering
        self.total_w = court_width_px + 2 * self.pad
        self.total_h = court_height_px + 2 * self.pad
        
    def draw(
        self,
        H: Optional[np.ndarray],
        players_court: List[Tuple],
        ball_court: Optional[Tuple[float, float]],
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        frame_idx: int = 0,
        show_label: bool = True,
    ) -> np.ndarray:
        """
        Render bird's-eye tactical view of court with players and ball.
        
        Args:
            H: Homography matrix (for rendering camera frustum)
            players_court: List of (x_ft, y_ft, track_id) or (x_ft, y_ft, track_id, team_id)
            ball_court: Ball position in court coords (x_ft, y_ft) or None
            team_colors: Optional dict mapping team_id -> (B, G, R) color tuple
            frame_idx: Frame index (for display only)
            show_label: Whether to show label at bottom
            
        Returns:
            Court image (H, W, 3) in BGR format
        """
        # Create blank court image (ensure C-contiguous for OpenCV)
        court = np.full((self.total_h, self.total_w, 3), self.bg_color, dtype=np.uint8)
        
        # Draw court markings
        self._draw_court_markings(court)
        
        # Draw camera frustum if homography available
        if H is not None:
            self._draw_camera_frustum(court, H)
        
        # Draw players with team colors
        for entry in players_court:
            if len(entry) == 4:
                px_ft, py_ft, track_id, team_id = entry
            else:
                px_ft, py_ft, track_id = entry
                team_id = None
            
            if -5 <= px_ft <= 99 and -5 <= py_ft <= 55:
                px, py = self._ft_to_px(px_ft, py_ft)
                
                # Choose color
                if team_id is not None and team_colors and team_id in team_colors:
                    color = team_colors[team_id]
                else:
                    color = _track_colour(track_id)
                
                # Draw player circle with ID (larger for better visibility)
                cv2.circle(court, (px, py), 10, color, -1)
                cv2.circle(court, (px, py), 10, (255, 255, 255), 2)
                
                # Draw track ID with text shadow for readability
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                text = str(track_id)
                
                # Shadow (offset black text)
                cv2.putText(
                    court, text,
                    (px + 12, py + 6),
                    font, font_scale, (0, 0, 0), thickness + 1
                )
                # Main text in white
                cv2.putText(
                    court, text,
                    (px + 12, py + 6),
                    font, font_scale, (255, 255, 255), thickness
                )
        
        # Draw ball (larger for visibility)
        if ball_court is not None:
            bx_ft, by_ft = ball_court
            if -10 <= bx_ft <= 104 and -10 <= by_ft <= 60:
                bx, by = self._ft_to_px(bx_ft, by_ft)
                cv2.circle(court, (bx, by), 12, (0, 200, 255), -1)
                cv2.circle(court, (bx, by), 12, (255, 255, 255), 3)
        
        # Add label with shadow for readability
        if show_label:
            label = "Tactical View (Phase 13a)"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # Shadow
            cv2.putText(
                court, label,
                (self.pad + 1, self.total_h - 8),
                font, font_scale, (0, 0, 0), thickness + 1
            )
            # Main text
            cv2.putText(
                court, label,
                (self.pad, self.total_h - 10),
                font, font_scale, (180, 180, 180), thickness
            )
        
        # Phase 13a: Draw team labels (HOME and AWAY)
        self._draw_team_labels(court)
        
        return court
    
    def _ft_to_px(self, x_ft: float, y_ft: float) -> Tuple[int, int]:
        """
        Convert court coordinates (feet) to pixel coordinates.
        Flips Y axis: court Y increases upward, image Y increases downward.
        """
        y_flipped = self.COURT_WIDTH_FT - y_ft
        x_px = int(
            self.pad + x_ft / self.COURT_LENGTH_FT * self.court_w_px
        )
        y_px = int(
            self.pad + y_flipped / self.COURT_WIDTH_FT * self.court_h_px
        )
        return (x_px, y_px)
    
    def _draw_court_markings(self, court: np.ndarray) -> None:
        """Draw NBA court lines and markings (bolder for higher resolution)."""
        def fp(x: float, y: float) -> Tuple[int, int]:
            return self._ft_to_px(x, y)
        
        # Court perimeter (thicker line)
        cv2.rectangle(
            court, fp(0, 0), fp(94, 50),
            self.bright_line_color, 3
        )
        
        # Halfcourt line (thicker)
        cv2.line(court, fp(47, 0), fp(47, 50), self.line_color, 2)
        
        # Paint / key areas (both ends, thicker)
        for baseline_x in [0.0, 75.0]:
            cv2.rectangle(
                court, fp(baseline_x, 17), fp(baseline_x + 19, 33),
                self.line_color, 2
            )
        
        # Center circle (thicker)
        center = fp(47, 25)
        radius = int(6 / self.COURT_WIDTH_FT * self.court_h_px)
        cv2.circle(court, center, radius, self.line_color, 2)
    
    def _draw_camera_frustum(self, court: np.ndarray, H: np.ndarray) -> None:
        """
        Draw camera frustum (visible court region from broadcast view).
        Projects broadcast frame corners through homography to show which
        court area is currently visible in the broadcast.
        """
        def fp(x: float, y: float) -> Tuple[int, int]:
            return self._ft_to_px(x, y)
        
        # Project broadcast frame corners (assuming 1920x1080)
        corners = np.array(
            [[0, 0], [1920, 0], [1920, 1080], [0, 1080]],
            dtype=np.float32
        )
        
        # Add homogeneous coordinate
        ch2 = np.hstack([corners, np.ones((4, 1), np.float32)])
        
        # Project through homography
        proj = (H @ ch2.T).T
        w2 = proj[:, 2:3]
        
        # Avoid division by zero
        w2 = np.where(
            np.abs(w2) < 1e-6,
            np.sign(w2 + 1e-9) * 1e-6,
            w2
        )
        
        # Normalize
        cc2 = proj[:, :2] / w2
        
        # Filter valid projections (within extended court bounds)
        pts = [
            fp(float(x), float(y))
            for x, y in cc2
            if -20 < x < 114 and -20 < y < 70
        ]
        
        # Draw frustum polygon if at least 3 points are valid (thicker)
        if len(pts) >= 3:
            cv2.polylines(
                court,
                [np.array(pts, np.int32)],
                True,
                self.bright_line_color,
                3
            )
    
    def _draw_team_labels(self, court: np.ndarray) -> None:
        """
        Phase 13a: Draw HOME and AWAY team labels at top of tactical view.
        Uses team colors for visual distinction.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        y_pos = 25
        
        # HOME label (Team 1) on left side
        home_label = f"HOME: {self.home_team_name}"
        home_size = cv2.getTextSize(home_label, font, font_scale, thickness)[0]
        home_x = self.pad
        
        # Shadow for HOME label
        cv2.putText(court, home_label, (home_x + 1, y_pos + 1), font, font_scale, (0, 0, 0), thickness + 1)
        # Main text with team color
        cv2.putText(court, home_label, (home_x, y_pos), font, font_scale, self.team1_color, thickness)
        
        # AWAY label (Team 2) on right side
        away_label = f"AWAY: {self.away_team_name}"
        away_size = cv2.getTextSize(away_label, font, font_scale, thickness)[0]
        away_x = self.total_w - away_size[0] - self.pad
        
        # Shadow for AWAY label
        cv2.putText(court, away_label, (away_x + 1, y_pos + 1), font, font_scale, (0, 0, 0), thickness + 1)
        # Main text with team color
        cv2.putText(court, away_label, (away_x, y_pos), font, font_scale, self.team2_color, thickness)
    
    def draw_with_frame(
        self,
        broadcast_frame: np.ndarray,
        H: Optional[np.ndarray],
        players_court: List[Tuple],
        ball_court: Optional[Tuple[float, float]],
        team_colors: Optional[Dict[int, Tuple[int, int, int]]] = None,
        position: str = "top-right",
        scale_factor: float = 0.25,
    ) -> np.ndarray:
        """
        Overlay tactical view onto broadcast frame.
        
        Args:
            broadcast_frame: Original broadcast frame
            H: Homography matrix
            players_court: List of player positions
            ball_court: Ball position
            team_colors: Team color mapping
            position: Where to place overlay ("top-right", "bottom-right", "top-left", "bottom-left")
            scale_factor: Size of overlay relative to frame height
            
        Returns:
            Frame with overlay
        """
        # Render tactical view
        tactical = self.draw(H, players_court, ball_court, team_colors)
        
        # Resize to overlay size
        frame_h = broadcast_frame.shape[0]
        overlay_h = int(frame_h * scale_factor)
        aspect_ratio = self.total_w / self.total_h
        overlay_w = int(overlay_h * aspect_ratio)
        
        tactical_resized = cv2.resize(tactical, (overlay_w, overlay_h))
        
        # Determine placement
        result = broadcast_frame.copy()
        frame_w = broadcast_frame.shape[1]
        
        if position == "top-right":
            y, x = 10, frame_w - overlay_w - 10
        elif position == "bottom-right":
            y, x = frame_h - overlay_h - 10, frame_w - overlay_w - 10
        elif position == "top-left":
            y, x = 10, 10
        elif position == "bottom-left":
            y, x = frame_h - overlay_h - 10, 10
        else:
            y, x = 10, frame_w - overlay_w - 10
        
        # Paste overlay
        y1, y2 = max(0, y), min(frame_h, y + overlay_h)
        x1, x2 = max(0, x), min(frame_w, x + overlay_w)
        
        # Handle edge cases where overlay extends past frame
        ty1, tx1 = max(0, -y), max(0, -x)
        ty2 = ty1 + (y2 - y1)
        tx2 = tx1 + (x2 - x1)
        
        result[y1:y2, x1:x2] = tactical_resized[ty1:ty2, tx1:tx2]
        
        return result
