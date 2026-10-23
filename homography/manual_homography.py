"""
Manual homography calibration tool for basketball court.

User clicks 4+ court corners (baseline, sideline intersections) on video frame,
then the tool computes homography matrix H that maps pixels → real court coordinates.

Mathematical Foundation (see docs/MATHEMATICAL_FOUNDATIONS.md):
- Homography H is 3×3 matrix: p' = H × p
- Computed using Direct Linear Transform (DLT) with SVD
- Minimum 4 point correspondences required
- More points = more robust (8-12 recommended)
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse


class CourtHomographyTool:
    """Interactive tool for manual court homography calibration."""
    
    # NBA court dimensions in feet
    COURT_LENGTH = 94.0  # feet (baseline to baseline)
    COURT_WIDTH = 50.0   # feet (sideline to sideline)
    
    # Standard court feature locations (in feet from origin at baseline corner)
    # Coordinate system: (distance_along_court, distance_across_width)
    # Origin at baseline corner, measures along length (94 ft) and width (50 ft)
    # "Near/Far" = closer/farther from camera (typically sideline dimension)
    
    # NBA Official Dimensions:
    # Court: 94' x 50'
    # Key: 19' long x 16' wide
    # Free throw line: 15' from backboard = 19' from baseline (backboard is 4' from baseline)
    # 3-point line: 23'9" (23.75 ft) at top, 22' in corners, straight line starts 14' from baseline
    # 3-point line from sideline: 3' 
    # Center circle radius: 6'
    # Restricted area: 4' radius
    # Timeout hash marks: 28' from baseline
    
    COURT_FEATURES = {
        # Court corners (baseline to baseline, sideline to sideline)
        'baseline_corner_near': (0, 0),              # Baseline corner, near sideline
        'baseline_corner_far': (0, COURT_WIDTH),     # Baseline corner, far sideline
        
        # Halfcourt line intersections
        'halfcourt_near': (COURT_LENGTH/2, 0),     # Halfcourt at near sideline (47ft, 0ft)
        'halfcourt_far': (COURT_LENGTH/2, COURT_WIDTH),  # Halfcourt at far sideline (47ft, 50ft)
        
        # Timeout hash marks (28ft from baseline on sidelines)
        'hash_mark_near': (28, 0),       # Near sideline hash mark
        'hash_mark_far': (28, COURT_WIDTH),  # Far sideline hash mark
        
        # Free throw line corners (19ft from baseline, key is 16ft wide, centered at 25ft)
        # Key spans from (50-16)/2 = 17 ft to 17+16 = 33 ft
        'free_throw_corner_near': (19, 17),     # Free throw line, near key edge
        'free_throw_corner_far': (19, 33),      # Free throw line, far key edge
        
        # 3-point arc key points
        # Top of arc: 23.75 ft from basket center = 27.75 ft from baseline (basket is 4 ft from baseline)
        # Wings: 14 ft from baseline where arc meets straight line, 3 ft from sideline
        'three_point_top': (27.75, 25),       # Top of 3-pt arc (center, 23'9" from basket center)
        'three_point_wing_near': (14, 3),     # Near wing (14' from baseline, 3' from sideline)
        'three_point_wing_far': (14, 47),     # Far wing (14' from baseline, 3' from sideline)
    }
    
    def __init__(self, video_path, start_frame=30):
        """Initialize homography tool with video."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(str(video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Skip to specified start frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frame at start_frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {start_frame} from video")
        
        self.frame_height, self.frame_width = self.frame.shape[:2]
        self.display_frame = self.frame.copy()
        
        # Storage for clicked points
        self.image_points = []  # Points in pixel coordinates
        self.court_points = []  # Corresponding points in feet
        self.point_labels = []  # Labels for each point
        
        # Draw reference guide on display frame
        self.draw_reference_guide()
        
        self.homography_matrix = None
        self.window_name = 'Court Homography - Click court corners'
    
    def draw_reference_guide(self):
        """Draw reference lines and guide on the frame to help with keypoint selection."""
        # Create overlay for visualization
        overlay = self.display_frame.copy()
        
        # Draw court reference lines if we have some points
        # This helps visualize where lines are as user clicks
        if len(self.image_points) >= 2:
            # Draw baseline if we have 2+ points
            cv2.line(overlay, tuple(self.image_points[0]), 
                    tuple(self.image_points[1]), (0, 255, 255), 2)
            cv2.putText(overlay, "Baseline", 
                       (self.image_points[0][0] + 10, self.image_points[0][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        if len(self.image_points) >= 4:
            # Draw free throw line if we have 4+ points
            cv2.line(overlay, tuple(self.image_points[2]), 
                    tuple(self.image_points[3]), (255, 0, 255), 2)
            cv2.putText(overlay, "Free Throw Line (19 ft)", 
                       (self.image_points[2][0] + 10, self.image_points[2][1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Text instructions overlay (semi-transparent background)
        # Add semi-transparent black box for text readability
        text_bg = np.zeros((250, 500, 3), dtype=np.uint8)
        
        # Text configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        text_color = (0, 255, 255)  # Yellow
        line_height = 25
        
        # Title
        y_pos = 30
        cv2.putText(text_bg, "RECOMMENDED KEYPOINTS:", 
                   (10, y_pos), font, 0.6, (0, 255, 0), 2)
        y_pos += line_height + 5
        
        # Key reference points to select
        reference_points = [
            "TYPICAL BROADCAST VIEW POINTS:",
            "1-2: Baseline endpoints",
            "3-4: Free throw lane corners (19 ft)",
            "5-6: Lane hash marks (28 ft)",
            "7: Top of 3-point arc (5.25 ft)",
            "8-9: 3-point wings (14 ft, ~12ft from center)",
            "",
            "Optional if visible:",
            "- Halfcourt line intersections",
            "- Opposite baseline points"
        ]
        
        for text in reference_points:
            cv2.putText(text_bg, text, (10, y_pos), font, font_scale, 
                       text_color, font_thickness)
            y_pos += line_height
        
        # Court dimensions
        y_pos += 10
        cv2.putText(text_bg, "Court Dimensions:", (10, y_pos), 
                   font, 0.5, (255, 255, 0), 2)
        y_pos += line_height
        
        dimensions = [
            "Court: 94 ft x 50 ft",
            "Free throw line: 19 ft from baseline",
            "Key: 19 ft x 16 ft (centered)",
            "3-pt: 23'9\" top, 22' corners",
            "Hash marks: 28 ft from baseline"
        ]
        
        for text in dimensions:
            cv2.putText(text_bg, text, (10, y_pos), font, 0.4, 
                       (200, 200, 200), font_thickness)
            y_pos += line_height - 5
        
        # Overlay text box on frame (top-left corner)
        h, w = text_bg.shape[:2]
        overlay[0:h, 0:w] = cv2.addWeighted(text_bg, 0.7, overlay[0:h, 0:w], 0.3, 0)
        
        self.display_frame = overlay
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select court points."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.image_points.append([x, y])
            
            # Redraw display frame from scratch
            self.display_frame = self.frame.copy()
            self.draw_reference_guide()
            
            # Draw all points
            for i, pt in enumerate(self.image_points):
                cv2.circle(self.display_frame, tuple(pt), 5, (0, 255, 0), -1)
                cv2.putText(self.display_frame, str(i + 1), 
                           (pt[0] + 10, pt[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, self.display_frame)
            
            print(f"\n✅ Point {len(self.image_points)} added at: ({x}, {y})")
            print(f"   Total points: {len(self.image_points)}")
            
            if len(self.image_points) >= 4:
                print("\n💡 You have enough points (≥4). Press 'c' to compute homography.")
                print("   Or continue adding more points for better accuracy (8-12 recommended).")
    
    def display_instructions(self):
        """Display usage instructions."""
        print("\n" + "="*70)
        print("MANUAL HOMOGRAPHY CALIBRATION TOOL")
        print("="*70)
        print("\nINSTRUCTIONS:")
        print("1. Click on visible court corners/intersections in the frame")
        print("2. Minimum 4 points required (more is better, 8-12 recommended)")
        print("3. Reference guide shown on frame with recommended keypoints")
        print("\nRECOMMENDED POINTS (typical broadcast view):")
        print("   • Baseline endpoints - 2 points")
        print("   • Free throw lane corners (19 ft) - 2 points")
        print("   • Lane hash marks (28 ft) - 2 points")
        print("   • Top of 3-point arc (5.25 ft from baseline)")
        print("   • 3-point wings (14 ft, left & right)")
        print("   • Halfcourt line (if visible)")
        print("\nTIPS:")
        print("   • Use 3-point ARC TOP and WINGS (easier to see than corners)")
        print("   • Use lane hash marks if halfcourt not visible")
        print("   • Well-distributed points = better accuracy")
        print("\nCONTROLS:")
        print("   • LEFT CLICK: Add point")
        print("   • 'u': Undo last point")
        print("   • 'r': Reset all points")
        print("   • 'c': Compute homography (after ≥4 points)")
        print("   • 'q': Quit without saving")
        print("="*70 + "\n")
    
    def select_court_correspondences(self):
        """Interactive selection of point correspondences."""
        self.display_instructions()
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)  # Start with smaller window (720p)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, self.display_frame)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('u'):  # Undo last point
                if self.image_points:
                    self.image_points.pop()
                    
                    # Redraw from scratch
                    self.display_frame = self.frame.copy()
                    self.draw_reference_guide()
                    
                    # Redraw remaining points
                    for i, pt in enumerate(self.image_points):
                        cv2.circle(self.display_frame, tuple(pt), 5, (0, 255, 0), -1)
                        cv2.putText(self.display_frame, str(i+1), 
                                   (pt[0] + 10, pt[1] + 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    cv2.imshow(self.window_name, self.display_frame)
                    print(f"❌ Removed last point. Total: {len(self.image_points)}")
            
            elif key == ord('r'):  # Reset all points
                self.image_points = []
                self.display_frame = self.frame.copy()
                self.draw_reference_guide()
                cv2.imshow(self.window_name, self.display_frame)
                print("🔄 Reset all points.")
            
            elif key == ord('c'):  # Compute homography
                if len(self.image_points) >= 4:
                    self.map_to_court_coordinates()
                    break
                else:
                    print(f"⚠️  Need at least 4 points. Current: {len(self.image_points)}")
            
            elif key == ord('q'):  # Quit
                print("\n❌ Cancelled by user.")
                cv2.destroyAllWindows()
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def map_to_court_coordinates(self):
        """Map selected image points to court coordinates."""
        print("\n" + "="*70)
        print("MAPPING IMAGE POINTS TO COURT COORDINATES")
        print("="*70)
        
        # Display available court features
        print("\nAVAILABLE COURT FEATURES:")
        for i, (name, (x, y)) in enumerate(self.COURT_FEATURES.items(), 1):
            print(f"  {i:2d}. {name:30s} → ({x:5.1f}, {y:5.1f}) feet")
        
        print("\nFor each image point, enter the corresponding court feature number:")
        print("(Or enter custom coordinates as 'x,y' in feet)\n")
        
        self.court_points = []
        self.point_labels = []
        
        for i, img_pt in enumerate(self.image_points):
            print(f"\nPoint {i+1} at pixel ({img_pt[0]}, {img_pt[1]}):")
            
            while True:
                response = input(f"  Enter feature number (1-{len(self.COURT_FEATURES)}) or 'x,y': ").strip()
                
                try:
                    # Check if custom coordinates
                    if ',' in response:
                        x, y = map(float, response.split(','))
                        self.court_points.append([x, y])
                        self.point_labels.append(f"Custom ({x:.1f}, {y:.1f})")
                        print(f"  ✅ Mapped to court: ({x:.1f}, {y:.1f}) feet")
                        break
                    else:
                        # Feature number
                        idx = int(response) - 1
                        if 0 <= idx < len(self.COURT_FEATURES):
                            feature_name = list(self.COURT_FEATURES.keys())[idx]
                            x, y = self.COURT_FEATURES[feature_name]
                            self.court_points.append([x, y])
                            self.point_labels.append(feature_name)
                            print(f"  ✅ Mapped to: {feature_name} → ({x:.1f}, {y:.1f}) feet")
                            break
                        else:
                            print(f"  ⚠️  Invalid number. Enter 1-{len(self.COURT_FEATURES)}")
                except (ValueError, IndexError):
                    print("  ⚠️  Invalid input. Try again.")
        
        print("\n" + "="*70)
    
    def compute_homography(self):
        """
        Compute homography matrix using Direct Linear Transform (DLT).
        
        Mathematical formula (see docs/MATHEMATICAL_FOUNDATIONS.md):
        
        Given N point correspondences (pᵢ, p'ᵢ), solve:
            p'ᵢ = H × pᵢ
        
        where H is 3×3 homography matrix computed via SVD of system:
            A × h = 0
        
        OpenCV's cv2.findHomography() implements DLT with RANSAC for robustness.
        """
        print("\n" + "="*70)
        print("COMPUTING HOMOGRAPHY MATRIX")
        print("="*70)
        
        # Convert to numpy arrays
        src_pts = np.array(self.image_points, dtype=np.float32)
        dst_pts = np.array(self.court_points, dtype=np.float32)
        
        print(f"\nSource points (pixels): {len(src_pts)}")
        print(f"Destination points (feet): {len(dst_pts)}")
        
        # Compute homography using DLT + RANSAC
        # RANSAC helps reject outliers if user clicked incorrectly
        self.homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if self.homography_matrix is None:
            print("\n❌ ERROR: Could not compute homography matrix.")
            print("   Check that points are not collinear and correspond correctly.")
            return False
        
        print("\n✅ Homography matrix computed successfully!")
        print("\nH =")
        print(self.homography_matrix)
        
        # Compute reprojection error
        self.validate_homography(src_pts, dst_pts, mask)
        
        return True
    
    def validate_homography(self, src_pts, dst_pts, mask):
        """
        Validate homography by computing reprojection error.
        
        Reprojection error formula:
            errorᵢ = ||p'ᵢ - H × pᵢ||₂
            RMS = √(Σᵢ errorᵢ² / N)
        
        Good homography: RMS < 5 pixels for manual selection
        """
        print("\n" + "-"*70)
        print("VALIDATION - REPROJECTION ERROR")
        print("-"*70)
        
        # Transform source points using homography
        src_pts_reshaped = src_pts.reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(src_pts_reshaped, self.homography_matrix)
        transformed_pts = transformed_pts.reshape(-1, 2)
        
        # Calculate errors
        errors = np.linalg.norm(transformed_pts - dst_pts, axis=1)
        rms_error = np.sqrt(np.mean(errors**2))
        max_error = np.max(errors)
        
        print(f"\nReprojection errors (feet):")
        for i, (err, used) in enumerate(zip(errors, mask)):
            status = "✅ inlier" if used else "❌ outlier"
            print(f"  Point {i+1}: {err:.3f} feet ({status})")
        
        print(f"\nRMS Error: {rms_error:.3f} feet")
        print(f"Max Error: {max_error:.3f} feet")
        
        if rms_error < 2.0:
            print("✅ Excellent homography quality (RMS < 2 feet)")
        elif rms_error < 5.0:
            print("✅ Good homography quality (RMS < 5 feet)")
        else:
            print("⚠️  Moderate quality. Consider re-selecting points more carefully.")
        
        # Visualize transformation on court diagram
        self.visualize_court_overlay()
    
    def visualize_court_overlay(self):
        """Overlay court template on video frame using homography."""
        # Define court outline in real coordinates (feet)
        court_outline = np.array([
            [0, 0],                      # Baseline near left
            [0, self.COURT_WIDTH],       # Baseline near right
            [self.COURT_LENGTH, self.COURT_WIDTH],  # Baseline far right
            [self.COURT_LENGTH, 0],      # Baseline far left
            [0, 0]                       # Close loop
        ], dtype=np.float32)
        
        # Halfcourt line
        halfcourt_line = np.array([
            [self.COURT_LENGTH/2, 0],
            [self.COURT_LENGTH/2, self.COURT_WIDTH]
        ], dtype=np.float32)
        
        # Free throw circle (19 ft from baseline, 6 ft radius)
        ft_center_x = 19
        ft_center_y = 25
        ft_radius = 6
        num_points = 100
        angles = np.linspace(0, 2*np.pi, num_points)  # Complete circle
        free_throw_circle = np.array([
            [ft_center_x + ft_radius * np.cos(angle), ft_center_y + ft_radius * np.sin(angle)]
            for angle in angles
        ], dtype=np.float32)
        
        # 3-point arc (basket at 4 ft from baseline, 23.75 ft radius)
        # Arc connects the two corner points at (14, 3) and (14, 47)
        basket_x = 4
        basket_y = 25
        three_pt_radius = 23.75
        
        # Calculate angles where arc meets straight lines at (14, 3) and (14, 47)
        angle_start = np.arctan2(3 - basket_y, 14 - basket_x)  # Left corner
        angle_end = np.arctan2(47 - basket_y, 14 - basket_x)    # Right corner
        
        arc_angles = np.linspace(angle_start, angle_end, 100)
        three_point_arc = np.array([
            [basket_x + three_pt_radius * np.cos(angle), basket_y + three_pt_radius * np.sin(angle)]
            for angle in arc_angles
        ], dtype=np.float32)
        
        # 3-point straight lines (14 ft from baseline, 3 ft from sideline)
        three_pt_left_line = np.array([
            [0, 3],
            [14, 3]
        ], dtype=np.float32)
        
        three_pt_right_line = np.array([
            [0, 47],
            [14, 47]
        ], dtype=np.float32)
        
        # Key/paint (19 ft long, 16 ft wide, centered)
        key_outline = np.array([
            [0, 17],
            [19, 17],
            [19, 33],
            [0, 33],
            [0, 17]
        ], dtype=np.float32)
        
        # Hash marks (28 ft from baseline on sidelines, small perpendicular marks)
        hash_mark_length = 3  # 3 ft long marks
        hash_mark_near = np.array([
            [28, 0],
            [28, hash_mark_length]
        ], dtype=np.float32)
        
        hash_mark_far = np.array([
            [28, self.COURT_WIDTH],
            [28, self.COURT_WIDTH - hash_mark_length]
        ], dtype=np.float32)
        
        # Transform all court features to pixel coordinates
        H_inv = np.linalg.inv(self.homography_matrix)
        
        def transform_points(pts):
            pts_reshaped = pts.reshape(-1, 1, 2)
            pixels = cv2.perspectiveTransform(pts_reshaped, H_inv)
            return pixels.reshape(-1, 2).astype(np.int32)
        
        court_pixels = transform_points(court_outline)
        halfcourt_pixels = transform_points(halfcourt_line)
        ft_circle_pixels = transform_points(free_throw_circle)
        three_pt_arc_pixels = transform_points(three_point_arc)
        three_pt_left_pixels = transform_points(three_pt_left_line)
        three_pt_right_pixels = transform_points(three_pt_right_line)
        key_pixels = transform_points(key_outline)
        hash_mark_near_pixels = transform_points(hash_mark_near)
        hash_mark_far_pixels = transform_points(hash_mark_far)
        
        # Draw court overlay
        overlay = self.frame.copy()
        
        # Court outline (yellow)
        cv2.polylines(overlay, [court_pixels], isClosed=True, 
                     color=(0, 255, 255), thickness=3)
        
        # Halfcourt line (yellow)
        cv2.polylines(overlay, [halfcourt_pixels], isClosed=False,
                     color=(0, 255, 255), thickness=2)
        
        # Key/paint (white)
        cv2.polylines(overlay, [key_pixels], isClosed=True,
                     color=(255, 255, 255), thickness=2)
        
        # Free throw circle (cyan)
        cv2.polylines(overlay, [ft_circle_pixels], isClosed=False,
                     color=(255, 255, 0), thickness=2)
        
        # 3-point line (green)
        cv2.polylines(overlay, [three_pt_arc_pixels], isClosed=False,
                     color=(0, 255, 0), thickness=2)
        cv2.polylines(overlay, [three_pt_left_pixels], isClosed=False,
                     color=(0, 255, 0), thickness=2)
        cv2.polylines(overlay, [three_pt_right_pixels], isClosed=False,
                     color=(0, 255, 0), thickness=2)
        
        # Hash marks (orange)
        cv2.polylines(overlay, [hash_mark_near_pixels], isClosed=False,
                     color=(0, 165, 255), thickness=3)
        cv2.polylines(overlay, [hash_mark_far_pixels], isClosed=False,
                     color=(0, 165, 255), thickness=3)
        
        # Draw selected points
        for i, pt in enumerate(self.image_points):
            cv2.circle(overlay, tuple(pt), 8, (0, 0, 255), -1)
            cv2.putText(overlay, str(i+1), (pt[0] + 12, pt[1] + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show overlay
        cv2.namedWindow('Court Overlay Validation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Court Overlay Validation', 1280, 720)  # Start with smaller window
        cv2.imshow('Court Overlay Validation', overlay)
        print("\n📊 Court overlay displayed. Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_homography(self, output_path):
        """Save homography matrix and metadata to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            'video_path': str(self.video_path),
            'frame_width': int(self.frame_width),
            'frame_height': int(self.frame_height),
            'court_dimensions': {
                'length_feet': float(self.COURT_LENGTH),
                'width_feet': float(self.COURT_WIDTH)
            },
            'num_points': len(self.image_points),
            'image_points': [[float(x), float(y)] for x, y in self.image_points],
            'court_points': [[float(x), float(y)] for x, y in self.court_points],
            'point_labels': self.point_labels,
            'homography_matrix': self.homography_matrix.tolist(),
            'homography_matrix_inverse': np.linalg.inv(self.homography_matrix).tolist()
        }
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Homography saved to: {output_path}")
        return output_path
    
    def run(self, output_path='homography_matrix.json'):
        """Run the complete homography calibration workflow."""
        try:
            # Step 1: Select points
            if not self.select_court_correspondences():
                return None
            
            # Step 2: Compute homography
            if not self.compute_homography():
                return None
            
            # Step 3: Save result
            saved_path = self.save_homography(output_path)
            
            print("\n" + "="*70)
            print("✅ HOMOGRAPHY CALIBRATION COMPLETE")
            print("="*70)
            print(f"\nHomography matrix saved to: {saved_path}")
            print("\nYou can now use this matrix to transform pixel coordinates")
            print("to real court coordinates for trajectory analysis.")
            print("="*70 + "\n")
            
            return saved_path
            
        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    """Command-line interface for manual homography tool."""
    parser = argparse.ArgumentParser(
        description='Manual homography calibration tool for basketball court',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python homography/manual_homography.py --video data/sample/videos/Luka_Bron_PNR.mp4
  
This will:
  1. Show first frame of video
  2. Let you click court corners/intersections
  3. Map clicked points to real court coordinates (feet)
  4. Compute and save homography matrix
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default='homography_matrix.json',
                       help='Output path for homography matrix JSON (default: homography_matrix.json)')
    parser.add_argument('--start-frame', type=int, default=30,
                       help='Frame number to use for calibration (default: 30)')
    
    args = parser.parse_args()
    
    # Check video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ ERROR: Video file not found: {video_path}")
        return
    
    # Run tool
    tool = CourtHomographyTool(video_path, start_frame=args.start_frame)
    tool.run(args.output)


if __name__ == '__main__':
    main()
