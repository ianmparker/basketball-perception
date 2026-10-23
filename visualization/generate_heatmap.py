"""
Generate heatmap visualization showing player density across court zones.

Creates spatial density maps to show where players spend most of their time
on the court. Useful for analyzing positioning, spacing, and play patterns.

Techniques:
- Gaussian kernel density estimation
- Binned histograms by court zone
- Color gradients (cool → hot) for density
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from scipy.ndimage import gaussian_filter


class HeatmapGenerator:
    """Generate heatmaps showing player position density."""
    
    # NBA court dimensions
    COURT_LENGTH = 94.0
    COURT_WIDTH = 50.0
    
    def __init__(self, pixels_per_foot: int = 15, blur_sigma: float = 2.0):
        """
        Initialize heatmap generator.
        
        Args:
            pixels_per_foot: Scale factor
            blur_sigma: Gaussian blur sigma (higher = smoother heatmap)
        """
        self.ppf = pixels_per_foot
        self.blur_sigma = blur_sigma
        
        self.width_px = int(self.COURT_LENGTH * self.ppf)
        self.height_px = int(self.COURT_WIDTH * self.ppf)
        
        print(f"✅ Heatmap generator initialized: {self.width_px}×{self.height_px} pixels")
        print(f"   Blur sigma: {self.blur_sigma}")
    
    def feet_to_pixels(self, x_feet: float, y_feet: float) -> Tuple[int, int]:
        """Convert court coordinates to pixel coordinates."""
        x_px = int(x_feet * self.ppf)
        y_px = int(y_feet * self.ppf)
        
        # Clamp to image bounds
        x_px = max(0, min(x_px, self.width_px - 1))
        y_px = max(0, min(y_px, self.height_px - 1))
        
        return (x_px, y_px)
    
    def generate_density_map(self, trajectories: List[List[Tuple[float, float]]]) -> np.ndarray:
        """
        Generate 2D density map from trajectories.
        
        Args:
            trajectories: List of trajectories, each is list of (x_feet, y_feet)
            
        Returns:
            2D array (height, width) with density values
        """
        # Create empty density map
        density = np.zeros((self.height_px, self.width_px), dtype=np.float32)
        
        # Accumulate all positions
        total_points = 0
        for trajectory in trajectories:
            for x_feet, y_feet in trajectory:
                x_px, y_px = self.feet_to_pixels(x_feet, y_feet)
                
                # Increment density at this position
                density[y_px, x_px] += 1
                total_points += 1
        
        print(f"   Processed {total_points} position points")
        
        # Apply Gaussian blur for smooth heatmap
        if self.blur_sigma > 0:
            density = gaussian_filter(density, sigma=self.blur_sigma * self.ppf)
        
        # Normalize to 0-1 range
        if density.max() > 0:
            density = density / density.max()
        
        return density
    
    def density_to_heatmap(self, density: np.ndarray, 
                          colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """
        Convert density map to colored heatmap image.
        
        Args:
            density: 2D density array (0-1 range)
            colormap: OpenCV colormap (COLORMAP_JET, COLORMAP_HOT, etc.)
            
        Returns:
            BGR heatmap image
        """
        # Convert to 0-255 range
        density_uint8 = (density * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(density_uint8, colormap)
        
        return heatmap
    
    def overlay_on_court(self, heatmap: np.ndarray, court_img: np.ndarray,
                        alpha: float = 0.6) -> np.ndarray:
        """
        Overlay heatmap on court image with transparency.
        
        Args:
            heatmap: Colored heatmap image
            court_img: Court background image
            alpha: Heatmap opacity (0=transparent, 1=opaque)
            
        Returns:
            Blended image
        """
        # Make low-density areas transparent
        density_mask = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        threshold = 10  # Only show areas with density > threshold
        mask = density_mask > threshold
        
        # Blend heatmap with court
        result = court_img.copy()
        result[mask] = cv2.addWeighted(
            heatmap[mask], alpha,
            court_img[mask], 1 - alpha, 0
        )
        
        return result
    
    def add_zone_statistics(self, density: np.ndarray, court_img: np.ndarray) -> np.ndarray:
        """
        Add statistics for different court zones.
        
        Zones: Paint, Mid-range, 3-point, Halfcourt
        
        Args:
            density: Density map
            court_img: Court image to annotate
            
        Returns:
            Annotated court image
        """
        result = court_img.copy()
        
        # Define zones in feet
        zones = {
            'Paint (0-19 ft)': {'x_range': (0, 19), 'y_range': (17, 33)},
            'Mid-range (19-22 ft)': {'x_range': (19, 22), 'y_range': (0, 50)},
            '3-Point (22+ ft)': {'x_range': (22, 47), 'y_range': (0, 50)},
            'Halfcourt (47+ ft)': {'x_range': (47, 94), 'y_range': (0, 50)}
        }
        
        # Calculate density for each zone
        zone_stats = []
        for zone_name, bounds in zones.items():
            x_min_px, x_max_px = self.feet_to_pixels(bounds['x_range'][0], 0)[0], \
                                  self.feet_to_pixels(bounds['x_range'][1], 0)[0]
            y_min_px, y_max_px = self.feet_to_pixels(0, bounds['y_range'][0])[1], \
                                  self.feet_to_pixels(0, bounds['y_range'][1])[1]
            
            zone_density = density[y_min_px:y_max_px, x_min_px:x_max_px]
            avg_density = zone_density.mean()
            
            zone_stats.append((zone_name, avg_density))
        
        # Sort by density
        zone_stats.sort(key=lambda x: x[1], reverse=True)
        
        # Add text overlay
        overlay = result.copy()
        cv2.rectangle(overlay, (self.width_px - 320, 10), 
                     (self.width_px - 10, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
        
        y_offset = 35
        cv2.putText(result, "Zone Activity:", 
                   (self.width_px - 310, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        for zone_name, density_val in zone_stats:
            # Calculate percentage
            pct = density_val * 100 if zone_stats[0][1] > 0 else 0
            
            text = f"{zone_name.split()[0]}: {pct:.0f}%"
            cv2.putText(result, text,
                       (self.width_px - 310, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        return result
    
    def create_comparison_view(self, density: np.ndarray, court_img: np.ndarray,
                              heatmap: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison of court, density, and overlay.
        
        Args:
            density: Raw density map
            court_img: Court background
            heatmap: Colored heatmap
            
        Returns:
            Combined comparison image
        """
        overlay = self.overlay_on_court(heatmap, court_img, alpha=0.6)
        
        # Add labels
        labeled_court = self._add_label(court_img.copy(), "Court Template")
        labeled_heatmap = self._add_label(heatmap.copy(), "Density Heatmap")
        labeled_overlay = self._add_label(overlay.copy(), "Court + Heatmap")
        
        # Stack horizontally
        comparison = np.hstack([labeled_court, labeled_heatmap, labeled_overlay])
        
        return comparison
    
    def _add_label(self, img: np.ndarray, label: str) -> np.ndarray:
        """Add label text to top of image."""
        result = img.copy()
        cv2.putText(result, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return result


def draw_court_simple(width_px: int, height_px: int, ppf: int) -> np.ndarray:
    """
    Draw simplified court for heatmap background.
    
    Args:
        width_px: Width in pixels
        height_px: Height in pixels
        ppf: Pixels per foot
        
    Returns:
        Court image (BGR)
    """
    # Import here to avoid circular dependency
    import sys
    sys.path.append(str(Path(__file__).parent))
    from draw_court import CourtDrawer
    
    drawer = CourtDrawer(pixels_per_foot=ppf, background_color=(20, 50, 20))
    return drawer.court_img


def main():
    """Main function to generate heatmap visualization."""
    parser = argparse.ArgumentParser(description='Generate player density heatmap')
    parser.add_argument('--trajectories', required=True, help='Path to trajectories.json')
    parser.add_argument('--output', default='court_heatmap.png', help='Output image')
    parser.add_argument('--scale', type=int, default=15, help='Pixels per foot')
    parser.add_argument('--blur', type=float, default=2.0, help='Blur sigma (smoothness)')
    parser.add_argument('--colormap', default='jet', 
                       choices=['jet', 'hot', 'rainbow', 'cool', 'winter'],
                       help='Color scheme')
    parser.add_argument('--comparison', action='store_true', 
                       help='Generate side-by-side comparison')
    parser.add_argument('--zones', action='store_true', help='Show zone statistics')
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print("PLAYER DENSITY HEATMAP GENERATION")
    print(f"{'='*70}\n")
    
    # Load trajectories
    with open(args.trajectories, 'r') as f:
        trajectories_data = json.load(f)
    
    print(f"✅ Loaded {len(trajectories_data['tracks'])} tracks")
    
    # Extract all trajectories
    all_trajectories = [track['court_trajectory'] for track in trajectories_data['tracks']]
    
    # Generate heatmap
    generator = HeatmapGenerator(pixels_per_foot=args.scale, blur_sigma=args.blur)
    
    print(f"\n📊 Generating density map...")
    density = generator.generate_density_map(all_trajectories)
    
    # Convert colormap name to OpenCV constant
    colormap_dict = {
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'cool': cv2.COLORMAP_COOL,
        'winter': cv2.COLORMAP_WINTER
    }
    colormap = colormap_dict[args.colormap]
    
    print(f"📊 Applying {args.colormap} colormap...")
    heatmap = generator.density_to_heatmap(density, colormap=colormap)
    
    # Draw court
    print(f"📊 Drawing court template...")
    court_img = draw_court_simple(generator.width_px, generator.height_px, args.scale)
    
    # Create visualization
    if args.comparison:
        print(f"📊 Creating comparison view...")
        result = generator.create_comparison_view(density, court_img, heatmap)
    else:
        print(f"📊 Overlaying heatmap on court...")
        result = generator.overlay_on_court(heatmap, court_img, alpha=0.6)
        
        if args.zones:
            print(f"📊 Adding zone statistics...")
            result = generator.add_zone_statistics(density, result)
    
    # Save output
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), result)
    
    print(f"\n✅ Saved heatmap to: {output_path}")
    print(f"   Image size: {result.shape[1]}×{result.shape[0]} pixels")
    
    # Print density statistics
    print(f"\n📈 Density Statistics:")
    print(f"   Max density: {density.max():.3f}")
    print(f"   Mean density: {density.mean():.3f}")
    print(f"   Non-zero pixels: {(density > 0.01).sum()} / {density.size}")
    
    print(f"\n{'='*70}\n")


if __name__ == '__main__':
    main()
