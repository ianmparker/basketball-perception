"""
Generate NBA court diagram with calibration keypoints
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Arc, Circle, Rectangle, FancyBboxPatch
import numpy as np

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(12, 20))

# Court dimensions (in feet)
COURT_LENGTH = 94
COURT_WIDTH = 50

# Set up the plot
ax.set_xlim(-2, COURT_WIDTH + 2)
ax.set_ylim(-2, COURT_LENGTH + 2)
ax.set_aspect('equal')
ax.set_facecolor('#d2a679')  # Wood court color

# Court outline
court = Rectangle((0, 0), COURT_WIDTH, COURT_LENGTH, 
                   linewidth=3, edgecolor='white', facecolor='none')
ax.add_patch(court)

# Sideline labels
ax.text(-1, COURT_LENGTH/2, 'SIDELINE', rotation=90, va='center', ha='right', 
        fontsize=11, weight='bold', color='white', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
ax.text(COURT_WIDTH + 1, COURT_LENGTH/2, 'SIDELINE', rotation=270, va='center', ha='right', 
        fontsize=11, weight='bold', color='white', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

# Baseline labels
ax.text(COURT_WIDTH/2, -0.5, 'BASELINE', ha='center', va='top', 
        fontsize=11, weight='bold', color='white',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))

# Halfcourt line
ax.plot([0, COURT_WIDTH], [47, 47], 'white', linewidth=2)

# Left basket area (near baseline)
# Backboard (6 ft wide, 4 ft from baseline)
backboard_width = 6
backboard_y = 4
ax.plot([25 - backboard_width/2, 25 + backboard_width/2], [backboard_y, backboard_y], 
        'orange', linewidth=4, solid_capstyle='round', zorder=5)

# Paint/Key (19 ft x 16 ft, centered)
key_left = 25 - 8  # Center at 25, width 16
key_right = 25 + 8
paint = Rectangle((key_left, 0), 16, 19, 
                   linewidth=2, edgecolor='white', facecolor='none')
ax.add_patch(paint)

# Free throw circle
ft_circle = Arc((25, 19), 12, 12, theta1=0, theta2=180,
                linewidth=2, edgecolor='white', fill=False)
ax.add_patch(ft_circle)

# Free throw circle (bottom half - dashed)
ft_circle_dash = Arc((25, 19), 12, 12, theta1=180, theta2=360,
                      linewidth=2, edgecolor='white', fill=False, linestyle='--')
ax.add_patch(ft_circle_dash)

# 3-point line (left side)
# Arc radius: 23.75 ft from basket (which is 4 ft from baseline)
basket_y = 4
three_pt_radius = 23.75
three_pt_arc = Arc((25, basket_y), three_pt_radius*2, three_pt_radius*2, 
                    theta1=22, theta2=158, linewidth=2, edgecolor='white', fill=False)
ax.add_patch(three_pt_arc)

# 3-point straight lines (14 ft from baseline, 3 ft from sideline)
ax.plot([3, 3], [0, 14], 'white', linewidth=2)
ax.plot([47, 47], [0, 14], 'white', linewidth=2)

# Restricted area (4 ft radius arc)
restricted = Arc((25, basket_y), 8, 8, theta1=0, theta2=180,
                 linewidth=2, edgecolor='white', fill=False)
ax.add_patch(restricted)

# Hash marks on sidelines at 28 ft
hash_mark_length = 0.5
ax.plot([0, hash_mark_length], [28, 28], 'white', linewidth=2)
ax.plot([COURT_WIDTH - hash_mark_length, COURT_WIDTH], [28, 28], 'white', linewidth=2)

# Center circle
center_circle = Circle((25, 47), 6, linewidth=2, edgecolor='white', fill=False)
ax.add_patch(center_circle)

# Mirror for right basket (far baseline)
# Backboard
ax.plot([25 - backboard_width/2, 25 + backboard_width/2], 
        [COURT_LENGTH - backboard_y, COURT_LENGTH - backboard_y], 
        'orange', linewidth=4, solid_capstyle='round', zorder=5)

# Paint/Key
paint2 = Rectangle((key_left, COURT_LENGTH - 19), 16, 19, 
                    linewidth=2, edgecolor='white', facecolor='none')
ax.add_patch(paint2)

# Free throw circle
ft_circle2 = Arc((25, COURT_LENGTH - 19), 12, 12, theta1=180, theta2=360,
                 linewidth=2, edgecolor='white', fill=False)
ax.add_patch(ft_circle2)
ft_circle2_dash = Arc((25, COURT_LENGTH - 19), 12, 12, theta1=0, theta2=180,
                       linewidth=2, edgecolor='white', fill=False, linestyle='--')
ax.add_patch(ft_circle2_dash)

# 3-point line (right side)
three_pt_arc2 = Arc((25, COURT_LENGTH - basket_y), three_pt_radius*2, three_pt_radius*2, 
                     theta1=202, theta2=338, linewidth=2, edgecolor='white', fill=False)
ax.add_patch(three_pt_arc2)
ax.plot([3, 3], [COURT_LENGTH - 14, COURT_LENGTH], 'white', linewidth=2)
ax.plot([47, 47], [COURT_LENGTH - 14, COURT_LENGTH], 'white', linewidth=2)

# Restricted area
restricted2 = Arc((25, COURT_LENGTH - basket_y), 8, 8, theta1=180, theta2=360,
                  linewidth=2, edgecolor='white', fill=False)
ax.add_patch(restricted2)

# Hash marks
ax.plot([0, hash_mark_length], [COURT_LENGTH - 28, COURT_LENGTH - 28], 'white', linewidth=2)
ax.plot([COURT_WIDTH - hash_mark_length, COURT_WIDTH], [COURT_LENGTH - 28, COURT_LENGTH - 28], 'white', linewidth=2)

# ============= CALIBRATION KEYPOINTS =============

# Primary keypoints (green)
primary_points = {
    'baseline_corner_near': (0, 0),
    'baseline_corner_far': (50, 0),
    'free_throw_corner_near': (17, 19),
    'free_throw_corner_far': (33, 19),
    'three_point_top': (25, 27.75),  # 23'9" (23.75 ft) from basket + 4 ft basket offset = 27.75 ft from baseline
}

# Secondary keypoints (yellow)
secondary_points = {
    'hash_mark_near': (0, 28),
    'hash_mark_far': (50, 28),
    'three_point_wing_near': (3, 14),
    'three_point_wing_far': (47, 14),
}

# Optional keypoints (blue)
optional_points = {
    'halfcourt_near': (0, 47),
    'halfcourt_far': (50, 47),
}

# Plot primary keypoints
for name, (x, y) in primary_points.items():
    ax.plot(x, y, 'o', color='#00ff00', markersize=12, markeredgecolor='black', 
            markeredgewidth=1.5, zorder=10)
    # Add label
    # Special positioning for free throw corners to avoid overlap
    if 'free_throw_corner_near' in name:
        offset_x = -2
        offset_y = -1.5
        ha = 'right'
        va = 'top'
    elif 'free_throw_corner_far' in name:
        offset_x = 2
        offset_y = -1.5
        ha = 'left'
        va = 'top'
    else:
        offset_x = 2 if x < 25 else -2
        offset_y = 1.5
        ha = 'left' if x < 25 else 'right'
        va = 'bottom'
    
    ax.text(x + offset_x, y + offset_y, name.replace('_', ' ').title(), 
            fontsize=8, ha=ha, va=va, color='white', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7, edgecolor='none'))

# Plot secondary keypoints
for name, (x, y) in secondary_points.items():
    ax.plot(x, y, 's', color='#ffff00', markersize=10, markeredgecolor='black', 
            markeredgewidth=1.5, zorder=10)
    offset_x = 2 if x < 25 else -2
    offset_y = 1.5
    ha = 'left' if x < 25 else 'right'
    ax.text(x + offset_x, y + offset_y, name.replace('_', ' ').title(), 
            fontsize=8, ha=ha, va='bottom', color='black', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7, edgecolor='none'))

# Plot optional keypoints
for name, (x, y) in optional_points.items():
    ax.plot(x, y, 'd', color='#00bfff', markersize=10, markeredgecolor='black', 
            markeredgewidth=1.5, zorder=10)
    offset_x = 2 if x < 25 else -2
    offset_y = 1.5
    ha = 'left' if x < 25 else 'right'
    ax.text(x + offset_x, y + offset_y, name.replace('_', ' ').title(), 
            fontsize=8, ha=ha, va='bottom', color='black', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='cyan', alpha=0.7, edgecolor='none'))

# Add dimension annotations
# Court dimensions
ax.annotate('', xy=(COURT_WIDTH + 1, 0), xytext=(COURT_WIDTH + 1, COURT_LENGTH),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(COURT_WIDTH + 1.5, COURT_LENGTH/2, '94 ft', rotation=90, 
        va='center', ha='left', color='red', fontsize=12, weight='bold')

ax.annotate('', xy=(0, -1), xytext=(COURT_WIDTH, -1),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(COURT_WIDTH/2, -1.5, '50 ft', 
        va='top', ha='center', color='red', fontsize=12, weight='bold')

# Key measurements
ax.text(25, 9.5, '19 ft', ha='center', va='center', color='white', 
        fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
ax.text(key_left - 1.5, 9.5, '16 ft', rotation=90, ha='center', va='center', 
        color='white', fontsize=10, weight='bold', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))

# Hash mark distance
ax.text(-1.5, 28, '28 ft', rotation=90, ha='center', va='center', color='white', 
        fontsize=9, weight='bold', bbox=dict(boxstyle='round', facecolor='purple', alpha=0.7))

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#00ff00', 
               markersize=12, label='Primary Keypoints (Most Visible)', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#ffff00', 
               markersize=10, label='Secondary Keypoints (Good)', markeredgecolor='black', markeredgewidth=1.5),
    plt.Line2D([0], [0], marker='d', color='w', markerfacecolor='#00bfff', 
               markersize=10, label='Optional Keypoints (If Visible)', markeredgecolor='black', markeredgewidth=1.5),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)

# Title
ax.set_title('NBA Court Calibration Keypoints\n(Half-Court View - Near Baseline)', 
             fontsize=16, weight='bold', pad=20)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xlabel('X - Width (feet)', fontsize=12, weight='bold')
ax.set_ylabel('Y - Length from Baseline (feet)', fontsize=12, weight='bold')

# Add note box
note_text = """RECOMMENDED CALIBRATION POINTS:
✓ Baseline corners (0,0) and (50,0)
✓ Free throw corners (17,19) and (33,19)
✓ 3-point arc top (25, 27.75)
✓ 3-point wings (3,14) and (47,14)
✓ Hash marks (0,28) and (50,28) if visible
✓ Halfcourt line if visible

Minimum: 4-5 points
Recommended: 8-10 points"""

ax.text(25, -7, note_text, ha='center', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', 
                  edgecolor='black', linewidth=2, alpha=0.95),
        family='monospace')

# Add camera icon to show POV/origin at halfcourt near (0, 47)
camera_x = 0
camera_y = 47
# Camera body
camera_body = FancyBboxPatch((camera_x - 2.5, camera_y - 0.8), 3, 1.6,
                              boxstyle="round,pad=0.1", 
                              facecolor='#333333', edgecolor='black', linewidth=2, zorder=20)
ax.add_patch(camera_body)
# Camera lens
lens = Circle((camera_x - 1, camera_y), 0.6, facecolor='#555555', edgecolor='black', linewidth=1.5, zorder=21)
ax.add_patch(lens)
lens_center = Circle((camera_x - 1, camera_y), 0.3, facecolor='#87ceeb', edgecolor='white', linewidth=1, zorder=22)
ax.add_patch(lens_center)
# Camera label with arrow pointing to camera
ax.annotate('BROADCAST\nCAMERA VIEW', xy=(camera_x - 1, camera_y), xytext=(-8, 47),
            fontsize=9, weight='bold', color='red', ha='right', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', edgecolor='red', linewidth=2, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('docs/nba_court_calibration_keypoints.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("✓ Court diagram saved to: docs/nba_court_calibration_keypoints.png")
plt.close()

print("\n" + "="*60)
print("NBA COURT CALIBRATION KEYPOINTS")
print("="*60)
print("\nPRIMARY KEYPOINTS (Green circles):")
for name, coords in primary_points.items():
    print(f"  • {name:30s} {coords}")
print("\nSECONDARY KEYPOINTS (Yellow squares):")
for name, coords in secondary_points.items():
    print(f"  • {name:30s} {coords}")
print("\nOPTIONAL KEYPOINTS (Blue diamonds):")
for name, coords in optional_points.items():
    print(f"  • {name:30s} {coords}")
print("="*60)
