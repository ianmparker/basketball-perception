# NBA Court Calibration Keypoints

## Court Dimensions Overview

```
                    HALFCOURT LINE (47 ft from baseline)
                            |
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    │         [HASH]        │        [HASH]         │  ← Hash marks at 28 ft
    │           (28,0)      │          (28,50)      │
    │                       │                       │
    │                       │                       │
    │      ┌────────────────┼────────────────┐      │
    │      │                │                │      │
    │      │    FT LANE     │     FT LANE    │      │  ← Free throw lane
    │      │   (19 ft)      │    (19 ft)     │      │    corners at 19 ft
    │      │                │                │      │
    │      │     [FT-L]     │     [FT-R]     │      │
    │      │    (19,17)     │    (19,33)     │      │
    │      │                │                │      │
    │      └────────────────┼────────────────┘      │
    │                       │                       │
    │                       │                       │
    │          [3PT-LW]     [3PT-TOP]  [3PT-RW]     │  ← 3-point arc
    │           (14,8.75)  (19.75,25)  (14,41.25)   │    Top: 19.75 ft
    │                                                │    Wings: 14 ft
    │                                                │
    │  [3PT-LC]                              [3PT-RC]│  ← 3-point corners
    │  (14,3)                                (14,47) │    14 ft from baseline
    │                                                │    3 ft from sideline
    │                                                │
    │                                                │
    [BL-NEAR]                                [BL-FAR]   ← Baseline
    (0,0)                                      (0,50)

                    50 ft (court width)
    └──────────────────────────────────────────────┘
```

## Available Calibration Keypoints

### 🟢 **PRIMARY KEYPOINTS** (Most Visible in Broadcast)

| Keypoint Name | Court Coordinates (ft) | Description | Visibility |
|---------------|------------------------|-------------|------------|
| `baseline_left_near` | (0, 0) | Baseline left corner | ⭐⭐⭐ Excellent |
| `baseline_left_far` | (0, 50) | Baseline right corner | ⭐⭐⭐ Excellent |
| `free_throw_left_near` | (19, 17) | FT lane corner (near side) | ⭐⭐⭐ Excellent |
| `free_throw_left_far` | (19, 33) | FT lane corner (far side) | ⭐⭐⭐ Excellent |
| `three_point_top` | (19.75, 25) | 3-point arc peak | ⭐⭐⭐ Excellent |
| `three_point_left_wing` | (14, 8.75) | 3-point arc break point (left) | ⭐⭐ Good |
| `three_point_right_wing` | (14, 41.25) | 3-point arc break point (right) | ⭐⭐ Good |

### 🟡 **SECONDARY KEYPOINTS** (Good for Half-Court Views)

| Keypoint Name | Court Coordinates (ft) | Description | Visibility |
|---------------|------------------------|-------------|------------|
| `hash_mark_left` | (28, 0) | Hash mark on left sideline | ⭐⭐ Good |
| `hash_mark_right` | (28, 50) | Hash mark on right sideline | ⭐⭐ Good |
| `three_point_left_corner` | (14, 3) | 3-point corner (left) | ⭐ Fair |
| `three_point_right_corner` | (14, 47) | 3-point corner (right) | ⭐ Fair |

### 🔵 **OPTIONAL KEYPOINTS** (If Full Court Visible)

| Keypoint Name | Court Coordinates (ft) | Description | Visibility |
|---------------|------------------------|-------------|------------|
| `halfcourt_near` | (47, 0) | Halfcourt line left | ⭐⭐ Good |
| `halfcourt_far` | (47, 50) | Halfcourt line right | ⭐⭐ Good |

## Official NBA Dimensions Reference

```
┌──────────────────────────────────────────────────┐
│                                                  │
│  Court:           94 ft × 50 ft                  │
│  Free throw line: 19 ft from baseline            │
│  Key (paint):     19 ft × 16 ft (centered)       │
│  3-point line:    23'9" (23.75 ft) at top        │
│                   22 ft in corners               │
│  3-point straight: 14 ft from baseline           │
│  3-point from sideline: 3 ft                     │
│  Hash marks:      28 ft from baseline (sidelines)│
│  Center circle:   6 ft radius                    │
│  Restricted area: 4 ft radius                    │
│                                                  │
└──────────────────────────────────────────────────┘
```

## Recommended Calibration Strategy

### For **Half-Court Broadcast View** (Most Common):

**Minimum 4-5 points** (for basic accuracy):
1. ✅ `baseline_left_near` (0, 0)
2. ✅ `baseline_left_far` (0, 50)
3. ✅ `three_point_top` (19.75, 25)
4. ✅ `free_throw_left_near` (19, 17)
5. ✅ `free_throw_left_far` (19, 33)

**Recommended 8-10 points** (for best accuracy):
- All 5 above, PLUS:
6. ✅ `three_point_left_wing` (14, 8.75)
7. ✅ `three_point_right_wing` (14, 41.25)
8. ✅ `hash_mark_left` (28, 0) — if visible
9. ✅ `hash_mark_right` (28, 50) — if visible
10. ✅ `halfcourt_near` (47, 0) — if visible (can click edge even if 1" off-screen)

### For **Full Court View**:
- Use all baseline points on both ends
- Use free throw and 3-point features on both ends
- Include halfcourt line points
- Aim for 12-16 well-distributed points

## Calibration Tips

### ✅ **DO:**
- Click on the **edge pixels** of court lines (not the center of thick lines)
- Use 3-point **arc features** (top, wings) instead of corners when possible
- Click edge of screen for features slightly off-screen (extrapolation works)
- Use **90-degree corners** (FT lane corners, baseline corners) for precision
- Distribute points across depth (near/far) and width (left/right)

### ❌ **DON'T:**
- Skip features that are 1-2 inches off-screen (edge clicks work!)
- Click corners when arc features are more visible
- Use only features on one side of the court
- Forget to check that points are well-distributed spatially

## Visual Reference

```
             HALFCOURT (47 ft)
                  │
        ┌─────────┼─────────┐
        │    28ft │         │  ← Hash marks (28 ft)
        │         │         │
        │    ┌────┼────┐    │
        │    │ 19 │    │    │  ← FT lane corners (19 ft)
        │    └────┼────┘    │
        │         │         │
        │     19.75ft       │  ← 3-pt top (19.75 ft)
        │      /  │  \      │
        │   14ft  │  14ft   │  ← 3-pt wings (14 ft)
        │   /     │     \   │
        │  │      │      │  │  ← 3-pt corners (14 ft)
        │                  │
        [0]──────────────[0]   ← Baseline (0 ft)
```

## Usage in Manual Calibration Tool

When you run the calibration tool:

```bash
python homography/manual_homography.py --video data/sample/videos/broadcast_01.mp4 --output homography_matrix_broadcast_01.json
```

The tool displays these keypoints and allows you to click corresponding pixel locations in the video frame. The tool then computes the homography matrix that maps pixel coordinates → court coordinates (in feet).

**Target accuracy**: < 1.0 ft RMS error

