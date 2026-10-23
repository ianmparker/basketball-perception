# Mathematical Foundations - AI Basketball Coach

**Purpose:** Document all mathematical formulas, algorithms, and equations used throughout the project.

---

## 📐 Table of Contents

1. [Homography Transformation](#1-homography-transformation)
2. [Object Detection (YOLOv8)](#2-object-detection-yolov8)
3. [Multi-Object Tracking (ByteTrack)](#3-multi-object-tracking-bytetrack)
4. [Trajectory Analysis](#4-trajectory-analysis)
5. [Pose Estimation](#5-pose-estimation)

---

## 1. Homography Transformation

**Purpose:** Transform pixel coordinates from camera view → real-world court coordinates.

### 1.1 Homogeneous Coordinates

A point in 2D space (x, y) is represented in homogeneous coordinates as:

```
p = [x, y, 1]ᵀ
```

**Why?** Allows translation, rotation, and scaling to be represented as matrix multiplication.

### 1.2 Homography Matrix

A homography H is a 3×3 matrix that maps points from one plane to another:

```
p' = H × p

where:
p = [x, y, 1]ᵀ         (source point in pixels)
p' = [x', y', w']ᵀ     (destination point in homogeneous coords)
H = 3×3 homography matrix
```

The homography matrix structure:

```
     ⎡ h₀₀  h₀₁  h₀₂ ⎤
H =  ⎢ h₁₀  h₁₁  h₁₂ ⎥
     ⎣ h₂₀  h₂₁  h₂₂ ⎦
```

### 1.3 Computing Homography from Point Correspondences

Given N ≥ 4 point pairs (pᵢ, p'ᵢ), solve for H using Direct Linear Transform (DLT):

**Input:**
- Source points (image): [(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)]
- Destination points (court): [(x'₁, y'₁), (x'₂, y'₂), ..., (x'ₙ, y'ₙ)]

**For each point pair, we have:**

```
x'ᵢ = (h₀₀·xᵢ + h₀₁·yᵢ + h₀₂) / (h₂₀·xᵢ + h₂₁·yᵢ + h₂₂)
y'ᵢ = (h₁₀·xᵢ + h₁₁·yᵢ + h₁₂) / (h₂₀·xᵢ + h₂₁·yᵢ + h₂₂)
```

**Rearranging to eliminate division:**

```
h₀₀·xᵢ + h₀₁·yᵢ + h₀₂ - h₂₀·xᵢ·x'ᵢ - h₂₁·yᵢ·x'ᵢ - h₂₂·x'ᵢ = 0
h₁₀·xᵢ + h₁₁·yᵢ + h₁₂ - h₂₀·xᵢ·y'ᵢ - h₂₁·yᵢ·y'ᵢ - h₂₂·y'ᵢ = 0
```

**Matrix form (for all N points):**

```
A × h = 0

where A is 2N × 9 matrix, h = [h₀₀, h₀₁, h₀₂, h₁₀, h₁₁, h₁₂, h₂₀, h₂₁, h₂₂]ᵀ
```

**Solution:** Singular Value Decomposition (SVD) of A
- h is the right singular vector corresponding to smallest singular value
- OpenCV: `cv2.findHomography()` implements this with RANSAC for robustness

### 1.4 Applying Homography Transformation

**Forward transformation (pixel → court):**

```
⎡ x' ⎤       ⎡ x ⎤
⎢ y' ⎥ = H × ⎢ y ⎥
⎣ w' ⎦       ⎣ 1 ⎦

Final coordinates:
x_court = x' / w'
y_court = y' / w'
```

**Inverse transformation (court → pixel):**

```
⎡ x ⎤           ⎡ x' ⎤
⎢ y ⎥ = H⁻¹ × ⎢ y' ⎥
⎣ w ⎦           ⎣ 1  ⎦

Final coordinates:
x_pixel = x / w
y_pixel = y / w
```

**Implementation:**
```python
# OpenCV provides optimized functions
import cv2
import numpy as np

# Apply homography to single point
point_pixel = np.array([[x, y]], dtype=np.float32)
point_court = cv2.perspectiveTransform(point_pixel.reshape(-1, 1, 2), H)

# Apply to multiple points (N×1×2 array)
points_court = cv2.perspectiveTransform(points_pixel, H)
```

### 1.5 NBA Court Coordinate System

**Standard NBA court dimensions:**

```
Length: 94 feet (28.65 meters)
Width: 50 feet (15.24 meters)

Coordinate system (origin at baseline corner):
  (0, 0) ──────────────────── (94, 0)
    │                            │
    │                            │
    │        NBA Court           │
    │                            │
    │                            │
  (0, 50) ──────────────────── (94, 50)
```

**Key court features:**
- **3-point arc distance:** 23.75 feet (22 feet in corners)
- **Free-throw line:** 15 feet from baseline
- **Paint (key) width:** 16 feet
- **Basket location:** 5.25 feet from baseline, 25 feet from sideline

### 1.6 Homography Quality Metrics

**Reprojection Error:** Measure transformation accuracy

```
For each point pair (pᵢ, p'ᵢ):

p'ᵢ_predicted = H × pᵢ
error_i = ||p'ᵢ - p'ᵢ_predicted||₂

Total RMS error:
RMS = √(Σᵢ error_i² / N)
```

**Good homography:** RMS error < 5 pixels for manual selection, < 10 pixels for automatic detection

---

## 2. Object Detection (YOLOv8)

**Algorithm:** You Only Look Once (YOLO) - Single-stage object detector

### 2.1 Network Architecture

**Backbone:** CSPDarknet53 (Cross Stage Partial connections)
**Neck:** FPN + PAN (Feature Pyramid Network + Path Aggregation Network)
**Head:** Decoupled detection head (classification + regression)

### 2.2 Bounding Box Representation

**Format:** Center coordinates + width + height

```
bbox = [x_center, y_center, width, height]

where:
x_center, y_center = center of box (pixels)
width, height = box dimensions (pixels)
```

**Conversion to corner coordinates:**

```
x_min = x_center - width/2
y_min = y_center - height/2
x_max = x_center + width/2
y_max = y_center + height/2
```

### 2.3 Loss Function

YOLOv8 uses a combination of three losses:

```
L_total = λ_box · L_box + λ_cls · L_cls + λ_dfl · L_dfl

where:
L_box = CIoU loss (bounding box regression)
L_cls = Binary Cross Entropy loss (classification)
L_dfl = Distribution Focal Loss (box distribution)
```

**Complete Intersection over Union (CIoU) Loss:**

```
L_CIoU = 1 - IoU + ρ²(b, bᵍᵗ)/c² + α·v

where:
IoU = Intersection over Union
ρ = Euclidean distance between centers
c = diagonal length of smallest enclosing box
v = consistency of aspect ratio
α = positive trade-off parameter
```

### 2.4 Intersection over Union (IoU)

**Definition:** Measure of overlap between two boxes

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)

Area(A ∩ B) = intersection area
Area(A ∪ B) = union area = Area(A) + Area(B) - Area(A ∩ B)

Range: [0, 1] where 1 = perfect overlap
```

**Implementation:**

```python
def box_iou(box1, box2):
    # box format: [x_min, y_min, x_max, y_max]
    
    # Intersection area
    x_min = max(box1[0], box2[0])
    y_min = max(box1[1], box2[1])
    x_max = min(box1[2], box2[2])
    y_max = min(box1[3], box2[3])
    
    intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
    
    # Union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0
```

### 2.5 Non-Maximum Suppression (NMS)

**Purpose:** Remove duplicate detections

**Algorithm:**
1. Sort detections by confidence score (descending)
2. Select detection with highest score
3. Remove all detections with IoU > threshold (typically 0.45)
4. Repeat until no detections remain

**Pseudocode:**

```
NMS(detections, iou_threshold):
    sort detections by score (descending)
    selected = []
    
    while detections not empty:
        best = detections[0]
        selected.append(best)
        detections.remove(best)
        
        for det in detections:
            if IoU(best.bbox, det.bbox) > iou_threshold:
                detections.remove(det)
    
    return selected
```

### 2.6 Evaluation Metrics

**Mean Average Precision (mAP):**

```
AP = ∫₀¹ Precision(Recall) dRecall

Precision = TP / (TP + FP)
Recall = TP / (TP + FN)

mAP = (1/N) · Σᵢ APᵢ  (average over all classes)

mAP@50 = mAP at IoU threshold 0.5
mAP@50:95 = average mAP at IoU thresholds [0.5, 0.55, ..., 0.95]
```

**Our Model Performance:**
- mAP@50: 0.678
- Player class: 0.918 (excellent)
- Ball class: 0.232 (needs improvement)

---

## 3. Multi-Object Tracking (ByteTrack)

**Algorithm:** ByteTrack with Kalman filtering for motion prediction

### 3.1 Kalman Filter State Model

**State vector (8 dimensions):**

```
x = [u, v, s, r, u̇, v̇, ṡ, 0]ᵀ

where:
u, v = bounding box center coordinates (pixels)
s = scale (box area)
r = aspect ratio (width/height)
u̇, v̇ = velocities of center (pixels/frame)
ṡ = scale change rate
```

**State transition (constant velocity model):**

```
xₖ = F · xₖ₋₁ + w

F = ⎡ 1  0  0  0  Δt  0   0   0 ⎤
    ⎢ 0  1  0  0  0   Δt  0   0 ⎥
    ⎢ 0  0  1  0  0   0   Δt  0 ⎥
    ⎢ 0  0  0  1  0   0   0   0 ⎥
    ⎢ 0  0  0  0  1   0   0   0 ⎥
    ⎢ 0  0  0  0  0   1   0   0 ⎥
    ⎢ 0  0  0  0  0   0   1   0 ⎥
    ⎣ 0  0  0  0  0   0   0   1 ⎦

where:
Δt = time step (typically 1 frame)
w = process noise ~ N(0, Q)
```

### 3.2 Kalman Filter Prediction

**Predict next state:**

```
x̂ₖ|ₖ₋₁ = F · xₖ₋₁|ₖ₋₁
P̂ₖ|ₖ₋₁ = F · Pₖ₋₁|ₖ₋₁ · Fᵀ + Q

where:
x̂ₖ|ₖ₋₁ = predicted state
P̂ₖ|ₖ₋₁ = predicted covariance
Q = process noise covariance matrix
```

### 3.3 Kalman Filter Update

**Measurement model:**

```
zₖ = H · xₖ + v

H = ⎡ 1  0  0  0  0  0  0  0 ⎤
    ⎢ 0  1  0  0  0  0  0  0 ⎥
    ⎢ 0  0  1  0  0  0  0  0 ⎥
    ⎣ 0  0  0  1  0  0  0  0 ⎦

where:
zₖ = measurement vector [u, v, s, r]ᵀ (observed bbox)
v = measurement noise ~ N(0, R)
```

**Update equations:**

```
Innovation (measurement residual):
yₖ = zₖ - H · x̂ₖ|ₖ₋₁

Innovation covariance:
Sₖ = H · P̂ₖ|ₖ₋₁ · Hᵀ + R

Kalman gain:
Kₖ = P̂ₖ|ₖ₋₁ · Hᵀ · Sₖ⁻¹

Updated state estimate:
xₖ|ₖ = x̂ₖ|ₖ₁ + Kₖ · yₖ

Updated covariance:
Pₖ|ₖ = (I - Kₖ · H) · P̂ₖ|ₖ₋₁
```

### 3.4 Data Association (Hungarian Algorithm)

**Purpose:** Match detections to existing tracks

**Cost matrix:** IoU-based matching

```
C[i, j] = 1 - IoU(track_i, detection_j)

where i ∈ tracks, j ∈ detections
```

**Linear Assignment Problem (LAP):**

```
minimize: Σᵢⱼ C[i,j] · xᵢⱼ

subject to:
Σⱼ xᵢⱼ ≤ 1  ∀i  (each track matched to ≤1 detection)
Σᵢ xᵢⱼ ≤ 1  ∀j  (each detection matched to ≤1 track)
xᵢⱼ ∈ {0, 1}
```

**Solution:** Hungarian algorithm (Kuhn-Munkres) - O(n³) complexity
- Implemented via `lap.lapjv()` (Jonker-Volgenant algorithm, faster)

### 3.5 ByteTrack Two-Stage Association

**High-confidence detections (score > 0.6):**

```
1. Match high-score detections to tracked objects (IoU matching)
2. Update matched tracks with Kalman filter
3. Initialize new tracks for unmatched high-score detections
```

**Low-confidence detections (0.1 < score < 0.6):**

```
4. Match low-score detections to unmatched tracks
5. Recover tracks that were temporarily lost
6. Ignore remaining low-score detections
```

**Track state transitions:**

```
New → Tracked (after 1 successful match)
Tracked → Lost (after miss)
Lost → Tracked (if recovered within 30 frames)
Lost → Removed (after 30 frames without match)
```

---

## 4. Trajectory Analysis

### 4.1 Euclidean Distance

**2D distance between two points:**

```
d = √[(x₂ - x₁)² + (y₂ - y₁)²]

In court coordinates (feet):
distance = √[(x_court₂ - x_court₁)² + (y_court₂ - y_court₁)²]
```

### 4.2 Velocity Calculation

**Average velocity between consecutive frames:**

```
v = Δd / Δt

where:
Δd = distance traveled between frames (feet)
Δt = time between frames (seconds)

For 30 FPS video: Δt = 1/30 ≈ 0.0333 seconds
```

**Speed conversion:**

```
v (mph) = v (feet/second) × 0.681818
v (km/h) = v (feet/second) × 1.09728
v (m/s) = v (feet/second) × 0.3048
```

### 4.3 Acceleration

**Change in velocity:**

```
a = Δv / Δt = (v₂ - v₁) / Δt

where:
v₁, v₂ = velocities at consecutive time steps
Δt = time between measurements
```

### 4.4 Trajectory Smoothing

**Moving average filter (simple smoothing):**

```
x̄ᵢ = (1/w) · Σⱼ₌₀^(w-1) xᵢ₋ⱼ

where:
w = window size (e.g., 5 frames)
x̄ᵢ = smoothed position at frame i
```

**Gaussian smoothing:**

```
x̄ᵢ = Σⱼ G(j) · xᵢ₊ⱼ

where:
G(j) = (1/√(2πσ²)) · exp(-j²/(2σ²))  (Gaussian kernel)
σ = standard deviation
```

---

## 5. Pose Estimation

### 5.1 Keypoint Detection

**Output:** N keypoints per person

```
Keypoint = [x, y, confidence]

where:
x, y = pixel coordinates
confidence ∈ [0, 1] = detection confidence
```

**Common keypoint sets:**
- **COCO format:** 17 keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
- **MediaPipe:** 33 keypoints (full body including hands and face)
- **Basketball dataset:** 7 keypoints (neck, l_arm, r_arm, head, hip, l_leg, r_leg)

### 5.2 Joint Angle Calculation

**Angle between three keypoints (e.g., elbow angle):**

```
Given three points A, B, C forming angle at B:

Vector BA = [Ax - Bx, Ay - By]
Vector BC = [Cx - Bx, Cy - By]

Dot product:
BA · BC = |BA| · |BC| · cos(θ)

Angle:
θ = arccos((BA · BC) / (|BA| · |BC|))

where:
|BA| = √[(Ax - Bx)² + (Ay - By)²]
|BC| = √[(Cx - Bx)² + (Cy - By)²]
```

**Implementation:**

```python
import numpy as np

def calculate_angle(A, B, C):
    """Calculate angle ABC in degrees."""
    BA = np.array([A[0] - B[0], A[1] - B[1]])
    BC = np.array([C[0] - B[0], C[1] - B[1]])
    
    cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    
    return np.degrees(angle)
```

### 5.3 Pose Similarity (Optional)

**Object Keypoint Similarity (OKS):**

```
OKS = Σᵢ exp(-dᵢ²/(2s²κᵢ²)) · δ(vᵢ > 0) / Σᵢ δ(vᵢ > 0)

where:
dᵢ = Euclidean distance between predicted and GT keypoint i
s = √(object area) = scale factor
κᵢ = per-keypoint constant (controls falloff)
vᵢ = visibility flag
δ = indicator function
```

---

## 📚 References

### Homography:
- Hartley & Zisserman (2004): "Multiple View Geometry in Computer Vision"
- OpenCV Documentation: `cv2.findHomography()`, `cv2.perspectiveTransform()`

### Object Detection:
- Ultralytics YOLOv8: https://docs.ultralytics.com/
- YOLO series papers: Redmon et al. (2016-2023)

### Multi-Object Tracking:
- ByteTrack: Zhang et al. (2021) "ByteTrack: Multi-Object Tracking by Associating Every Detection Box"
- Kalman Filter: Kalman (1960) "A New Approach to Linear Filtering and Prediction Problems"
- Hungarian Algorithm: Kuhn (1955), Munkres (1957)

### Trajectory Analysis:
- Basic kinematics: Physics textbooks
- Smoothing techniques: Signal processing literature

### Pose Estimation:
- COCO keypoints: Lin et al. (2014) "Microsoft COCO: Common Objects in Context"
- MediaPipe: Google Research, https://google.github.io/mediapipe/

---

**Last Updated:** January 7, 2026  
**Next Addition:** Court zone classification, play detection heuristics
