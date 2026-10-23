# Learned Homography Network

Automatic court calibration using deep learning. Replaces manual point-clicking with a neural network that predicts homography matrices from video frames.

## 🎯 Overview

**Problem:** Manual homography calibration is labor-intensive and required for every new video.

**Solution:** Train a neural network once on 10-20 manually calibrated videos, then automatically calibrate infinite future videos.

**Inspiration:** End-to-end learning from autonomous driving (camera→world transformation).

## 📁 Files

- `model.py` - LearnedHomography network (ResNet50 backbone)
- `prepare_data.py` - Bootstrap dataset from manual calibrations
- `train.py` - Training pipeline with custom loss function
- `predict_homography.py` - Inference script for new videos

## 🚀 Quick Start

### Step 1: Prepare Training Data

Manually calibrate 10-20 NBA broadcast videos (one-time effort):

```bash
# Calibrate each video manually
python homography/manual_homography.py --video game1.mp4
python homography/manual_homography.py --video game2.mp4
# ... repeat for 10-20 videos

# Prepare dataset from calibrations
python homography/learned/prepare_data.py \
  --calibrations . \
  --videos data/sample/videos \
  --output homography/learned/data/dataset.json \
  --frames-per-video 50
```

This extracts 50 frames per video and creates a training dataset.

### Step 2: Train the Network

```bash
# Train on local GPU
python homography/learned/train.py \
  --data homography/learned/data/dataset.json \
  --output homography/learned/checkpoints \
  --epochs 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --target-rmse 1.0 \
  --wandb

# Or train on Kaggle/Google Cloud for faster training
```

Training takes ~1-2 hours on GPU. Target: < 1.0 ft RMS error.

### Step 3: Predict on New Videos

```bash
# Automatic prediction (no manual clicking!)
python homography/learned/predict_homography.py \
  --video new_game.mp4 \
  --model homography/learned/checkpoints/best.pt \
  --visualize

# Output: homography_matrix_new_game.json
```

Done in 0.1 seconds! Now use with existing pipeline:

```bash
python map_ball_to_court.py \
  --ball-positions ball_positions.json \
  --calibration homography_matrix_new_game.json
```

## 🏗️ Architecture

### Network Design

```python
LearnedHomography:
  ├── ResNet50 backbone (pretrained on ImageNet)
  │   └── Extracts 2048 visual features
  │
  └── Homography head (3 layers)
      ├── Linear(2048 → 512) + ReLU + BatchNorm + Dropout
      ├── Linear(512 → 256) + ReLU + BatchNorm + Dropout
      └── Linear(256 → 9)  # Output: 3x3 matrix
```

**Total parameters:** 24.7M (mostly from pretrained ResNet50)

### Loss Function

Combines two objectives:

1. **Matrix MSE:** Direct comparison of predicted vs ground truth matrix
2. **Spatial Consistency:** Reprojection error in feet (weighted 10x)

```python
loss = matrix_loss + 10.0 * spatial_loss
```

Spatial loss is weighted higher because accuracy in feet matters more than exact matrix values.

### Data Augmentation

Broadcast-specific augmentations:

- ✅ Brightness/contrast (different arena lighting)
- ✅ Color jitter (different TV color grading)
- ✅ Gaussian noise (compression artifacts)
- ✅ Slight zoom/crop (different broadcast framing)

NOT using:
- ❌ Rotation (broadcast angle is consistent)
- ❌ Perspective warping (that's what we're learning!)
- ❌ Flips (court orientation matters)

## 📊 Training Strategy

### Bootstrap Training

**One-time investment:**
- Manually calibrate 10-20 NBA broadcast videos (~2 hours)
- Different arenas: Lakers, Celtics, Mavs, Warriors, etc.
- Different networks: ESPN, TNT, NBA TV
- Different games: regular season, playoffs, different years

**Dataset preparation:**
- Extract 50 frames per video
- Total: 10 videos × 50 frames = 500 images
- Augment 5x: 2,500+ training samples

**Training:**
- 100 epochs on GPU (~1-2 hours)
- Early stopping when RMSE < 1.0 ft
- Best model saved automatically

**Result:**
- Never manually calibrate again!
- Automatic prediction for all future videos

### Expected Performance

- **Target:** < 1.0 ft RMS error
- **Your manual calibration:** 0.696 ft RMS error (excellent!)
- **Network goal:** Match or exceed manual accuracy

## 🎓 Technical Details

### Homography Matrix

A 3×3 matrix that transforms pixel coordinates → court coordinates:

```
[x']   [h11 h12 h13]   [x]
[y'] = [h21 h22 h23] × [y]
[w']   [h31 h32 h33]   [1]

x_court = x' / w'  (in feet)
y_court = y' / w'  (in feet)
```

### Why This Works

1. **Court features are consistent:** Lines, paint, 3-point arc always same shape
2. **Broadcast angle is consistent:** Similar camera position across games
3. **ResNet50 is good at geometry:** Pretrained on ImageNet, knows lines/corners
4. **Spatial loss ensures accuracy:** Network learns to minimize error in feet

### Advantages Over Manual

| Manual Calibration | Learned Homography |
|-------------------|-------------------|
| 5 minutes per video | 0.1 seconds per video |
| Requires clicking 4-5 points | Fully automatic |
| Human error possible | Consistent accuracy |
| Labor intensive | Scales infinitely |
| Camera-specific | Generalizes to new cameras |

## 📈 Next Steps

After Phase 1 (Learned Homography) is working:

**Phase 2: Multi-Task Detection**
- Extend network with ball/player detection heads
- Detections output directly in court coordinates
- Single forward pass (faster than current pipeline)

**Phase 3: Temporal Reasoning**
- Add TransformerEncoder for video sequences
- Trajectory prediction during occlusion
- Smooth tracking with temporal consistency

## 🔧 Requirements

```bash
# Already in environment.yml
torch>=2.0.0
torchvision
albumentations
opencv-python
wandb (optional, for logging)
```

## 📝 File Formats

### Input: Dataset JSON

```json
[
  {
    "image_path": "frames/luka_ayton/luka_ayton_frame_000150.jpg",
    "homography_matrix": [[...], [...], [...]],
    "calibration_points": [[px, py, cx, cy], ...],
    "video_source": "data/sample/videos/Luka_Ayton_PNR_Luka_3.mp4",
    "frame_number": 150,
    "arena": "American Airlines Center",
    "rms_error": 0.696
  },
  ...
]
```

### Output: Homography JSON

Same format as manual calibration:

```json
{
  "homography_matrix": [[...], [...], [...]],
  "video_path": "new_game.mp4",
  "frame_number": 152,
  "model_path": "best.pt",
  "model_rmse": 0.85,
  "video_info": {
    "total_frames": 305,
    "fps": 30.0,
    "width": 1280,
    "height": 720
  }
}
```

## 🤝 Contributing

This is Phase 1 of the end-to-end learning approach. Once working, we'll extend to Phase 2 (multi-task detection).

---

**Status:** Ready to start training after collecting 10-20 manual calibrations

**Next:** Download NBA highlights, manually calibrate, train network!
