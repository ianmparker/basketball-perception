# Ball Detection Training - Quick Guide

## Dataset: NBA Ball Detection (4,038 images)
- **Train**: 3,522 images
- **Val**: 344 images  
- **Test**: 172 images
- **Class**: basketball (1 class only)

## Training Strategy: FAST & FOCUSED

### Why This Approach:
- **Speed**: 30-45 minutes (not 11 hours)
- **GPU Cost**: ~2 hours (not 23 hours)
- **Focus**: Ball detection + homography (court position mapping)
- **Useful Output**: "Ball at 23.4 feet from basket" instead of just pixel coordinates

### Model: YOLOv8n (Nano)
- **Parameters**: 3.2M (vs 25.8M for YOLOv8m)
- **Speed**: 8x faster training
- **Size**: 6MB (vs 52MB)
- **Performance**: 80-90% mAP@50 expected for single class

### Kaggle Setup:

**Step 1: Upload Dataset to Kaggle**
```bash
# Zip the dataset
Compress-Archive -Path "data\raw\nba-ball-detection\*" -DestinationPath "nba-ball-detection.zip"

# Upload to Kaggle Datasets:
# 1. Go to https://www.kaggle.com/datasets
# 2. Click "New Dataset"
# 3. Upload nba-ball-detection.zip
# 4. Title: "NBA Ball Detection - 4K Images"
# 5. Make it private
```

**Step 2: Create Kaggle Notebook**

Use the 7-cell structure below (optimized for speed).

---

## Kaggle Notebook Cells

### Cell 1: Setup
```python
!pip install ultralytics -q
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

### Cell 2: Create YAML Config
```python
import os

yaml_content = """
path: /kaggle/input/nba-ball-detection/nba-ball-detection
train: train/images
val: valid/images
test: test/images

names:
  0: basketball
"""

yaml_path = '/kaggle/working/ball-detection.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# Verify dataset
train_imgs = len(os.listdir('/kaggle/input/nba-ball-detection/nba-ball-detection/train/images'))
val_imgs = len(os.listdir('/kaggle/input/nba-ball-detection/nba-ball-detection/valid/images'))
test_imgs = len(os.listdir('/kaggle/input/nba-ball-detection/nba-ball-detection/test/images'))

print(f"✅ Config created!")
print(f"📊 Train: {train_imgs} | Val: {val_imgs} | Test: {test_imgs}")
```

### Cell 3: Train YOLOv8n (FAST)
```python
from ultralytics import YOLO

# Use NANO model for speed
model = YOLO('yolov8n.pt')

# Train with aggressive settings for speed
results = model.train(
    data=yaml_path,
    epochs=50,           # Reduced from 200 (ball is easier to detect)
    batch=64,            # 2x larger (nano is smaller)
    device=0,
    workers=8,
    amp=True,
    cache=True,
    save_period=10,      # Save every 10 epochs
    project='/kaggle/working',
    name='ball-train',
    exist_ok=True,
    patience=20,         # Early stopping after 20 epochs no improvement
    imgsz=640
)

print("✅ Training complete!")
```

### Cell 4: Copy Weights to Root
```python
!cp /kaggle/working/ball-train/weights/best.pt /kaggle/working/best.pt
!cp /kaggle/working/ball-train/weights/last.pt /kaggle/working/last.pt
print("✅ Weights copied to /kaggle/working/")
```

### Cell 5: View Results
```python
from IPython.display import Image, display

# Show training curves
display(Image('/kaggle/working/ball-train/results.png'))
```

### Cell 6: Validation Metrics
```python
# Validate the best model
model = YOLO('/kaggle/working/best.pt')
metrics = model.val()

print(f"\n{'='*60}")
print("FINAL METRICS")
print(f"{'='*60}")
print(f"mAP@50:    {metrics.box.map50:.3f}")
print(f"mAP@50-95: {metrics.box.map:.3f}")
print(f"Precision: {metrics.box.p[0]:.3f}")
print(f"Recall:    {metrics.box.r[0]:.3f}")
```

### Cell 7: Create Download Package
```python
!mkdir -p /kaggle/working/download
!cp /kaggle/working/best.pt /kaggle/working/download/
!cp /kaggle/working/ball-train/results.csv /kaggle/working/download/
!cp /kaggle/working/ball-train/results.png /kaggle/working/download/

print("✅ Download package ready in /kaggle/working/download/")
print("📦 Files: best.pt, results.csv, results.png")
```

---

## Expected Results

**Training Time**: ~30-45 minutes on P100 GPU
**GPU Hours Used**: ~2 hours
**Expected Performance**:
- mAP@50: 85-92% (excellent for single class)
- Speed: 300+ FPS on RTX 2070 Super
- Model size: 6 MB (easy to deploy)

## Next Steps After Training

1. **Download** `best.pt` from `/kaggle/working/download/`
2. **Test** on your Luka video:
   ```bash
   yolo detect predict model=best.pt source=data/sample/videos/Luka_Ayton_PNR_Luka_3.mp4
   ```
3. **Integrate** with homography for court position tracking
4. **Analyze** ball trajectory in real-world coordinates (feet)

## Why This Works

- **Simpler problem**: 1 class (ball) vs 17 classes (players, refs, keypoints)
- **Smaller model**: YOLOv8n trains 8x faster than YOLOv8m
- **Sufficient data**: 3,522 images is plenty for single-class detection
- **Early stopping**: Will likely stop around epoch 30-35 (not full 50)
- **Focused outcome**: Enables homography-based ball tracking immediately
