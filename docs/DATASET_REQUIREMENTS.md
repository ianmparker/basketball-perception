# Curated Basketball Datasets - Real Working Links

## ✅ Verified Working Datasets

### 1. **Basketball Players (Roboflow)** ⭐⭐⭐⭐⭐ PRIORITY #1
**URL:** https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2
**Type:** Object Detection
**Classes:** 3 classes
- `player` - Basketball players
- `ball` - Basketball
- `referee` - Game officials

**Stats:**
- Images: Available for download
- Format: COCO JSON, YOLO, Pascal VOC
- License: Public (check specific license on download)

**Why You Need This:**
- ✅ **ESSENTIAL** - Core detection for your MVP
- ✅ Players, ball, referee = primary objects
- ✅ Object detection is foundation of entire pipeline
- ✅ Use this FIRST before anything else

**Use Cases from Description:**
- Sports analytics ✓
- Real-time tracking ✓
- Automated highlights ✓
- Training feedback ✓
- Fan experience features ✓

---

### 2. **Basketball (NBA Stats - Kaggle)** ⭐⭐⭐⭐ PRIORITY #2
**URL:** https://www.kaggle.com/datasets/wyattowalsh/basketball
**Type:** Statistical Database (SQLite)
**Size:** 2.35 GB
**Contains:**
- 65,000+ games with box scores
- 13M+ play-by-play rows
- 3,632 players
- 4,800+ player career stats

**Why You Need This:**
- ✅ **VERY USEFUL** - Action recognition labels
- ✅ Shot success prediction training data
- ✅ Player statistics for context
- ✅ Validation data for your model outputs
- ✅ Advanced coaching features

**When to Use:**
- Phase 3: Action recognition training
- Phase 4+: Advanced analytics features
- NOT needed for initial detection model

---

## 🎯 What You Actually Need (Prioritized)

### **Phase 2: Detection Model (THIS WEEK)**
```
✅ MUST HAVE:
- Basketball Players (Roboflow) - Object Detection
  → Player, ball, referee detection

❌ DON'T NEED YET:
- Instance Segmentation (overkill for MVP)
- Semantic Segmentation (not needed)
- Classification alone (too limited)
```

### **Phase 3: Enhanced Detection (WEEKS 2-4)**
```
✅ SHOULD ADD:
- Keypoint Detection dataset (for court lines)
  → Baseline, sideline, three-point line
  → Camera calibration
  
✅ OPTIONAL:
- Player pose keypoints (for shot form analysis)
  → Use MediaPipe/BlazePose pre-trained instead
```

### **Phase 4: Advanced Features (MONTH 2+)**
```
✅ ADD LATER:
- NBA Stats Database (Kaggle)
  → Action labels from play-by-play
  → Shot success prediction
  
✅ MULTIMODAL (Future):
- Combine vision + text descriptions
  → Advanced coaching insights
```

---

## 🔍 Dataset Type Analysis

### **Object Detection** ⭐⭐⭐⭐⭐ **YOU NEED THIS**
**What it is:** Bounding boxes around objects
**Classes:** Player, ball, referee
**Use for:** 
- Primary detection pipeline
- Tracking pipeline input
- 90% of your project needs

**Decision:** ✅ **START HERE - Basketball Players dataset**

---

### **Instance Segmentation** ⭐⭐ **DON'T NEED FOR MVP**
**What it is:** Pixel-perfect masks around each object
**Use for:**
- Precise player outlines
- Removing backgrounds
- AR overlays

**Decision:** ❌ **Skip for now** - Overkill, slower inference, not needed for tracking
**Maybe Later:** If doing AR features or professional video editing

---

### **Keypoint Detection** ⭐⭐⭐⭐ **USEFUL FOR ADVANCED FEATURES**
**What it is:** Specific points on objects
**Two types you might use:**

1. **Court Keypoints** ⭐⭐⭐⭐
   - Baseline corners, sideline corners, free-throw line, etc.
   - Use for: Camera calibration, bird's-eye view, court mapping
   - **Decision:** ✅ **Add in Phase 3** after detection works

2. **Player Pose Keypoints** ⭐⭐⭐
   - Shoulders, elbows, knees, etc.
   - Use for: Shot form analysis, action recognition
   - **Decision:** ⚠️ **Use pre-trained MediaPipe/BlazePose** instead of custom dataset
   - Why: These models are already excellent, no need to retrain

---

### **Semantic Segmentation** ⭐ **NOT NEEDED**
**What it is:** Classify every pixel (all players = one class)
**Use for:** Scene understanding (sky, ground, court)
**Decision:** ❌ **Skip entirely** - Not useful for basketball analysis

---

### **Classification** ⭐⭐⭐ **USEFUL LATER**
**What it is:** Label entire images with actions
**Classes:** Shooting, dribbling, passing, defending
**Use for:** Action recognition
**Decision:** ⚠️ **Use detection + temporal models** instead
- Better to detect players THEN classify their actions
- Pure classification loses spatial information

---

### **Multimodal** ⭐⭐⭐⭐ **ADVANCED FEATURE**
**What it is:** Combine vision + text/audio
**Use for:** 
- Vision + play-by-play text → Better action labels
- Vision + commentary → Automatic highlight generation
**Decision:** ✅ **Phase 4+** - Use NBA database play-by-play text

---

## 🎯 Your Recommended Dataset Strategy

### **Week 1-2: MVP Detection**
```python
Primary Dataset:
✅ Basketball Players (Roboflow) - Object Detection
   - Download in COCO format
   - Train YOLOv8 model
   - Classes: player, ball, referee
   
Cost: $0 (free)
Time: 2-3 days to train
```

### **Week 3-4: Enhanced Tracking**
```python
Add Dataset:
✅ Court Keypoints (if available on Roboflow Universe)
   - Search: "basketball court keypoints"
   - Or create small dataset (100 images) yourself
   - 14 keypoints for court lines
   
Optional:
⚠️ Use MediaPipe Pose (pre-trained, no dataset needed)
   - Free, works out of box
   - No training required
```

### **Month 2: Advanced Analytics**
```python
Add Database:
✅ NBA Stats (Kaggle) - 2.35 GB
   - Download SQLite database
   - Map play-by-play to actions
   - Train action classifier with weak supervision
   
Cost: $0 (free)
Time: 1 week integration
```

---

## 🤔 Roboflow vs Kaggle: Do You Need Both?

### **SHORT ANSWER: YES, BUT FOR DIFFERENT THINGS**

```
┌─────────────────────────────────────────────┐
│                                             │
│  Roboflow = IMAGES/VIDEOS with annotations │
│  Kaggle   = DATA/STATISTICS                 │
│                                             │
└─────────────────────────────────────────────┘
```

### **Roboflow Universe** 🖼️
**What it provides:**
- Annotated images/videos
- Ready-to-train datasets
- COCO/YOLO formats
- Computer vision datasets

**When to use:**
- Training detection models ✓
- Training segmentation models ✓
- Training keypoint models ✓

**Your needs:**
- ✅ Basketball Players dataset → Detection
- ✅ Court Keypoints dataset → Court detection
- ❌ DON'T need segmentation datasets

---

### **Kaggle** 📊
**What it provides:**
- Statistical databases
- CSV/SQLite files
- Tabular data
- Play-by-play text

**When to use:**
- Action recognition labels ✓
- Statistical validation ✓
- Predictive modeling ✓
- Advanced analytics ✓

**Your needs:**
- ✅ NBA Stats database → Action labels + analytics
- ⚠️ Download later (2.35 GB, not needed for MVP)

---

## 📋 Final Recommendations

### ✅ **DOWNLOAD NOW (This Week)**
1. **Basketball Players (Roboflow)**
   - URL: https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2
   - Format: COCO JSON
   - Use: Train detection model
   - Priority: **CRITICAL**

### ⏳ **DOWNLOAD LATER (Weeks 2-4)**
2. **Court Keypoints Dataset (Roboflow)**
   - Search on: https://universe.roboflow.com/
   - Keywords: "basketball court keypoints" or "basketball court detection"
   - Use: Court line detection
   - Priority: **MEDIUM**

3. **NBA Stats Database (Kaggle)**
   - URL: https://www.kaggle.com/datasets/wyattowalsh/basketball
   - Download: 2.35 GB SQLite file
   - Use: Action recognition + analytics
   - Priority: **LOW (for MVP), HIGH (for production)**

### ❌ **DON'T NEED**
- Instance Segmentation datasets
- Semantic Segmentation datasets
- Classification-only datasets
- Multimodal datasets (for now)

---

## 🚀 Immediate Action Plan

### **TODAY:**
```bash
# 1. Sign up for Roboflow (free)
Go to: https://app.roboflow.com/

# 2. Download Basketball Players dataset
Go to: https://universe.roboflow.com/roboflow-universe-projects/basketball-players-fy4c2
Click: "Download Dataset"
Select: "COCO JSON" format
Extract to: data/raw/roboflow_basketball/
```

### **THIS WEEK:**
```bash
# 3. Install Roboflow SDK
conda activate basketball-perception
pip install roboflow

# 4. Test dataset
python -c "from roboflow import Roboflow; print('Roboflow SDK ready')"

# 5. Start training
python models/src/train.py --config models/configs/yolov8_small.yaml
```

### **NEXT WEEK:**
```bash
# 6. Search for court keypoints dataset
# 7. If not found, create small dataset (100 images)
# 8. Download NBA database (optional, 2.35 GB)
```

---

## 💾 Storage Planning

```
Estimated Storage Needs:
- Basketball Players dataset:    ~500 MB - 1 GB
- Court Keypoints dataset:       ~100-200 MB
- NBA Stats database:            2.35 GB
- Your own videos:               ~1-5 GB
- Model checkpoints:             ~200-500 MB
- DVC cache:                     Same as above
─────────────────────────────────────────────
Total:                           ~5-10 GB
```

Make sure you have at least **15-20 GB free** for comfortable development.

---

## 📝 Summary

**Question 1: Which datasets do I need?**
- ✅ Basketball Players (Roboflow) - **ESSENTIAL NOW**
- ✅ NBA Stats (Kaggle) - **USEFUL LATER**
- ⚠️ Court Keypoints - **NICE TO HAVE**

**Question 2: Do I need both Roboflow and Kaggle?**
- ✅ YES - They serve different purposes
- Roboflow = Vision datasets (images)
- Kaggle = Statistical datasets (text/numbers)

**Question 3: Which dataset types?**
- ✅ Object Detection - **MUST HAVE**
- ✅ Keypoint Detection - **FOR COURT LINES**
- ❌ Instance Segmentation - **SKIP**
- ❌ Semantic Segmentation - **SKIP**
- ❌ Classification - **USE DETECTION INSTEAD**
- ⏳ Multimodal - **PHASE 4+**

**Question 4: Are any redundant?**
- No redundancy identified
- Each serves specific purpose
- Download based on project phase

---

**NEXT STEP:** Download the Basketball Players dataset from Roboflow and let's train your first model! 🏀
