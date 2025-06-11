# Computer Vision Project: Re-ldentification in a Single Feed

## 📌 Project Overview

This project focuses on analyzing football video footage using advanced computer vision techniques. The goal is to detect and track players, referees, and the ball, while maintaining consistent identities across frames, even after occlusion or exit/re-entry.

---

## 🔧 Approach & Methodology

1. **Detection**
   - **Model Used**: YOLOv11 (Custom Model)
   - **Classes Detected**: players, referees, goalkeepers, ball
   - **Batch Processing**: Implemented batched frame processing (batch size = 20) for efficient GPU utilization.

2. **Tracking**
   - **Model Used**: BoT-SORT
     - Appearance model: ResNet-based ReID
     - Motion model: Kalman Filter
      - Integrated tracking post detection, maintaining consistent player IDs.
   - **Model Used**: ByteTrack
      - **No appearance feature used**
      - Based on IoU and Kalman Filter
      - Tracks both high- and low-confidence detections
      - Maintains object identities across frames efficiently.

3. **Re-identification**
   - Used ReID model embedded in BoT-SORT for appearance matching.
   - Ensured players retain the same ID upon re-entering the frame.

4. **Video Output**
   - Annotated video output with:
     - IDs, bounding boxes, labels
     - ball interpolation
     - Optional saving to disk with caching

---

## 🧪 Techniques Tried & Outcomes

| Technique | Outcome |
|----------|---------|
| YOLOv11 with batched detection | ✅ Fast and accurate detection for all relevant classes |
| BoT-SORT with ReID | ✅ Maintains ID for reappearing players with high reliability |
| Frame-by-frame vs. batch mode | ✅ Batch mode reduced processing time by ~30% |
| ByteTracker (tested separately) | ⚠️ Slightly worse ReID accuracy compared to BoT-SORT in football scenarios |

---

## 🧱 Challenges Encountered



### 1. **Inconsistent Player ID Assignment**
- **Issue**: When multiple players group tightly (e.g., contesting the ball), the tracker occasionally assigns new IDs to the same player.
- **Impact**: Breaks tracking continuity and affects re-identification accuracy.
### 2.**BALL INTERPOLATION**
- **Issue**: Ball tracking sometimes fails to maintain continuity, especially when the ball is occluded.
- **Impact**: Disrupts the smoothness of the ball's trajectory and can lead to incorrect

### 3. **Model Confuses Player vs Goalkeeper**
- **Issue**: The custom-trained YOLOv11 model was trained on a **limited dataset**, insufficient to differentiate subtle features'
- **Symptoms**:
  - Goalkeepers labeled as players
  - Missed goalkeeper detections
- **Impact**: Tracking and tactical analytics (e.g., passes, saves) are negatively affected.

### 4. **GPU resource**
- **Issue**: when the botSORT is selected it requires GPU for the realtime computation.Relying on CPU for botSORT takes more computational time.
- **Impact**: The botSORT is not able to run in real time when the CPU is used.
- **Solution**: ByteTracker is given as tracker option , which takes less computation and not a heavy GPU based.

---

## ⏳ What's Left & Next Steps

If more time/resources were available:

1. **Cross-camera ReID**  
   - Extract features from tracked crops and use a triplet-loss ReID model for camera-invariant embeddings.

2. **Analytics Dashboard**  
   - Create visualizations for:
     - Player heatmaps
     - Pass maps
     - Possession stats

---

## 📁 Output Files

- `output_video.mp4` — Annotated video
- `models/` — ReID weights
- `caches/` — Cached intermediate results

---

## 👨‍💻 Tools & Frameworks

- Python, OpenCV
- Ultralytics YOLOv11(custom model)
- Boxmot(for botSORT)
- Torchreid (for ReID backbone)
- Matplotlib & Seaborn (for analytics, future scope)
-pandas(for ball interpolation)

---

_Authored by: HIMANESH V  
Date: 11th June 2025

