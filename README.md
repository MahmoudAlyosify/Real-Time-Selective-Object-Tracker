# Real Time Selective Object Tracking

A real-time object detection and manual tracking system.  
Detect objects using **YOLOv3**, then manually **select** any object to track with **CSRT tracker**.

Built for live camera feeds and CCTV systems.

---

## 📽️ Demo Video

[▶️ Watch the demo](https://github.com/user-attachments/assets/24ac0119-9842-43c6-8980-a78e8f96fa2c)

---

## ✅ Features

- Uses **YOLOv3** for fast object detection
- Manual object selection using the mouse
- Tracks selected object using **CSRT** (high accuracy)
- Resilient tracking with automatic reset after loss
- Instructions shown on screen

---

## 📁 Project Files

You need the following files in the same directory as the script:

- `real-time-oelective-object-tracker_yolo.py` → main script
- `yolov3.weights` → YOLOv3 pre-trained weights  
  [Download from GitHub Mirror](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov3.weights)
- `yolov3.cfg` → YOLOv3 configuration file  
  [Download from here](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg)
- `coco.names` → Class labels  
  [Download from here](https://github.com/pjreddie/darknet/blob/master/data/coco.names)


---

## 🔧 Setup

Install required libraries:

```bash
pip install opencv-python opencv-contrib-python numpy
