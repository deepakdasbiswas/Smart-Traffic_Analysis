# Smart Traffic Analysis & Speed Estimation System

A computer vision Proof of Concept (PoC) built to automate road monitoring. This system processes recorded traffic footage to detect, track, classify, and estimate the speed of vehicles. It includes congestion heatmaps and automated data logging for city planning analytics.

---

##  Features

The system is broken down into modular demonstration scripts, each highlighting a specific capability:

1. **Real-Time Detection & Tracking (`demo_speed.py`)** - Utilizes YOLOv8 to detect cars, trucks, buses, and motorcycles.
   - Assigns persistent IDs using Ultralytics tracking solutions.
   - Estimates vehicle speed dynamically as they cross a designated virtual ROI (Region of Interest).

2. **Traffic Density Heatmaps (`demo_heatmap.py`)**
   - Generates a dynamic, color-coded thermal map over the video feed.
   - Visualizes congestion zones, braking areas, and road bottlenecks to aid in infrastructure planning.

3. **Automated Volume Logging (`demo_data.py`)**
   - Counts vehicles passing through specific zones.
   - Automatically exports live traffic flow data into a structured `traffic_report.csv` file for external analysis.

---

##  The Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3** | Core application logic and data handling. |
| **Ultralytics YOLOv8** | Pre-trained model (`yolov8n.pt`) for fast, lightweight object detection and tracking. |
| **OpenCV** | Video frame extraction, manipulation, and rendering visual overlays (bounding boxes, lines, text). |
| **CSV Module** | Structured data export and logging. |

---

##  How It Works (The Pipeline)

```text
Raw Video Frame → YOLOv8 Inference → Centroid Tracking → Distance/Time Speed Math → Visual Overlay → CSV Export