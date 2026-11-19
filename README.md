# 🥊 Combat Participant Analysis

### Real-Time Detection, Multi-Object Tracking & Participant Dropout Analysis

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-111F68?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Data%20Processing-013243?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Analytics-150458?logo=pandas)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An extensible computer vision pipeline for detecting, tracking, and analyzing participants in dynamic combat and multi-person scenes.

> [!IMPORTANT]
> **Status: design stage.** This repository currently contains the project plan,
> configuration (`configs/config.yaml`), dependency list and folder layout only.
> Detection, tracking, dropout, trajectory, pose and analytics code is **not implemented yet**;
> the features, usage and outputs described below are the planned design, not working functionality.

---

## 📌 Overview

**Combat Participant Analysis** is a computer vision and deep learning project designed to analyze dynamic scenes involving multiple participants.

The system combines **YOLO-based object detection**, **multi-object tracking**, **trajectory analysis**, and **spatial-temporal event analysis** to maintain persistent participant identities and identify important changes in participant presence and movement.

Although the project is motivated by combat and athletic environments, the underlying architecture can be adapted to a broader range of multi-person video analytics applications.

### Example Applications

* 🥊 Combat sports analysis
* 🥋 Martial arts and sparring analysis
* 🏃 Sports participant tracking
* 🎥 Surveillance and CCTV analytics
* 👥 Multi-person scene analysis
* 🚪 Participant exit and disappearance detection
* 📍 Spatial movement analysis
* 📊 Video-based behavioral analytics

---

# 🎯 Problem Statement

Detecting a person in a video is only the first step.

In a dynamic environment, an analysis system also needs to answer questions such as:

* Who is currently present?
* Is the same participant still visible?
* Where has the participant moved?
* How long has the participant remained active?
* Did the participant leave the active scene?
* Where did the participant exit?
* What was the participant's movement trajectory before disappearing?

This project addresses these questions by combining object detection with persistent identity tracking and spatial-temporal analysis.

---

# ✨ Key Features

### 🔍 Person Detection

Uses YOLO-based object detection models to identify participants in individual video frames.

### 🎯 Multi-Object Tracking

Maintains persistent tracking identities across consecutive frames, allowing the system to distinguish between multiple participants.

### 🆔 Persistent Participant IDs

Each detected participant can be associated with a tracking ID that remains consistent throughout the available tracking sequence.

### 🚪 Dropout / Exit Detection

Analyzes participant visibility and movement to identify when a participant leaves the active scene or becomes inactive according to configured conditions.

### 📍 Trajectory Analysis

Stores spatial information from tracked bounding boxes and uses it to reconstruct participant movement trajectories.

### ⏱️ Temporal Analysis

Tracks participant states and events over time rather than treating each video frame independently.

### 📊 Structured Analytics

Produces structured data containing tracking information, coordinates, timestamps, and event markers for downstream analysis.

### 🎥 Visualization

Supports visualization of detections, tracking identities, trajectories, and processed video outputs.

### ⚙️ Configurable Pipeline

Project-wide parameters are centralized in a YAML configuration file, allowing experiments and inference settings to be changed without modifying core source code.

---

# 🧠 System Architecture

The overall processing pipeline follows a modular computer vision architecture:

```text
                     ┌─────────────────┐
                     │    Input Video  │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Video Processing│
                     │ & Preprocessing │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ YOLO Detection  │
                     │ Person / Object │
                     │   Detection     │
                     └────────┬────────┘
                              │
                              ▼
                     ┌─────────────────┐
                     │ Multi-Object    │
                     │    Tracking     │
                     └────────┬────────┘
                              │
                 ┌────────────┼────────────┐
                 │            │            │
                 ▼            ▼            ▼
          ┌────────────┐ ┌───────────┐ ┌─────────────┐
          │ Participant│ │Trajectory │ │ Temporal /  │
          │   State    │ │ Analysis  │ │ Event Data  │
          └─────┬──────┘ └─────┬─────┘ └──────┬──────┘
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                     ┌─────────────────┐
                     │ Combat / Scene  │
                     │    Analytics    │
                     └────────┬────────┘
                              │
                 ┌────────────┼────────────┐
                 ▼            ▼            ▼
            ┌────────┐  ┌──────────┐  ┌──────────┐
            │ Video  │  │   CSV    │  │   JSON   │
            │ Output │  │  Logs    │  │  Events  │
            └────────┘  └──────────┘  └──────────┘
```

---

# 🔎 Detection & Tracking

The detection stage uses **Ultralytics YOLO** models to identify people within video frames.

The resulting detections are passed to the tracking stage, where the system maintains participant identities across frames.

Conceptually:

```text
Frame N
   │
   ├── Person → ID 1
   └── Person → ID 2
          │
          ▼
Frame N+1
   │
   ├── Person → ID 1
   └── Person → ID 2
          │
          ▼
Frame N+2
   │
   ├── Person → ID 1
   └── Person → ID 2
```

This persistent identity information forms the foundation for trajectory and participant-state analysis.

---

# 🚪 Participant Dropout Analysis

One of the primary goals of this project is identifying participant disappearance, withdrawal, or exit events.

Instead of simply asking:

> "Is a person detected in this frame?"

the system can analyze a participant across time:

```text
ACTIVE
  │
  ▼
TRACKED
  │
  ▼
MOVING
  │
  ▼
APPROACHING BOUNDARY
  │
  ▼
VISIBILITY DECREASE
  │
  ▼
EXIT / DROPOUT EVENT
```

A dropout event can then be associated with:

* Tracking ID
* Frame number
* Timestamp
* Last known bounding box
* Spatial position
* Previous trajectory
* Event type
* Confidence / detection information where available

This provides a foundation for higher-level participant behavior analysis.

---

# 📍 Trajectory Analysis

For every tracked participant, spatial information can be collected over time.

A simplified trajectory can be represented as:

```text
                 Participant #1

                       ●
                    ●
                 ●
              ●
           ●
        ●
     ●
  ●─────────────────────────────►
 Start                         Exit
```

The trajectory data can be used to calculate or analyze:

* Participant position
* Movement direction
* Displacement
* Distance traveled
* Temporal movement patterns
* Boundary interactions
* Exit location
* Pre-exit movement

These features can later be used for more advanced behavioral or tactical analysis.

---

# 📊 Analytics Data

The pipeline is designed to produce structured data suitable for further analysis.

A conceptual tracking record may contain:

```json
{
  "frame": 120,
  "timestamp": 4.80,
  "track_id": 2,
  "bbox": {
    "x1": 412,
    "y1": 183,
    "x2": 563,
    "y2": 487
  },
  "event": "active"
}
```

Depending on the configured pipeline, additional information can be stored for downstream analytics.

Supported output formats include:

* CSV
* JSON
* Processed video
* Visualization outputs
* Analytical plots

---

# 🥊 Combat Analytics

The repository is structured to support higher-level combat analysis beyond basic detection and tracking.

The `src/analytics/` module provides a dedicated location for combat-oriented analytical functionality such as:

* Participant activity analysis
* Strike-related analysis
* Knockdown-related analysis
* Movement analysis
* Participant state analysis
* Event aggregation

The architecture is intentionally modular so additional temporal and behavioral models can be integrated without redesigning the entire pipeline.

---

# 🧍 Pose Estimation

The project also provides a dedicated pose-processing module.

Pose estimation can provide additional information about participant body configuration and movement.

Potential applications include:

* Body keypoint extraction
* Movement analysis
* Posture analysis
* Action recognition
* Combat event detection
* Motion feature extraction

Pose functionality can therefore complement bounding-box-based tracking when finer-grained participant analysis is required.

---

# 🗂️ Dataset

The project supports a dataset organization designed around video-based participant analysis.

```text
dataset/
├── CCTV_DATA/
├── NON_CCTV_DATA/
└── ground-truth.json
```

### CCTV_DATA

Contains CCTV-style video data for testing and developing tracking and participant-analysis methods.

### NON_CCTV_DATA

Contains additional non-CCTV video sources to improve robustness across different visual environments.

### ground-truth.json

Contains dataset annotation metadata following the project's annotation format.

> Large video files and dataset assets should remain outside Git history and be managed through the project's data workflow.

---

# 🏗️ Project Structure

```text
combat-participant-analysis/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── dataset/
│   ├── CCTV_DATA/
│   ├── NON_CCTV_DATA/
│   └── ground-truth.json
│
├── experiments/
│
├── models/
│
├── outputs/
│
├── scripts/
│
├── src/
│   ├── analytics/
│   │   └── Combat and event analytics
│   │
│   ├── database/
│   │   └── Database utilities
│   │
│   ├── detection/
│   │   └── Participant detection
│   │
│   ├── features/
│   │   └── Feature extraction
│   │
│   ├── inference/
│   │   └── Inference pipeline
│   │
│   ├── learning/
│   │   └── Temporal model learning
│   │
│   ├── mlops/
│   │   └── Experiment and model management
│   │
│   ├── pose/
│   │   └── Pose estimation
│   │
│   ├── preprocessing/
│   │   └── Video preprocessing
│   │
│   ├── tracking/
│   │   └── Multi-object tracking
│   │
│   ├── utils/
│   │   └── Shared utilities
│   │
│   └── visualization/
│       └── Visualization tools
│
├── run_fight_analysis.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

# ⚙️ Technical Stack

| Category             | Technology       |
| -------------------- | ---------------- |
| Programming Language | Python 3.10+     |
| Object Detection     | Ultralytics YOLO |
| Pose Estimation      | YOLO Pose        |
| Computer Vision      | OpenCV           |
| Numerical Computing  | NumPy            |
| Data Analysis        | Pandas           |
| Configuration        | YAML             |
| Version Control      | Git              |

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/mmd607/combat-participant-analysis.git
cd combat-participant-analysis
```

## 2. Create a Virtual Environment

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🧠 Model Weights

Place the required model weights inside the `models/` directory.

Example:

```text
models/
├── yolov8n.pt
└── yolov8n-pose.pt
```

### YOLOv8n

Used as a lightweight pretrained object detector.

### YOLOv8n-Pose

Used for pose estimation when body keypoint information is required.

> Model files are intentionally excluded from Git when they are large. Download the appropriate weights separately and place them in the configured model directory.

---

# 📁 Data Setup

Place input videos according to the configured directory structure.

```text
data/
├── raw/
└── processed/
```

Dataset-specific video assets can be organized under:

```text
dataset/
├── CCTV_DATA/
└── NON_CCTV_DATA/
```

Large video datasets should not be committed directly to the repository.

---

# ▶️ Usage

After configuring the environment and model paths, run:

```bash
python run_fight_analysis.py
```

The pipeline processes the configured input video and generates the corresponding analytical outputs.

Generated results are stored under:

```text
outputs/
```

Depending on the configuration and enabled pipeline components, this may include:

```text
outputs/
├── videos/
├── logs/
├── plots/
└── analytics/
```

---

# 🔧 Configuration

Project-wide configuration is centralized in:

```text
configs/config.yaml
```

This allows important parameters to be adjusted without changing the core implementation.

Typical configuration areas include:

* Input video paths
* Output directories
* Model paths
* Detection thresholds
* Tracking parameters
* Visualization settings
* Analytics parameters
* Dataset locations

Example structure:

```yaml
model:
  path: models/yolov8n.pt

input:
  path: data/raw/input.mp4

output:
  directory: outputs/

detection:
  confidence: 0.5
```

> The exact available configuration keys depend on the current implementation of `config.yaml`.

---

# 📈 Output & Visualization

The project is designed to produce both machine-readable and human-readable outputs.

### Processed Video

Visualizes detections and tracking information directly on the original video.

### Tracking Logs

Stores participant tracking information for further analysis.

### Event Logs

Records important participant state changes and detected events.

### Trajectory Visualization

Provides a spatial representation of participant movement.

Example conceptual visualization:

```text
Y
│
│       ●
│      ●
│     ●
│   ●
│  ●
│ ●
│●
└──────────────────────────── X
```

---

# 🧪 Experiments

The `experiments/` directory is intended to keep experiment-specific configurations and results organized.

A recommended experiment workflow is:

```text
Dataset
   │
   ▼
Preprocessing
   │
   ▼
Detection
   │
   ▼
Tracking
   │
   ▼
Feature Extraction
   │
   ▼
Analytics
   │
   ▼
Evaluation
```

This structure makes it easier to compare different model configurations and processing strategies.

---

# 📊 Evaluation

The system can be evaluated at multiple levels.

## Detection

Potential metrics:

* Precision
* Recall
* mAP
* Detection confidence
* FPS

## Tracking

Potential metrics:

* ID consistency
* ID switches
* Track fragmentation
* Track duration
* MOTA / MOT metrics where applicable

## Dropout Analysis

Potential metrics:

* Dropout precision
* Dropout recall
* Event detection accuracy
* False dropout rate
* Detection latency

## Performance

Additional system-level metrics can include:

* Processing FPS
* Inference latency
* CPU/GPU utilization
* Memory consumption

> Benchmark values should be added after running reproducible experiments on the project's target datasets.

---

# 🔬 Development Roadmap

The modular architecture allows the project to evolve from participant tracking into a broader combat video intelligence system.

### Current / Core Capabilities

* [x] YOLO-based participant detection
* [x] Multi-person tracking architecture
* [x] Participant identity tracking
* [x] Trajectory collection
* [x] Dropout / exit analysis architecture
* [x] Structured output pipeline
* [x] Configurable processing pipeline

### Planned Extensions

* [ ] Advanced tracking benchmarks
* [ ] Robust boundary-based exit detection
* [ ] Automated trajectory metrics
* [ ] Pose-based movement analysis
* [ ] Strike detection
* [ ] Knockdown detection
* [ ] Action recognition
* [ ] Fighter activity scoring
* [ ] Temporal deep learning models
* [ ] Real-time analytics dashboard
* [ ] Automated experiment tracking
* [ ] Comprehensive benchmark suite

---

# 🧩 Design Principles

The project follows several engineering principles.

### Modularity

Detection, tracking, preprocessing, pose estimation, analytics, and visualization are separated into dedicated modules.

### Reproducibility

Configuration files and experiment directories provide a consistent structure for reproducing experiments.

### Extensibility

New detection models, tracking approaches, temporal models, and analytics modules can be integrated without rewriting the entire pipeline.

### Separation of Concerns

Computer vision processing, analytics, visualization, and data handling remain logically separated.

### Data-Driven Development

The pipeline is designed around structured tracking and event data, enabling future statistical and machine learning analysis.

---

# 🛠️ Extending the Project

A new detection model can be integrated through the detection layer without changing the higher-level analytics pipeline.

Similarly, a new tracking algorithm can be implemented inside:

```text
src/tracking/
```

while maintaining the same downstream participant representation.

Additional analytics can be implemented inside:

```text
src/analytics/
```

and visualization components inside:

```text
src/visualization/
```

This makes the repository suitable for iterative computer vision research and experimentation.

---

# 🤝 Contributing

Contributions, improvements, issues, and feature requests are welcome.

A typical contribution workflow:

```bash
git checkout -b feature/my-feature
```

Make your changes, test them, and commit:

```bash
git add .
git commit -m "Add new tracking feature"
```

Push the branch:

```bash
git push origin feature/my-feature
```

Then open a Pull Request.

---

# 📌 Project Status

This project is under active development.

The current architecture focuses on building a modular foundation for:

**Detection → Tracking → Trajectory Analysis → Participant State Analysis → Combat Analytics**

Additional computer vision and temporal-learning capabilities can be integrated as the project evolves.

---

# 📚 References & Technologies

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* [OpenCV](https://opencv.org/)
* [NumPy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [Python](https://www.python.org/)

---

# 📄 License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

---

# 👤 Author

**Mohammad Homayian**

Computer Engineering Student
Artificial Intelligence • Computer Vision • Machine Learning

GitHub: [@mmd607](https://github.com/mmd607)

---

<div align="center">

### 🥊 Combat Participant Analysis

**Detect · Track · Analyze · Understand**

Built for computer vision experimentation, video analytics, and intelligent combat-scene understanding.

⭐ If you find this project useful, consider giving it a star.

</div>
