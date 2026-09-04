# Combat Participant Tracking & Dropout Analysis

An advanced computer vision and deep learning pipeline designed to track individuals in combat or dynamic scene environments and analyze participant dropout, disengagement, or exit events using state-of-the-art YOLO object detection and tracking architectures.

## Overview

In dynamic multi-person environments—such as athletic competitions, martial arts sparring, or crowd monitoring—detecting presence is only the first step. Understanding when and how participants disengage, withdraw, or leave the active scene ("dropout analysis") provides critical quantitative metrics. This project leverages **YOLO** models for robust real-time bounding box detection, integrated with advanced multi-object tracking (MOT) algorithms to maintain persistent IDs and log spatial-temporal trajectories leading up to exit events.

---

## Key Features

- **Real-Time Detection & Tracking:** Integrates YOLO object detection with deep tracking algorithms to maintain unique participant identities across frames.
- **Dropout & Disengagement Analysis:** Automatically flags when a participant exits the active frame, retreats, or drops out of the active combat/scene boundary.
- **Trajectory & Spatial Mapping:** Records coordinate histories to analyze movement velocity, positioning, and tactical retreat patterns.
- **Modular Data Pipeline:** Exports structured CSV/JSON logs containing timestamps, tracking IDs, bounding box coordinates, and event markers for downstream data analysis.
- **Reproducible & Extensible:** Clean codebase structured for rapid experimentation, custom weight integration, and visualization.

---

## Technical Stack

- **Language:** Python 3.10+
- **Core Framework:** Ultralytics YOLO (YOLOv8/YOLOv11)
- **Data Processing & Analysis:** Pandas, NumPy
- **Computer Vision Utilities:** OpenCV
- **Version Control:** Git

---

## Project Structure

```text
combat-participant-analysis/
├── configs/               # Configuration files
│   └── config.yaml
├── data/                  # Input videos, sample frames, and datasets (gitignored)
│   ├── raw/
│   └── processed/
├── dataset/               # Training/validation dataset metadata
│   ├── CCTV_DATA/         # (gitignored - large video files)
│   ├── NON_CCTV_DATA/     # (gitignored - large video files)
│   └── ground-truth.json  # Dataset annotations
├── experiments/           # Experiment configurations and results
├── models/                # Custom trained YOLO weights (gitignored)
├── outputs/               # Generated logs, plots, and processed videos (gitignored)
├── scripts/               # Utility scripts for data prep and pipeline steps
├── src/                   # Core source code
│   ├── analytics/         # Fight analytics, strike/knockdown detection
│   ├── database/          # Database utilities
│   ├── detection/         # Fighter/person detection
│   ├── features/          # Feature extraction
│   ├── inference/         # Real-time inference
│   ├── learning/          # Temporal model learning
│   ├── mlops/             # Experiment tracking
│   ├── pose/              # Pose estimation
│   ├── preprocessing/     # Video I/O and preprocessing
│   ├── tracking/          # Multi-object tracking
│   ├── utils/             # Utility functions
│   └── visualization/     # Visualization utilities
├── run_fight_analysis.py  # Main entry point
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mmd607/combat-participant-analysis.git
   cd combat-participant-analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights:**
   - `yolov8n.pt` — YOLOv8n person detector (COCO)
   - `yolov8n-pose.pt` — YOLOv8n pose estimation model

   Place these in the `models/` directory.

5. **Download dataset:**
   - Video data should be placed in `dataset/CCTV_DATA/` and `dataset/NON_CCTV_DATA/`
   - See `dataset/ground-truth.json` for annotation format

---

## Usage

1. **Place your input video** inside the `data/` directory.
2. **Run the tracking and analysis pipeline:**
   ```bash
   python run_fight_analysis.py
   ```
3. **Check the outputs:** Processed visualization videos and analytics logs will be saved automatically in the `outputs/` folder.

---

## Configuration

All project-wide settings are defined in `configs/config.yaml`. Override paths and parameters as needed before running.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page or submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
