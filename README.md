# Real Time Person Detection Using CV

Real time person detection system based on computer vision techniques, developed as part of an Electrical Civil Engineering thesis project.  
The project focuses on detecting human presence in monitored environments using image processing and deep learning models, with potential applications in security and access control systems.

---

## Project Overview

This repository contains the implementation and experimental evaluation of a real-time person detection system using computer vision.  
The main objective is to identify and detect people in video streams through the integration of image processing techniques and pretrained YOLO-based models.

The project was developed in the context of an undergraduate thesis, aiming to explore the feasibility of computer vision as a practical solution for person detection in security-oriented environments.

---

## Objectives

- Develop a real-time person detection system using computer vision techniques.
- Evaluate the performance of pretrained YOLO models for human detection.
- Analyze detection behavior under different experimental scenarios.
- Compare and consolidate results from multiple experimental replicas.
- Support future applications in security monitoring and access control systems.

---

## Repository Structure

```bash
PERSON-DETECTION-OPENCV/
│
├── draft/                     # Preliminary scripts and development-stage experiments
├── results/                   # Experimental outputs, performance metrics, and generated results
│
├── analyze_s0.py              # Analysis of scenario S0 results
├── analyze_s1.py              # Analysis of scenario S1 results
├── analyze_s2.py              # Analysis of scenario S2 results
│
├── combine_s0_replicas.py     # Consolidation of scenario S0 experimental replicas
├── combine_s1_replicas.py     # Consolidation of scenario S1 experimental replicas
├── combine_s2_replicas.py     # Consolidation of scenario S2 experimental replicas
│
├── main.py                    # Main execution script for real-time person detection
├── yolov8n.pt                 # YOLOv8 Nano model for efficient person detection
├── yolov8s-pose.pt            # YOLOv8 Pose model for human pose estimation
└── README.md                  # Project documentation
```

---

## Methodology

The system is based on a computer vision pipeline designed for real-time person detection.  
Its workflow includes:

1. **Video acquisition** from a live camera or video source.
2. **Frame preprocessing** to improve detection conditions when necessary.
3. **Model inference** using pretrained YOLO models.
4. **Person detection and localization** through bounding boxes and confidence thresholds.
5. **Optional pose estimation** to support human presence analysis.
6. **Experimental result collection** for later analysis and comparison.

Different experimental scenarios (S0, S1, and S2) were defined to evaluate the system under varying conditions, with multiple replicas performed to ensure consistency and reproducibility.

---

## Models Used

### YOLOv8 Nano (`yolov8n.pt`)
Pretrained lightweight object detection model used as the primary detector.  
It was selected due to its balance between:

- Fast inference speed
- Low computational cost
- Suitability for real-time applications

### YOLOv8 Small Pose (`yolov8s-pose.pt`)
Pretrained pose estimation model used as a complementary approach for human body keypoint detection.  
This model supports:

- Human pose estimation
- Person presence verification
- Enhanced analysis in monitored scenes

---

## Experimental Scenarios

The project includes three main experimental scenarios:

- **S0**: Baseline scenario for initial system evaluation.
- **S1**: Intermediate scenario with modified testing conditions.
- **S2**: Advanced scenario for comparative performance analysis under different constraints.

Each scenario includes multiple experimental replicas, which are later consolidated and analyzed using the provided scripts.

---

## Requirements

Recommended environment:

- MacOS Tahoe 26.3.1
- Python 3.10+
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas
- Matplotlib (optional, for result visualization)

Install dependencies with:

```bash
pip install opencv-python ultralytics numpy pandas matplotlib
```

---

## How to Run

### Main real-time detection system

```bash
python main.py
```

### Result analysis scripts

```bash
python analyze_s0.py
python analyze_s1.py
python analyze_s2.py
```

### Replica consolidation scripts

```bash
python combine_s0_replicas.py
python combine_s1_replicas.py
python combine_s2_replicas.py
```

---

## Current Status

**Project status:** In development

This repository contains the implementation and experimental work associated with an undergraduate thesis project.  
The system is currently under continuous refinement, including performance evaluation, scenario-based testing, and result consolidation.

---

## Potential Applications

This project can be adapted or extended for applications such as:

- Access control systems
- Security monitoring
- Occupancy detection
- Smart surveillance
- Industrial safety monitoring

---

## Future Work

Possible future improvements include:

- Improved robustness under occlusion and lighting variations
- Integration with access control hardware
- Alarm/event triggering system
- Performance optimization for embedded platforms
- Comparative evaluation with additional deep learning models

---

## Author

**Benjamín Becerra**  
Electrical Civil Engineering Student  
Undergraduate Thesis Project

---

## Notes

- The `experiments/` folder contains preliminary and experimental scripts used during development.
- The `results/` folder stores outputs and performance-related results from experimental runs.
- Pretrained model weights are included for testing purposes, but may be replaced or downloaded externally depending on repository size constraints.

---