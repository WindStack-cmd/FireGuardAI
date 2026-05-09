# FireGuard AI 🧯

Automated Fire Extinguisher Compliance Detection System

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-111827?style=for-the-badge)](https://docs.ultralytics.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/cloud)
[![License](https://img.shields.io/badge/License-Proprietary-orange?style=for-the-badge)](#license)

## Live Links

- Live demo: https://fireguardai-kb7caahuauzqhnbqm3mmzb.streamlit.app/
- GitHub: https://github.com/WindStack-cmd/FireGuardAI

## Overview

FireGuard AI is a workplace safety compliance solution that detects fire extinguishers in room and corridor images using a custom-trained YOLOv8s model. The web application helps users upload images, run automated checks, and receive a clear COMPLIANT or NON-COMPLIANT result with annotated bounding boxes. A companion YOLOvX mobile workflow supports live camera-based detection for real-world field usage.

The project was built for German and global workplace safety monitoring where fire extinguishers must remain visible, accessible, and properly maintained.

## Key Features

- 🔍 Detects fire extinguishers in images using a YOLOv8s computer vision model.
- 📤 Upload an image and receive an instant compliance verdict.
- ✅ Displays COMPLIANT and NON-COMPLIANT status with clear visual feedback.
- 📦 Shows bounding boxes and confidence scores for each detected object.
- 📱 Supports a second deployment path through the YOLOvX mobile app.
- 🌍 Designed for workplace safety monitoring in Germany and worldwide.
- ⚡ Built with a professional dashboard-style Streamlit interface.
- 🛡️ Reduces manual inspection effort and enables 24/7 automated monitoring.

## Results

### Model Performance

| Metric | Value |
| --- | ---: |
| mAP@50 | 95% |
| Precision | 98% |
| Recall | 95% |
| mAP@50-95 | 80% |
| Inference Speed | 42 ms |

### Training Outcome

| Item | Value |
| --- | --- |
| Model | YOLOv8s (Ultralytics) |
| Training Platform | Google Colab with Tesla T4 GPU |
| Deployment Readiness | Streamlit web + YOLOvX mobile |
| Primary Use Case | Fire extinguisher compliance detection |

## Dataset

| Dataset Item | Details |
| --- | --- |
| Total Images | 1,188 |
| Personal Real-World Images | 90 college corridor photos |
| Augmented Dataset Size | 2,850 images |
| Train / Val / Test Split | 831 / 236 / 121 |
| Primary Sources | Roboflow Universe + personal collection |
| Annotation Method | Auto-labeled using SAM3 |

The dataset was designed to improve robustness across corridor, room, and indoor inspection scenarios.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Model | YOLOv8s (Ultralytics) |
| Web Framework | Streamlit |
| Mobile App | YOLOvX |
| Training Platform | Google Colab |
| Dataset Platform | Roboflow |
| Deployment | Streamlit Community Cloud |
| Version Control | GitHub with Git LFS |

## Deployment Methods

### 1) Web App - Streamlit

Use the Streamlit deployment to upload room or corridor images, run compliance checks, and inspect detection results.

- Upload an image from your device.
- Adjust the confidence threshold.
- Review the compliance verdict.
- Inspect the annotated image with bounding boxes.

Live demo: https://fireguardai-kb7caahuauzqhnbqm3mmzb.streamlit.app/

**Screenshot placeholder**

> Replace this block with a project screenshot when available.

```text
[Web App Screenshot Placeholder]
```

### 2) Mobile App - YOLOvX

The mobile deployment is intended for real-time live camera detection and field use.

- Live camera detection at 7.6 FPS.
- 83 ms inference.
- Detects multiple fire extinguishers simultaneously.
- Displays confidence scores during live monitoring.

**Screenshot placeholder**

> Replace this block with a mobile app screenshot when available.

```text
[YOLOvX Mobile App Screenshot Placeholder]
```

## How to Use the YOLOvX App

1. Open the YOLOvX mobile app on your device.
2. Grant camera permissions when prompted.
3. Point the camera at a corridor, room, or inspection area.
4. Wait for the live model overlay to identify fire extinguishers.
5. Review confidence scores for each detection.
6. Move around the scene to verify visibility and placement.
7. Use the app as a fast on-site inspection aid.

## Installation

### Prerequisites

- Python 3.11
- Git
- PowerShell or another terminal

### Local Setup

1. Clone the repository.

```powershell
git clone https://github.com/WindStack-cmd/FireGuardAI.git
cd FireGuardAI
```

2. Create and activate a virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install the Python dependencies.

```powershell
pip install -r requirements.txt
```

4. Make sure the trained model file is available at the project root.

```text
best.pt
```

5. Launch the Streamlit app.

```powershell
streamlit run app.py
```

6. Open the local URL shown in the terminal.

## How It Works

1. The user uploads an image of a room or corridor.
2. The YOLOv8s model scans the scene for fire extinguishers.
3. Detected objects are highlighted with bounding boxes.
4. The app compares detection confidence with the selected threshold.
5. A compliance status is shown immediately.

## Threshold Guide

- Lower thresholds detect more objects and may include more false positives.
- Higher thresholds are stricter and only keep stronger detections.
- A threshold around 0.5 is a practical starting point.

## Project Structure

```text
FireGuardAI/
├── .gitattributes
├── .gitignore
├── .streamlit/
│   └── config.toml
├── ENHANCEMENTS.md
├── FireGuardAI/
├── app.py
├── best.pt
├── packages.txt
├── README.md
├── requirements.txt
└── runtime.txt
```

## Repository Notes

- `best.pt` is a binary model file and will not open as readable text.
- `packages.txt` contains Streamlit Cloud system dependencies.
- `runtime.txt` defines the Python version used during deployment.
- The app loads the model through Ultralytics YOLO at runtime.

## Acknowledgements

- Ultralytics for YOLOv8.
- Streamlit for the web application framework and cloud hosting.
- Roboflow for dataset tooling and workflow support.
- Google Colab for training infrastructure.
- St. John College of Engineering and Management (SJCEM) for academic support.
- The FireGuard AI internship context: Vision Technology Internship - YOLOvX Based Real World Object Detection.

## Developer

- Developer: Pratik Yadav
- Institution: St. John College of Engineering and Management (SJCEM)
- Batch: UC62 - Fire Extinguisher Detection

## License

This project is provided for portfolio and evaluation purposes. If you plan to reuse the code or model assets, please contact the developer first.

## Contact / Connect

- GitHub: https://github.com/WindStack-cmd/FireGuardAI
- Live demo: https://fireguardai-kb7caahuauzqhnbqm3mmzb.streamlit.app/

For collaboration, internship, or recruitment opportunities, connect through GitHub and reference the FireGuard AI project name in your message.
