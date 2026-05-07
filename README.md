# FireGuard AI

FireGuard AI is a Streamlit app for detecting fire extinguishers in room and corridor images using a custom-trained YOLOv8 model. The app helps determine whether a scene is compliant by checking if at least one fire extinguisher is detected.

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://fireguardai-sgmqma5vtlnpydinuprpzo.streamlit.app/)

Live app: https://fireguardai-sgmqma5vtlnpydinuprpzo.streamlit.app/

## Live demo

The app is deployed here:

https://fireguardai-sgmqma5vtlnpydinuprpzo.streamlit.app/

## Screenshots

The app uses a dark monitoring-style dashboard with a visible detection results panel, confidence summary, and improved table contrast for object names and confidence values.

If you want to add image examples later, place them under a folder like `assets/` and reference them here in the README.

## What the app does

- Upload an image of a room or corridor
- Adjust the detection confidence threshold
- View a compliance result in real time
- Inspect the annotated image with bounding boxes
- Review individual detection confidence scores in a compact table

## Current UI changes

- Dark, high-contrast dashboard layout with a custom gradient background
- Hero section with quick model, use-case, and status cards
- Sidebar guidance for upload flow, threshold usage, and image quality tips
- Live confidence feedback with a threshold note and status row
- Compliance banner for compliant and non-compliant scenes
- Result dashboard with detection count, highest confidence, and current threshold
- Detection details panel with stronger table visibility for object names and confidence values
- Improved Streamlit widget styling for a cleaner monitoring-style interface

## Current UI features

- Single-page dashboard layout
- Custom sidebar with quick usage instructions and threshold guidance
- Live status badges for readiness, confidence, and detection mode
- Result panel styled like a monitoring dashboard
- Compliance banner for compliant and non-compliant scenes

## Project structure

```text
FireGuardAI/
├── app.py
├── best.pt
├── README.md
├── requirements.txt
└── FireGuardAI/
```

## Model file

The app expects the trained model file to be available at the project root as `best.pt`.

If you trained the model in Google Colab, copy the exported `best.pt` file into:

```text
C:\Users\DELL\OneDrive\Desktop\FireGuardAI\best.pt
```

## Requirements

Install the dependencies listed in `requirements.txt`:

- ultralytics
- streamlit
- pillow
- opencv-python
- numpy

## How to run

1. Open PowerShell in the project folder.
2. Make sure `best.pt` is in the project root.
3. Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

4. Start the app:

```powershell
streamlit run app.py
```

5. Open the local URL shown in the terminal.

## How to use the app

1. Upload a clear room or corridor image.
2. Adjust the confidence threshold.
3. Read the compliance result.
4. Review the annotated image and the detection table.

## Confidence threshold guide

- Lower values detect more objects and may produce more false positives.
- Higher values are stricter and only keep stronger detections.
- A balanced value around 0.5 is a good starting point for testing.

## Training summary

- Model: YOLOv8s
- Training platform: Google Colab
- Task: Fire extinguisher compliance detection
- Output: `best.pt`

## Notes

- Opening `best.pt` in VS Code as text is normal to fail because it is a binary model file.
- The app loads the model directly with Ultralytics YOLO.
- If the file size is correct and the app runs, the model is working properly.

## Status

The app has been updated and tested locally with the current dashboard layout, sidebar guidance, confidence feedback, detection result flow, and readable detection details table.
