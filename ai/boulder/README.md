# SYN6 Boulder Detection Module

## Purpose
Detect boulders/rocks in side-scan sonar imagery.

## Model
YOLO11n

## Main function
detect_boulders(input_image, output_path=None, confidence=0.25)

## Input
PNG, JPG or TIFF sonar image.

## Output
List of detections containing:
- class
- confidence
- bounding box [x1, y1, x2, y2]

## Installation
pip install -r requirements.txt

## Command line
python src/boulder_detector.py path/to/image.png

## Validation Results
Precision: 0.00075
Recall: 0.0146
mAP50: 0.0000277
mAP50-95: 0.0000057

## Known limitation
The current model is experimental and not production-ready.
Many boulder targets are extremely small in the sonar imagery.
