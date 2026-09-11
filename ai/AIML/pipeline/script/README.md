# PS57 Stage-2 Sonar Pipeline

This folder contains the Stage-2 utilities for the PS57 side-scan-sonar pipeline.

## Flow

```text
SSS image
   |
   v
Stage-1 YOLO detector
   |
   +--> class / confidence / bounding box
   |
   v
sonar_inference.py
   |
   +--> crop + sonar statistics + shadow/edge features
   |
   v
run_stage2.py
   |
   v
stage2 JSON
   |
   +--> validate_json_stage2.py
   +--> visualize_stage2.py
```

## Files

- `prepare_yolo_dataset.py` - validates/converts an image + YOLO-label dataset and creates a clean YOLO structure.
- `sonar_inference.py` - reusable YOLO inference and sonar-feature extraction.
- `run_stage2.py` - command-line Stage-2 runner.
- `visualize_stage2.py` - draw detections and Stage-2 information on images.
- `validate_json_stage2.py` - validate Stage-2 JSON output.
- `test_sonar_inference.py` - tests for sonar inference utilities.
- `test_unified_runner.py` - integration-style tests for the runner.
- `test_visualize_stage2.py` - visualization tests.

## Installation

```bash
pip install ultralytics opencv-python numpy pytest
```

## Example

```bash
python run_stage2.py \
  --model best.pt \
  --source test/images \
  --output stage2_results
```

Visualize:

```bash
python visualize_stage2.py \
  --json stage2_results/predictions.json \
  --output stage2_results/visualized
```

Validate:

```bash
python validate_json_stage2.py stage2_results/predictions.json
```

## JSON

Each image produces a record containing the source image, image size, detections,
and Stage-2 sonar features. Bounding boxes are stored as pixel coordinates:

`[x1, y1, x2, y2]`

The detector remains the Stage-1 object detector. Stage-2 features are
supporting sonar/shadow information and should not be presented as an independently
trained classifier unless a trained Stage-2 classifier is actually supplied.
