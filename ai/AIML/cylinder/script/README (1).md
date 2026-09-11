# PS57 Cylinder + Manta Stage-2 Scripts

These scripts are the Cylinder/Manta equivalent of the PS57 Stage-2 utilities.

## Model

- Model family: YOLO26
- Task: object detection
- Input: 1280 x 1280
- Classes:
  - 0 = cylider
  - 1 = manta
- Dataset: Cylinder2, Roboflow Version 6

IMPORTANT: the class spelling `cylider` is intentional and matches the supplied dataset mapping.

## Flow

```text
SSS image
   |
   v
Cylinder/Manta YOLO detector
   |
   +--> class / confidence / bounding box
   |
   v
sonar_inference.py
   |
   +--> padded crop + sonar statistics + shadow/edge features
   |
   v
run_stage2.py
   |
   v
predictions.json
   |
   +--> validate_json_stage2.py
   +--> visualize_stage2.py
```

## Installation

```bash
pip install ultralytics opencv-python numpy pytest
```

## Run

```bash
python run_stage2.py \
  --model Cylinder.pt \
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

## Important

The detector is the trained Cylinder/Manta YOLO model. The sonar/shadow values
produced here are descriptive features; they are NOT a separately trained
false-positive classifier.

Use the actual trained `.pt` file when running inference.
