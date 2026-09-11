# Underwater Marine Debris and Anomaly Detection System
## Stage-2 Perception Module: Unified Multi-Model Evaluation Runner

### System Architecture Overview
The system operates in three decoupled stages:
- **Stage 1 (Upstream)**: Sonar preprocessing, beam normalization, and SNR/quality filtering.
- **Stage 2 (Current)**: Raw AI perception. Executes object detection and semantic segmentation on sonar imagery. Emits raw AI confidence, normalized bounding boxes, and frame telemetry (for YOLO) or native class masks, class area distributions, lossless Run-Length Encoding (RLE), and derived polygon contours (for U-Net). Does **NOT** fabricate downstream Stage-3 features.
- **Stage 3 (Downstream)**: Consumes Stage-2 normalized telemetry and fuses candidate detections with acoustic shadow geometry, temporal ping-to-ping tracking, seafloor geological context, and false-positive/false-negative error analysis.

---

### Local Model Registry

| Model File | Architecture / Type | Task | Input Dimensions | Output Dimensions | Class Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `models/crab_pot_yolo11s_832_best.pt` | Ultralytics YOLO11s | Detection | 832 × 832 × 3 | Bounding boxes + scores | `{0: "Crab-Pot"}` |
| `models/Cylinder.pt` | Ultralytics YOLO | Detection | 640 × 640 × 3 | Bounding boxes + scores | `{0: "marine_man_made_anomaly"}` |
| `models/pipeline_yolo26n.pt` | Ultralytics YOLO | Detection | 640 × 640 × 3 | Bounding boxes + scores | `{0: "Pipeline"}` |
| `models/PS57_Natural_Seafloor_UNet_best.pt` | PyTorch TorchScript U-Net | Semantic Segmentation | 1 × 1 × 64 × 256 (Grayscale) | 1 × 3 × 64 × 256 (Softmax) | `{0: "rock", 1: "sand", 2: "others"}` |

> [!NOTE]
> - **Dynamic Class Mapping**: Class IDs and names are resolved dynamically from `model.names` for YOLO models. Class 0 is **never** globally hardcoded to "Crab-Pot".
> - **U-Net Precision**: The TorchScript U-Net is loaded via `torch.jit.load(..., map_location=device)` without Keras or TensorFlow dependencies. Grayscale inputs are normalized and resized to (height=64, width=256). Per-pixel softmax distributions are converted to the native class mask via `argmax`.

---

### Model Inspection Commands

Inspect all models registered in `models/`:
```bash
python inspect_models.py
```

Inspect a specific model file:
```bash
python inspect_models.py --model models/Cylinder.pt
```

Inline inspection before inference:
```bash
python sonar_inference.py --model models/pipeline_yolo26n.pt --source "AI4Shipwrecks/test/images" --inspect
```

---

### Exact Inference Commands for All Four Models

#### 1. Crab-Pot YOLO Detector (`crab_pot_yolo11s_832_best.pt`)
```bash
python sonar_inference.py \
    --model models/crab_pot_yolo11s_832_best.pt \
    --source "AI4Shipwrecks/test/images" \
    --output results/crab_pot_eval.json \
    --conf 0.25 \
    --device cpu \
    --model-version yolo11s_crab_pot_baseline
```

#### 2. Cylinder Anomaly YOLO Detector (`Cylinder.pt`)
```bash
python sonar_inference.py \
    --model models/Cylinder.pt \
    --source "AI4Shipwrecks/test/images" \
    --output results/cylinder_eval.json \
    --conf 0.25 \
    --device cpu \
    --model-version cylinder_yolo_v1
```

#### 3. Pipeline YOLO Detector (`pipeline_yolo26n.pt`)
```bash
python sonar_inference.py \
    --model models/pipeline_yolo26n.pt \
    --source "AI4Shipwrecks/test/images" \
    --output results/pipeline_eval.json \
    --conf 0.25 \
    --device cpu \
    --model-version pipeline_yolo26n_v1
```

#### 4. Natural Seafloor TorchScript U-Net Segmentor (`PS57_Natural_Seafloor_UNet_best.pt`)
```bash
python sonar_inference.py \
    --model models/PS57_Natural_Seafloor_UNet_best.pt \
    --source "AI4Shipwrecks/test/images" \
    --output results/seafloor_unet_eval.json \
    --device cpu \
    --model-version seafloor_unet_ts_v1
```

---

### Output JSON Schema Specifications

#### Detection Schema (YOLO Models)
```json
{
  "model_version": "yolo11s_crab_pot_baseline",
  "metadata": {
    "model_file": "crab_pot_yolo11s_832_best.pt",
    "model_type": "Ultralytics YOLO (DetectionModel)",
    "task": "detection",
    "source": "/path/to/source",
    "confidence_threshold": 0.25,
    "device": "cpu",
    "total_images": 10,
    "images_processed": 10,
    "failed_images_count": 0,
    "total_detections": 12,
    "generated_at": "2026-09-10T08:30:00Z"
  },
  "failed_images": [],
  "frames": [
    {
      "frame_id": "sonar_ping_001",
      "model_version": "yolo11s_crab_pot_baseline",
      "image_width": 640,
      "image_height": 512,
      "inference_latency_ms": 12.5,
      "detections": [
        {
          "detection_id": "sonar_ping_001_det_001",
          "class_id": 0,
          "class_name": "Crab-Pot",
          "ai_confidence": 0.84,
          "bbox_xywh_normalized": [0.45, 0.55, 0.12, 0.18],
          "bbox_xyxy_pixel": [250.0, 235.0, 326.8, 327.2]
        }
      ]
    }
  ]
}
```
*Zero-Detection Frames*: If no anomalies or targets are detected above the confidence threshold, the frame is preserved with `"detections": []` for downstream false-negative auditing.

#### Segmentation Schema (TorchScript U-Net)
```json
{
  "model_version": "seafloor_unet_ts_v1",
  "metadata": {
    "model_file": "PS57_Natural_Seafloor_UNet_best.pt",
    "model_type": "PyTorch TorchScript U-Net",
    "task": "semantic segmentation",
    "source": "/path/to/source",
    "confidence_threshold": null,
    "device": "cpu",
    "total_images": 10,
    "images_processed": 10,
    "failed_images_count": 0,
    "total_detections": null,
    "generated_at": "2026-09-10T08:30:00Z"
  },
  "failed_images": [],
  "frames": [
    {
      "frame_id": "sonar_ping_001",
      "model_version": "seafloor_unet_ts_v1",
      "image_width": 1728,
      "image_height": 2476,
      "inference_latency_ms": 18.2,
      "segmentation": {
        "model_input_shape": [1, 1, 64, 256],
        "mask_shape": [64, 256],
        "num_classes": 3,
        "class_names": {
          "0": "rock",
          "1": "sand",
          "2": "others"
        },
        "class_distribution": {
          "rock": {
            "class_id": 0,
            "pixel_count": 13962,
            "area_percentage": 85.22,
            "mean_softmax_probability": 0.8914
          },
          "sand": {
            "class_id": 1,
            "pixel_count": 2392,
            "area_percentage": 14.60,
            "mean_softmax_probability": 0.8421
          },
          "others": {
            "class_id": 2,
            "pixel_count": 30,
            "area_percentage": 0.18,
            "mean_softmax_probability": 0.6125
          }
        },
        "mask_rle": {
          "rock": "...",
          "sand": "...",
          "others": "..."
        },
        "polygons": [
          {
            "class_id": 0,
            "class_name": "rock",
            "area_pixels": 13962.0,
            "contour_points_count": 48,
            "polygon_normalized": [[0.0, 0.0], [1.0, 0.0], ...]
          }
        ]
      }
    }
  ]
}
```

---

### Fault Isolation (`failed_images`)
If an image file is corrupted, unreadable, or encounters an unexpected exception, the runner catches the error, logs it, and records the incident in `failed_images`:
```json
{
  "failed_images": [
    {
      "file_name": "corrupted_sonar_ping.png",
      "file_path": "/full/path/to/corrupted_sonar_ping.png",
      "error": "UnidentifiedImageError: cannot identify image file"
    }
  ]
}
```
Batch processing continues uninterrupted for all other images.

---

### Unit & Integration Testing
To run the automated test suite verifying all 16 test cases:
```bash
python -m unittest test_sonar_inference.py
```
