"""
Underwater Marine Debris and Anomaly Detection System
Stage-2 Perception Module: Unified Multi-Model Sonar Inference Interface

PURPOSE:
    Provides the Stage-2 object detection and semantic segmentation inference
    interface for underwater marine debris and anomaly detection.
    Supports Ultralytics YOLO detectors (Crab-Pot, Cylinder Anomaly, Pipeline)
    and PyTorch TorchScript U-Net (Natural Seafloor).

SYSTEM ARCHITECTURE ROLE:
    - Stage-1 (Upstream): Sonar preprocessing, beam normalization, and SNR filtering.
    - Stage-2 (Current):  Raw AI perception on sonar frames. Emits raw AI confidence,
                          normalized bounding boxes (YOLO) or native class segmentation
                          masks and probabilities (U-Net) without fabricating Stage-3 features.
    - Stage-3 (Downstream): Fuses Stage-2 candidates with shadow geometry, temporal ping
                          tracking, and geological context.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from sonar_ai.base import (
    SUPPORTED_EXTENSIONS,
    collect_image_files,
    resolve_device,
)
from sonar_ai.model_inspector import detect_model_type, inspect_model, load_engine
from sonar_ai.runner import UnifiedSonarRunner

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SonarInference")

# ---------------------------------------------------------------------------
# Constants and System Defaults (Preserved for Stage-2 Backwards Compatibility)
# ---------------------------------------------------------------------------
DEFAULT_CRAB_POT_PATH = Path("models/crab_pot_yolo11s_832_best.pt")
DEFAULT_MODEL_PATH = str(DEFAULT_CRAB_POT_PATH) if DEFAULT_CRAB_POT_PATH.is_file() else "crab_pot_yolo11s_best.pt"
DEFAULT_MODEL_VERSION = "yolo11s_crab_pot_baseline"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_CLASS_ID = 0
DEFAULT_CLASS_NAME = "Crab-Pot"
CLASS_MAPPING = {DEFAULT_CLASS_ID: DEFAULT_CLASS_NAME}


# ---------------------------------------------------------------------------
# SonarDetector Facade Class (Preserving Public API & Backwards Compatibility)
# ---------------------------------------------------------------------------
class SonarDetector:
    """
    Stage-2 Inference Engine facade.
    Supports both Ultralytics YOLO object detectors and TorchScript U-Net segmentor.
    Preserves all existing attributes and methods expected by test suites and callers.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        device: Optional[str] = None,
        model_version: str = DEFAULT_MODEL_VERSION,
    ):
        self.model_path = Path(model_path)
        self.conf_threshold = float(conf_threshold)
        self.model_version = str(model_version)
        self.device = resolve_device(device)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Model weights file not found: {self.model_path.resolve()}\n"
                f"Please verify the path: {self.model_path}"
            )

        # Detect whether model is TorchScript U-Net or YOLO
        self.model_type = detect_model_type(self.model_path)
        self.is_unet = self.model_type == "torchscript_unet"

        if self.is_unet:
            from sonar_ai.unet_segmentor import SonarUNetSegmentor
            logger.info(f"Loading TorchScript U-Net model from '{self.model_path}' on device '{self.device}'...")
            self.engine = SonarUNetSegmentor(
                model_path=self.model_path,
                device=self.device,
                model_version=self.model_version,
            )
            self.model = self.engine.model
        else:
            logger.info(f"Loading Ultralytics YOLO model from '{self.model_path}' on device '{self.device}'...")
            try:
                from ultralytics import YOLO
                self.model = YOLO(str(self.model_path))
                logger.info("Model loaded successfully.")
            except ImportError as e:
                logger.critical("Ultralytics package is not installed.")
                raise e
            except Exception as e:
                logger.critical(f"Failed to load YOLO model from '{self.model_path}': {e}")
                raise e

            # Dynamically read model.names without hardcoding class 0 globally
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_mapping = {int(k): str(v) for k, v in self.model.names.items()}
            elif hasattr(self.model, "names") and isinstance(self.model.names, (list, tuple)):
                self.class_mapping = {int(i): str(v) for i, v in enumerate(self.model.names)}
            else:
                self.class_mapping = {DEFAULT_CLASS_ID: DEFAULT_CLASS_NAME}

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        return resolve_device(requested_device)

    def detect_frame(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Execute Stage-2 inference on a single sonar image file.
        For YOLO: Returns detection schema with normalized and pixel bboxes.
        For U-Net: Returns segmentation schema with native class mask and distribution.
        """
        path = Path(image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Sonar image file does not exist: {path.resolve()}")

        if self.is_unet:
            return self.engine.predict_frame(path)

        frame_id = path.stem

        t_start = time.perf_counter()
        results = self.model.predict(
            source=str(path),
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        t_end = time.perf_counter()
        wall_latency_ms = (t_end - t_start) * 1000.0

        if not results or len(results) == 0:
            raise RuntimeError(f"YOLO inference returned no results object for frame: {frame_id}")

        result = results[0]

        if hasattr(result, "orig_shape") and result.orig_shape is not None:
            image_height, image_width = int(result.orig_shape[0]), int(result.orig_shape[1])
        elif hasattr(result, "orig_img") and result.orig_img is not None:
            image_height, image_width = int(result.orig_img.shape[0]), int(result.orig_img.shape[1])
        else:
            from PIL import Image
            with Image.open(path) as img:
                image_width, image_height = img.size

        if hasattr(result, "speed") and isinstance(result.speed, dict) and "inference" in result.speed:
            inference_latency_ms = round(float(result.speed["inference"]), 2)
        else:
            inference_latency_ms = round(wall_latency_ms, 2)

        detections: List[Dict[str, Any]] = []
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            xywhn_list = boxes.xywhn.cpu().numpy()
            xyxy_list = boxes.xyxy.cpu().numpy()
            conf_list = boxes.conf.cpu().numpy()
            cls_list = boxes.cls.cpu().numpy()

            for idx in range(len(boxes)):
                class_id = int(cls_list[idx])
                raw_conf = float(conf_list[idx])

                # Dynamic class lookup: never hardcode class 0 as Crab-Pot for non-crab models
                if hasattr(self, "class_mapping") and class_id in self.class_mapping:
                    class_name = self.class_mapping[class_id]
                elif hasattr(self.model, "names") and isinstance(self.model.names, dict) and class_id in self.model.names:
                    class_name = str(self.model.names[class_id])
                elif hasattr(self.model, "names") and isinstance(self.model.names, (list, tuple)) and 0 <= class_id < len(self.model.names):
                    class_name = str(self.model.names[class_id])
                elif class_id == DEFAULT_CLASS_ID:
                    class_name = DEFAULT_CLASS_NAME
                else:
                    class_name = f"Class_{class_id}"

                det_id = f"{frame_id}_det_{idx + 1:03d}"
                bbox_norm = [round(float(v), 4) for v in xywhn_list[idx].tolist()]
                bbox_pixel = [round(float(v), 2) for v in xyxy_list[idx].tolist()]

                detections.append({
                    "detection_id": det_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "ai_confidence": round(raw_conf, 4),
                    "bbox_xywh_normalized": bbox_norm,
                    "bbox_xyxy_pixel": bbox_pixel,
                })

        return {
            "frame_id": frame_id,
            "model_version": self.model_version,
            "image_width": image_width,
            "image_height": image_height,
            "inference_latency_ms": inference_latency_ms,
            "detections": detections,
        }


# ---------------------------------------------------------------------------
# Batch Processing & Runner Orchestration
# ---------------------------------------------------------------------------
def run_inference(
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
    source_path: Union[str, Path] = "",
    output_path: Union[str, Path] = "sonar_inference_results.json",
    conf_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    device: Optional[str] = None,
    model_version: str = DEFAULT_MODEL_VERSION,
    save_per_frame: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    source_p = Path(source_path)
    if not source_p.exists():
        raise FileNotFoundError(f"Input source path does not exist: {source_p.resolve()}")

    runner = UnifiedSonarRunner(
        model_path=model_path,
        conf_threshold=conf_threshold,
        device=device,
        model_version=model_version,
    )
    return runner.run(
        source_path=source_p,
        output_path=output_path,
        save_per_frame=save_per_frame,
    )


def print_summary(stats: Dict[str, Any]) -> None:
    """
    Print formatted end-of-run execution metrics.
    """
    banner = "=" * 65
    print("\n" + banner)
    print("STAGE-2 SONAR INFERENCE RUN COMPLETE")
    print(banner)
    print(f"Model used            : {stats.get('model_used')}")
    print(f"Model type            : {stats.get('model_type')}")
    print(f"Task                  : {stats.get('task')}")
    print(f"Source                : {stats.get('source')}")
    if stats.get("confidence_threshold") is not None:
        print(f"Confidence threshold  : {stats.get('confidence_threshold')}")
    print(f"Device                : {stats.get('device')}")
    print(f"Images processed      : {stats.get('images_processed')}")
    print(f"Failed images         : {stats.get('failed_images')}")
    print(f"Total detections      : {stats.get('total_detections')}")
    print(f"Output path           : {stats.get('output_path')}")
    print(banner + "\n")


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------
def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-2 Unified Multi-Model Sonar Perception Interface for Underwater "
            "Marine Debris & Anomaly Detection."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to trained model weights (.pt). Supports YOLO detectors & TorchScript U-Net.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a single sonar image or an image directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sonar_inference_results.json",
        help="Path to output JSON file (or directory).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONFIDENCE_THRESHOLD,
        help="Confidence threshold for YOLO object detection (baseline: 0.25).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Inference device: '0' for CUDA GPU, 'cpu' for CPU execution.",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default=DEFAULT_MODEL_VERSION,
        help="Model version tag recorded in frame telemetry.",
    )
    parser.add_argument(
        "--save-per-frame",
        action="store_true",
        help="If set, also saves individual <frame_id>.json files alongside main output.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect model architecture, shapes, and classes before running.",
    )

    return parser


def main() -> int:
    parser = build_cli_parser()
    args = parser.parse_args()

    if not os.path.exists(args.model):
        sys.stderr.write(
            f"ERROR: Model file does not exist: {os.path.abspath(args.model)}\n"
            "Ensure model weight file is available or specify via --model.\n"
        )
        return 1

    if not os.path.exists(args.source):
        sys.stderr.write(
            f"ERROR: Input source does not exist: {os.path.abspath(args.source)}\n"
            "Ensure the image or directory exists.\n"
        )
        return 1

    if args.inspect:
        from sonar_ai.model_inspector import print_inspection_report
        info = inspect_model(args.model)
        print_inspection_report(info)

    try:
        _, stats = run_inference(
            model_path=args.model,
            source_path=args.source,
            output_path=args.output,
            conf_threshold=args.conf,
            device=args.device,
            model_version=args.model_version,
            save_per_frame=args.save_per_frame,
        )
        print_summary(stats)
        return 0

    except Exception as exc:
        logger.critical(f"Inference execution failed: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
