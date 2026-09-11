#!/usr/bin/env python3
"""Cylinder/Manta YOLO inference plus sonar feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


CLASS_NAMES = {
    0: "cylider",
    1: "manta",
}


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[int]
    crop: dict[str, Any]


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def safe_crop(image: np.ndarray, bbox: list[int], pad: int = 20) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0), dtype=image.dtype)

    return image[y1:y2, x1:x2]


def sonar_features(crop: np.ndarray) -> dict[str, Any]:
    if crop.size == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "edge_density": None,
            "shadow_fraction": None,
            "width": 0,
            "height": 0,
        }

    x = crop.astype(np.float32)
    edges = cv2.Canny(crop, 50, 150)
    dark_threshold = float(np.percentile(x, 25))

    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
        "edge_density": float(np.mean(edges > 0)),
        "shadow_fraction": float(np.mean(x <= dark_threshold)),
        "width": int(crop.shape[1]),
        "height": int(crop.shape[0]),
    }


class CylinderMantaInference:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.50,
        imgsz: int = 1280,
    ):
        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def predict_image(
        self,
        image_path: str | Path,
        pad: int = 20,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        gray = load_image(image_path)
        h, w = gray.shape[:2]

        results = self.model.predict(
            source=str(image_path),
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            verbose=False,
        )

        result = results[0]
        names = result.names if hasattr(result, "names") else CLASS_NAMES

        detections = []

        if result.boxes is not None:
            for box in result.boxes:
                cls = int(box.cls.item())
                confidence = float(box.conf.item())

                xyxy = box.xyxy[0].cpu().numpy().tolist()
                bbox = [int(round(v)) for v in xyxy]

                crop = safe_crop(gray, bbox, pad=pad)

                detections.append(
                    Detection(
                        class_id=cls,
                        class_name=str(names.get(cls, CLASS_NAMES.get(cls, cls))),
                        confidence=confidence,
                        bbox=bbox,
                        crop=sonar_features(crop),
                    )
                )

        return {
            "image": image_path.name,
            "source": str(image_path),
            "image_size": {
                "width": w,
                "height": h,
            },
            "detections": [asdict(d) for d in detections],
        }
