from dataclasses import dataclass
from typing import Optional


# ============================================================
# PS57 COMMON DATA SCHEMAS
# ============================================================
#
# These classes define the format in which information moves
# between the different PS57 modules.
#
# Pipeline:
#
# SSS preprocessing
#       ↓
# AI detection
#       ↓
# Detection intelligence
#       ↓
# Geospatial
#       ↓
# Backend / Database
#
# The goal is to prevent every module from using a different
# data format.
# ============================================================


# ============================================================
# BOUNDING BOX
# ============================================================

@dataclass
class BoundingBox:
    """
    Bounding box of a detected anomaly in image coordinates.

    x:
        Left/top horizontal pixel coordinate.

    y:
        Top vertical pixel coordinate.

    width:
        Bounding-box width in pixels.

    height:
        Bounding-box height in pixels.
    """

    x: float
    y: float
    width: float
    height: float


# ============================================================
# VISION DETECTION
# ============================================================

@dataclass
class VisionDetection:
    """
    Detection produced by a vision/AI model.

    This is retained for compatibility with the existing
    prototype fusion pipeline.
    """

    class_name: str

    confidence: float

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    width: float = 0.0
    height: float = 0.0

    bbox: Optional[BoundingBox] = None


# ============================================================
# SONAR DETECTION
# ============================================================

@dataclass
class SonarDetection:
    """
    Detection produced by sonar processing.

    range_index:
        Position of the detected target along the sonar range
        axis in the current prototype.

    intensity:
        Measured signal strength of the detected target.

    span_start / span_end:
        Range-bin extent of the target.
    """

    range_index: int

    intensity: float

    span_start: int
    span_end: int

    # Optional fields reserved for the final SSS pipeline.
    x: Optional[float] = None
    y: Optional[float] = None


# ============================================================
# RAW SSS ANOMALY DETECTION
# ============================================================

@dataclass
class SSSAnomalyDetection:
    """
    Detection returned by the Side-Scan Sonar AI model.

    This is the main schema that the final PS57 anomaly
    detection pipeline will use.

    The AI model produces:
        - class
        - confidence
        - bounding box

    Detection Intelligence can then add:
        - final confidence
        - severity
        - decision

    Geospatial can add:
        - latitude
        - longitude
        - position uncertainty
    """

    class_name: str

    confidence: float

    bbox: BoundingBox

    image_id: Optional[str] = None

    # Final intelligence fields.
    severity: Optional[str] = None

    decision: Optional[str] = None

    # Final geographic fields.
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    position_uncertainty_m: Optional[float] = None

    # Processing metadata.
    image_quality: Optional[float] = None
    noise_score: Optional[float] = None


# ============================================================
# FINAL PS57 DETECTION
# ============================================================

@dataclass
class PS57Detection:
    """
    Final unified detection record used by the backend,
    database and dashboard.

    This is the object we ultimately want to persist.
    """

    class_name: str

    confidence: float

    bbox: BoundingBox

    severity: str

    status: str = "new"

    decision: str = "accept"

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    position_uncertainty_m: Optional[float] = None

    image_id: Optional[str] = None

    timestamp: Optional[str] = None


# ============================================================
# LEGACY / PROTOTYPE FUSED DETECTION
# ============================================================
#
# Your existing fusion.py already uses this structure.
# We keep it so the existing prototype remains runnable
# while we transition toward the final SSS architecture.
# ============================================================

@dataclass
class FusedDetection:
    """
    Detection produced by the current prototype fusion layer.

    This preserves compatibility with the existing:
        detection_intelligence/fusion.py
    """

    source: str

    class_name: str

    confidence: float

    priority: str

    latitude: Optional[float] = None
    longitude: Optional[float] = None

    width: float = 0.0
    height: float = 0.0

    sonar_range_index: Optional[int] = None
    sonar_intensity: Optional[float] = None

    fusion_status: str = "vision_only"

    # Optional final-SSS fields.
    bbox: Optional[BoundingBox] = None

    severity: Optional[str] = None

    decision: Optional[str] = None

    position_uncertainty_m: Optional[float] = None

    image_id: Optional[str] = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def bbox_from_xywh(
    x: float,
    y: float,
    width: float,
    height: float,
) -> BoundingBox:
    """
    Convenience function for creating a BoundingBox.
    """

    return BoundingBox(
        x=float(x),
        y=float(y),
        width=float(width),
        height=float(height),
    )


def detection_to_dict(
    detection: PS57Detection,
) -> dict:
    """
    Converts a final PS57Detection into a JSON-friendly
    dictionary.

    Useful for FastAPI responses and database/API adapters.
    """

    return {
        "class_name": detection.class_name,
        "confidence": detection.confidence,

        "bbox": {
            "x": detection.bbox.x,
            "y": detection.bbox.y,
            "width": detection.bbox.width,
            "height": detection.bbox.height,
        },

        "severity": detection.severity,

        "status": detection.status,

        "decision": detection.decision,

        "latitude": detection.latitude,

        "longitude": detection.longitude,

        "position_uncertainty_m": (
            detection.position_uncertainty_m
        ),

        "image_id": detection.image_id,

        "timestamp": detection.timestamp,
    }