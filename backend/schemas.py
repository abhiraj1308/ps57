from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DetectionClass(str, Enum):
    GHOST_NET = "ghost_net"
    PIPE = "pipe"
    CYLINDER = "cylinder"
    SHIPWRECK = "shipwreck"
    OTHER_DEBRIS = "other_debris"
    UNKNOWN = "unknown"


class DetectionPriority(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class DetectionStatus(str, Enum):
    DETECTED = "detected"
    VALIDATED = "validated"
    REJECTED = "rejected"
    REVIEW_REQUIRED = "review_required"


class BoundingBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)


class Detection(BaseModel):
    detection_id: str
    image_id: str

    class_name: DetectionClass

    ai_confidence: float = Field(ge=0.0, le=1.0)
    final_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0
    )

    bbox: BoundingBox

    priority: Optional[DetectionPriority] = None
    status: DetectionStatus = DetectionStatus.DETECTED

    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)

    estimated_width_m: Optional[float] = Field(default=None, ge=0)
    estimated_height_m: Optional[float] = Field(default=None, ge=0)

    timestamp: Optional[datetime] = None