"""Data models for the marine debris geotagging system."""
from enum import Enum
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class QualityFlag(str, Enum):
    GPS_VALID = "GPS_VALID"
    GPS_INTERPOLATED = "GPS_INTERPOLATED"
    GPS_INVALID = "GPS_INVALID"
    ALTITUDE_VALID = "ALTITUDE_VALID"
    ALTITUDE_INTERPOLATED = "ALTITUDE_INTERPOLATED"
    DEPTH_VALID = "DEPTH_VALID"
    HEADING_INTERPOLATED = "HEADING_INTERPOLATED"
    LAYBACK_APPLIED = "LAYBACK_APPLIED"
    SLANT_RANGE_CORRECTED = "SLANT_RANGE_CORRECTED"


class SonarSystemConfig(BaseModel):
    """Configuration for a specific sonar system."""
    name: str
    frequency_khz: float = Field(gt=0)
    max_range_m: float = Field(gt=0)
    beam_width_deg: float = Field(gt=0, lt=90)
    towfish_depth_offset_m: float = 0.0
    towpoint_offset_stern_m: float = 0.0
    towpoint_offset_stbd_m: float = 0.0
    catenary_factor: float = Field(default=0.90, ge=0.5, le=1.0)
    speed_of_sound_ms: float = Field(default=1500.0, ge=1400.0, le=1600.0)

    @field_validator('catenary_factor')
    @classmethod
    def validate_catenary_factor(cls, v: float) -> float:
        if not (0.5 <= v <= 1.0):
            raise ValueError("Catenary factor must be between 0.5 and 1.0")
        return v


class PingMetadata(BaseModel):
    """Metadata for a single sonar ping."""
    model_config = ConfigDict(from_attributes=True)
    ping_number: int = Field(ge=0)
    timestamp: datetime  # UTC
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    heading_deg: float = Field(default=0.0)
    speed_knots: float = Field(default=0.0, ge=0)
    altitude_m: float = Field(default=0.0, ge=0, description="Height above seafloor")
    depth_m: float = Field(default=0.0, ge=0, description="Depth below surface")
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    heave_m: float = 0.0
    slant_range_m: float = Field(default=0.0, ge=0)
    swath_width_m: float = Field(default=0.0, ge=0)
    cable_out_m: float = Field(default=0.0, ge=0, description="Cable payout for towed systems")
    hdop: Optional[float] = None
    num_samples: int = Field(default=0, ge=0)
    quality_flags: List[QualityFlag] = Field(default_factory=list)
    ship_latitude: Optional[float] = None
    ship_longitude: Optional[float] = None

    @field_validator('heading_deg')
    @classmethod
    def normalize_heading(cls, v: float) -> float:
        return v % 360.0

    @model_validator(mode='after')
    def check_altitude_slant_range(self) -> 'PingMetadata':
        # When both are positive, altitude should normally be less than slant range.
        # We don't enforce this strictly as edge cases exist.
        return self


class Detection(BaseModel):
    """Raw detection from AI model."""
    detection_id: str
    bbox_pixels: List[float] = Field(min_length=4, max_length=4, description="[x_min, y_min, x_max, y_max]")
    confidence: float = Field(ge=0.0, le=1.0)
    class_label: str
    mask: Optional[List[List[float]]] = None  # Optional polygon mask
    ping_range: Optional[List[int]] = Field(default=None, min_length=2, max_length=2, description="[ping_start, ping_end]")

    @field_validator('bbox_pixels')
    @classmethod
    def check_bbox_coords(cls, v: List[float]) -> List[float]:
        x_min, y_min, x_max, y_max = v
        if not (x_min < x_max):
            raise ValueError("x_min must be less than x_max")
        if not (y_min < y_max):
            raise ValueError("y_min must be less than y_max")
        return v


class Geolocation(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    depth_meters: float = Field(ge=0)
    uncertainty_meters: float = Field(ge=0)


class Dimensions(BaseModel):
    length_m: float = Field(ge=0)
    width_m: float = Field(ge=0)
    area_sq_meters: float = Field(ge=0)


class GeolocatedDetection(BaseModel):
    """Detection with geographic coordinates and dimensions."""
    detection_id: str
    timestamp: datetime
    classification: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    bounding_box_pixels: List[float]
    geolocation: Geolocation
    dimensions_meters: Dimensions
    ping_range: List[int]
    quality_flags: List[QualityFlag]
    sonar_image_id: Optional[str] = None


class MissionMetadata(BaseModel):
    survey_id: str
    date: str  # ISO date string
    vessel: str = "Unknown"
    sonar_system: str = "Unknown"
    coordinate_system: str = "WGS84"
    operator: Optional[str] = None
    area_name: Optional[str] = None
    notes: Optional[str] = None


class DetectionSummary(BaseModel):
    total_detections: int = 0
    detections_by_class: Dict[str, int] = Field(default_factory=dict)
    survey_area_sq_km: float = 0.0
    avg_confidence: float = 0.0
    total_pings_processed: int = 0


class SurveyReport(BaseModel):
    """Complete survey report."""
    mission_metadata: MissionMetadata
    detections: List[GeolocatedDetection]
    summary: DetectionSummary
    qa_results: Optional[Dict] = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SonarImageInfo(BaseModel):
    """Sonar image dimensions and resolution."""
    width_pixels: int = Field(gt=0)
    height_pixels: int = Field(gt=0, description="Number of pings")
    samples_per_ping: int = Field(gt=0)
    range_m: float = Field(gt=0, description="Max slant range in meters")
    pixel_size_across_m: Optional[float] = None  # Computed: range / samples_per_ping
    pixel_size_along_m: Optional[float] = None   # Computed from ping spacing
