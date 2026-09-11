from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from src.models import MissionMetadata, Detection, SonarImageInfo

class UploadSonarRequest(BaseModel):
    """Metadata accompanying a sonar file upload."""
    survey_id: str = Field(..., json_schema_extra={"example": "SSS_2024_001"})
    sonar_system: str = Field(default="klein_3000", json_schema_extra={"example": "klein_3000"})
    vessel: str = Field(default="Unknown", json_schema_extra={"example": "RV_Surveyor"})
    notes: Optional[str] = None

class UploadResponse(BaseModel):
    task_id: str
    survey_id: str
    filename: str
    status: str = "queued"
    message: str

class ProcessDetectionsRequest(BaseModel):
    survey_id: str
    detections: List[Detection]
    image_info: SonarImageInfo
    mission_metadata: Optional[MissionMetadata] = None

class ProcessDetectionsResponse(BaseModel):
    task_id: str
    survey_id: str
    status: str = "processing"
    num_detections: int
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str  # queued, processing, completed, failed
    progress: Optional[float] = None  # 0.0 to 1.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

class ReportResponse(BaseModel):
    survey_id: str
    format: str
    data: Any  # JSON report data or file path
    generated_at: datetime

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    uptime_seconds: float
    active_tasks: int

class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int
