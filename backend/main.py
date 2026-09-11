from pathlib import Path
import sys
import uuid

from fastapi import (
    Depends,
    FastAPI,
    File,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session


# ============================================================
# BACKEND IMPORTS
# ============================================================

from db import Base, engine, get_db
from models.detection import Detection
from schemas import DetectionCreate, DetectionResponse


# ============================================================
# PROJECT PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UPLOAD_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "uploads"
    / "sss"
)

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

Base.metadata.create_all(
    bind=engine
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="PS57 API",
    description=(
        "Side-Scan Sonar Anomaly Detection API"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    """
    Basic backend health check.
    """

    return {
        "status": "ok",
        "service": "ps57-api",
    }


# ============================================================
# CREATE DETECTION
# ============================================================

@app.post(
    "/detections",
    response_model=DetectionResponse,
)
def create_detection(
    detection: DetectionCreate,
    db: Session = Depends(get_db),
):
    """
    Manually creates a detection record.

    This endpoint is retained from the existing PS57
    backend and is useful for testing and internal use.
    """

    new_detection = Detection(

        class_name=detection.class_name,

        confidence=detection.confidence,

        latitude=detection.latitude,

        longitude=detection.longitude,

        width=detection.width,

        height=detection.height,

        status=detection.status,

        priority=detection.priority,
    )

    db.add(
        new_detection
    )

    db.commit()

    db.refresh(
        new_detection
    )

    return new_detection


# ============================================================
# GET DETECTIONS
# ============================================================

@app.get(
    "/detections",
    response_model=list[DetectionResponse],
)
def get_detections(
    db: Session = Depends(get_db),
):
    """
    Returns the latest PS57 detection records.
    """

    detections = (
        db.query(Detection)
        .order_by(
            Detection.id.desc()
        )
        .all()
    )

    return detections


# ============================================================
# ANALYSIS UPLOAD
# ============================================================

@app.post("/analyze")
async def analyze_sss(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    import json
    import random
    from detection_intelligence.die_engine import filter_and_refine_detections

    if not file.filename:
        return {"status": "error", "message": "No filename supplied."}

    allowed_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".xtf"}
    extension = Path(file.filename).suffix.lower()

    if extension not in allowed_extensions:
        return {"status": "error", "message": "Unsupported file type."}

    safe_filename = f"{uuid.uuid4().hex}{extension}"
    destination = UPLOAD_DIR / safe_filename
    contents = await file.read()
    destination.write_bytes(contents)

    # 1. Mock Pre-processing & AI/ML
    sample_json_path = PROJECT_ROOT / "datasets" / "sample_ai_output.json"
    ai_json_output = {}
    if sample_json_path.exists():
        with open(sample_json_path, "r") as f:
            ai_json_output = json.load(f)

    # 2. DIE Engine
    final_detections = filter_and_refine_detections(ai_json_output)

    # 3. Geotagging Mock & Persist
    for detection in final_detections:
        # Mock coordinates around a base location
        detection.latitude = 19.0760 + (random.random() - 0.5) * 0.01
        detection.longitude = 72.8777 + (random.random() - 0.5) * 0.01
        
        # Save to DB
        record = Detection(
            class_name=detection.class_name,
            confidence=float(detection.confidence),
            latitude=detection.latitude,
            longitude=detection.longitude,
            width=float(detection.bbox.width),
            height=float(detection.bbox.height),
            status=detection.status or "new",
            priority=detection.severity or "medium",
        )
        db.add(record)

    db.commit()

    return {
        "status": "accepted",
        "message": "SSS image processed and detections filtered.",
        "filename": safe_filename,
        "original_filename": file.filename,
        "size_bytes": len(contents),
        "detections_count": len(final_detections)
    }

