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

    allow_origins=[
        "http://localhost:5173",
    ],

    allow_credentials=True,

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
):
    """
    Uploads a Side-Scan Sonar image for PS57 analysis.

    IMPORTANT:
    This endpoint currently handles the input side of the
    pipeline only.

    The actual sequence:

        preprocessing
        → AI
        → intelligence
        → geolocation
        → database

    will be connected here after the teammate modules are
    integrated.

    Supported image formats:

        PNG
        JPG
        JPEG
        TIF
        TIFF
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        return {
            "status": "error",
            "message": "No filename supplied.",
        }


    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    allowed_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    extension = (
        Path(file.filename)
        .suffix
        .lower()
    )

    if extension not in allowed_extensions:

        return {
            "status": "error",
            "message": (
                "Unsupported file type. "
                "Use PNG, JPG, JPEG, TIF or TIFF."
            ),
        }


    # --------------------------------------------------------
    # Generate safe filename
    # --------------------------------------------------------

    safe_filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    destination = (
        UPLOAD_DIR
        / safe_filename
    )


    # --------------------------------------------------------
    # Save uploaded SSS image
    # --------------------------------------------------------

    contents = await file.read()

    destination.write_bytes(
        contents
    )


    # --------------------------------------------------------
    # TEMPORARY PIPELINE RESPONSE
    # --------------------------------------------------------
    #
    # We deliberately do NOT fake AI results here.
    #
    # The endpoint currently proves that the backend can:
    #
    #     browser
    #         ↓
    #     POST /analyze
    #         ↓
    #     receive SSS image
    #         ↓
    #     save image
    #
    # Later the actual PS57 master pipeline will be called.
    # --------------------------------------------------------

    return {
        "status": "accepted",

        "message": (
            "SSS image uploaded successfully. "
            "Analysis pipeline pending integration."
        ),

        "filename": safe_filename,

        "original_filename": (
            file.filename
        ),

        "size_bytes": len(contents),
    }
