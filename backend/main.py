from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from db import Base, engine, get_db
from models.detection import Detection
from schemas import DetectionCreate, DetectionResponse


Base.metadata.create_all(bind=engine)


app = FastAPI(title="PS57 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post(
    "/detections",
    response_model=DetectionResponse
)
def create_detection(
    detection: DetectionCreate,
    db: Session = Depends(get_db)
):
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

    db.add(new_detection)
    db.commit()
    db.refresh(new_detection)

    return new_detection


@app.get(
    "/detections",
    response_model=list[DetectionResponse]
)
def get_detections(
    db: Session = Depends(get_db)
):
    detections = (
        db.query(Detection)
        .order_by(Detection.id.desc())
        .all()
    )

    return detections