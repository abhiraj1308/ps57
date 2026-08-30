from fastapi import Depends, FastAPI
from sqlalchemy.orm import Session

from backend.db import Base, engine, get_db
from backend.models.detection import DetectionModel
from backend.schemas import Detection


app = FastAPI(title="PS57 API")


Base.metadata.create_all(bind=engine)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detections")
def create_detection(
    detection: Detection,
    db: Session = Depends(get_db)
):
    db_detection = DetectionModel(
        detection_id=detection.detection_id,
        image_id=detection.image_id,
        class_name=detection.class_name.value,
        ai_confidence=detection.ai_confidence,
        final_confidence=detection.final_confidence,
        bbox=detection.bbox.model_dump(),
        priority=detection.priority.value if detection.priority else None,
        status=detection.status.value,
        latitude=detection.latitude,
        longitude=detection.longitude,
        estimated_width_m=detection.estimated_width_m,
        estimated_height_m=detection.estimated_height_m,
        timestamp=detection.timestamp,
    )

    db.add(db_detection)
    db.commit()
    db.refresh(db_detection)

    return {
        "status": "success",
        "detection": detection
    }