from sqlalchemy import Column, DateTime, Float, JSON, String
from sqlalchemy.sql import func

from backend.db import Base


class DetectionModel(Base):
    __tablename__ = "detections"

    detection_id = Column(String, primary_key=True, index=True)

    image_id = Column(String, nullable=False, index=True)

    class_name = Column(String, nullable=False)

    ai_confidence = Column(Float, nullable=False)

    final_confidence = Column(Float, nullable=True)

    bbox = Column(JSON, nullable=False)

    priority = Column(String, nullable=True)

    status = Column(String, nullable=False)

    latitude = Column(Float, nullable=True)

    longitude = Column(Float, nullable=True)

    estimated_width_m = Column(Float, nullable=True)

    estimated_height_m = Column(Float, nullable=True)

    timestamp = Column(DateTime, nullable=True)

    created_at = Column(
        DateTime,
        server_default=func.now(),
        nullable=False
    )