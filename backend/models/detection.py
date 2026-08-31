from sqlalchemy import Column, Float, Integer, String

from db import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    class_name = Column(String, nullable=False)

    confidence = Column(Float, nullable=False)

    latitude = Column(Float, nullable=True)

    longitude = Column(Float, nullable=True)

    width = Column(Float, nullable=True)

    height = Column(Float, nullable=True)

    status = Column(String, nullable=False, default="new")

    priority = Column(String, nullable=False, default="medium")