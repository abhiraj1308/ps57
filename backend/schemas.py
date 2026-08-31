from pydantic import BaseModel, Field


class DetectionCreate(BaseModel):
    class_name: str = Field(..., min_length=1)

    confidence: float = Field(
        ...,
        ge=0,
        le=1
    )

    latitude: float | None = None

    longitude: float | None = None

    width: float | None = None

    height: float | None = None

    status: str = "new"

    priority: str = "medium"


class DetectionResponse(DetectionCreate):
    id: int

    class Config:
        from_attributes = True