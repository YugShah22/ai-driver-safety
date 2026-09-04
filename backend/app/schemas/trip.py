"""
Pydantic schemas for trip-related API request/response models.
These match the database tables defined in supabase/migrations/001_initial_schema.sql.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ─── Enums ───────────────────────────────────────────────────
TripStatus = Literal["UPLOADED", "PROCESSING", "ANALYZING", "COMPLETED", "FAILED"]
RiskClass  = Literal["SAFE", "MODERATE", "HIGH"]
Severity   = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


# ─── Trip ─────────────────────────────────────────────────────
class TripCreate(BaseModel):
    """Request body for creating a new trip."""
    title:      str = Field(..., min_length=1, max_length=255, description="Human-readable trip name")
    video_path: Optional[str] = Field(None, description="Supabase Storage path to the uploaded video")


class TripUpdate(BaseModel):
    """Request body for updating a trip (partial)."""
    title:        Optional[str]       = Field(None, min_length=1, max_length=255)
    status:       Optional[TripStatus] = None
    completed_at: Optional[datetime]  = None
    duration:     Optional[float]     = Field(None, ge=0, description="Video duration in seconds")


class TripResponse(BaseModel):
    """Full trip record returned to the caller."""
    id:           UUID
    user_id:      UUID
    title:        str
    video_path:   Optional[str]
    duration:     Optional[float]
    status:       TripStatus
    created_at:   datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class TripListResponse(BaseModel):
    """Paginated list of trips."""
    items: list[TripResponse]
    total: int
    page:  int
    size:  int


# ─── Frame ────────────────────────────────────────────────────
class FrameResponse(BaseModel):
    id:           UUID
    trip_id:      UUID
    frame_number: int
    timestamp:    float
    image_path:   Optional[str]

    class Config:
        from_attributes = True


# ─── Detection ────────────────────────────────────────────────
class DetectionResponse(BaseModel):
    id:          UUID
    trip_id:     UUID
    frame_id:    Optional[UUID]
    object_type: str
    confidence:  float
    x1:          float
    y1:          float
    x2:          float
    y2:          float
    created_at:  datetime

    class Config:
        from_attributes = True


# ─── Driving Metrics ──────────────────────────────────────────
class DrivingMetricResponse(BaseModel):
    id:                  UUID
    trip_id:             UUID
    timestamp:           float
    speed:               Optional[float]  # km/h ESTIMATED
    acceleration:        Optional[float]  # m/s² ESTIMATED
    lane_deviation:      Optional[float]  # px ESTIMATED
    vehicle_density:     Optional[int]    # count INFERRED
    pedestrian_distance: Optional[float]  # px ESTIMATED
    time_to_collision:   Optional[float]  # s ESTIMATED
    created_at:          datetime

    class Config:
        from_attributes = True


# ─── Driving Event ────────────────────────────────────────────
class DrivingEventResponse(BaseModel):
    id:          UUID
    trip_id:     UUID
    event_type:  str
    severity:    Severity
    confidence:  Optional[float]
    timestamp:   float
    frame_id:    Optional[UUID]
    description: Optional[str]
    created_at:  datetime

    class Config:
        from_attributes = True


# ─── Risk Prediction ──────────────────────────────────────────
class RiskPredictionResponse(BaseModel):
    id:            UUID
    trip_id:       UUID
    timestamp:     float
    risk_score:    float
    risk_class:    RiskClass
    model_name:    str
    model_version: str
    created_at:    datetime

    class Config:
        from_attributes = True


# ─── Process trigger ──────────────────────────────────────────
class ProcessTripRequest(BaseModel):
    """Request to trigger the AI analysis pipeline for a trip."""
    force: bool = Field(
        False,
        description="Re-process even if the trip was already analyzed"
    )


class ProcessTripResponse(BaseModel):
    trip_id:    UUID
    status:     TripStatus
    message:    str
    job_id:     Optional[str] = None
