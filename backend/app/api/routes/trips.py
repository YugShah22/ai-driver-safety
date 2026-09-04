"""
Trip API routes.

Provides CRUD operations for trips and stubs for the AI analysis pipeline.
Authentication is validated via the Authorization: Bearer <jwt> header.
AI inference is NOT performed here — that belongs to the ML services (Phase 4+).
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, status

from app.db.supabase_client import get_supabase_admin, get_supabase_anon
from app.schemas.trip import (
    ProcessTripRequest,
    ProcessTripResponse,
    TripCreate,
    TripListResponse,
    TripResponse,
    TripUpdate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/trips", tags=["Trips"])


# ─── Auth helper ─────────────────────────────────────────────
async def get_current_user_id(authorization: str = Header(...)) -> str:
    """
    Extract and verify the user ID from the Supabase JWT token.
    The frontend passes: Authorization: Bearer <supabase_jwt>
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid Authorization header",
        )
    token = authorization.removeprefix("Bearer ").strip()

    try:
        supabase = get_supabase_admin()
        # Use Supabase admin to verify the JWT and get the user
        response = supabase.auth.get_user(token)
        if response.user is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return str(response.user.id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Token verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not verify credentials",
        ) from exc


# ─── Routes ──────────────────────────────────────────────────

@router.get("", response_model=TripListResponse, summary="List user trips")
async def list_trips(
    page: int          = Query(1, ge=1),
    size: int          = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    user_id: str       = Depends(get_current_user_id),
):
    """Return a paginated list of trips owned by the authenticated user."""
    try:
        supabase = get_supabase_admin()
        query = (
            supabase.table("trips")
            .select("*", count="exact")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .range((page - 1) * size, page * size - 1)
        )
        if status_filter:
            query = query.eq("status", status_filter)

        result = query.execute()
        return TripListResponse(
            items=result.data,
            total=result.count or 0,
            page=page,
            size=size,
        )
    except Exception as exc:
        logger.error("Error listing trips for user %s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch trips") from exc


@router.post("", response_model=TripResponse, status_code=status.HTTP_201_CREATED, summary="Create trip")
async def create_trip(
    body: TripCreate,
    user_id: str = Depends(get_current_user_id),
):
    """Create a new trip record. The video must have been uploaded to Supabase Storage first."""
    try:
        supabase = get_supabase_admin()
        result = (
            supabase.table("trips")
            .insert({
                "user_id":    user_id,
                "title":      body.title,
                "video_path": body.video_path,
                "status":     "UPLOADED",
            })
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create trip record")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error creating trip for user %s: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Failed to create trip") from exc


@router.get("/{trip_id}", response_model=TripResponse, summary="Get trip")
async def get_trip(
    trip_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Get a single trip. Returns 404 if the trip doesn't exist or belongs to another user."""
    try:
        supabase = get_supabase_admin()
        result = (
            supabase.table("trips")
            .select("*")
            .eq("id", str(trip_id))
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Trip not found")
        return result.data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error fetching trip %s: %s", trip_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch trip") from exc


@router.get("/{trip_id}/events", summary="Get trip driving events")
async def get_trip_events(
    trip_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Return all driving events detected for a trip."""
    try:
        supabase = get_supabase_admin()
        # Verify ownership
        trip = supabase.table("trips").select("id").eq("id", str(trip_id)).eq("user_id", user_id).single().execute()
        if not trip.data:
            raise HTTPException(status_code=404, detail="Trip not found")

        result = (
            supabase.table("driving_events")
            .select("*")
            .eq("trip_id", str(trip_id))
            .order("timestamp")
            .execute()
        )
        return {"trip_id": str(trip_id), "events": result.data or []}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch events") from exc


@router.get("/{trip_id}/metrics", summary="Get trip driving metrics")
async def get_trip_metrics(
    trip_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Return driving metrics time-series for a trip."""
    try:
        supabase = get_supabase_admin()
        trip = supabase.table("trips").select("id").eq("id", str(trip_id)).eq("user_id", user_id).single().execute()
        if not trip.data:
            raise HTTPException(status_code=404, detail="Trip not found")

        result = (
            supabase.table("driving_metrics")
            .select("*")
            .eq("trip_id", str(trip_id))
            .order("timestamp")
            .execute()
        )
        return {"trip_id": str(trip_id), "metrics": result.data or []}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch metrics") from exc


@router.get("/{trip_id}/risk", summary="Get trip risk predictions")
async def get_trip_risk(
    trip_id: UUID,
    user_id: str = Depends(get_current_user_id),
):
    """Return risk prediction time-series for a trip."""
    try:
        supabase = get_supabase_admin()
        trip = supabase.table("trips").select("id").eq("id", str(trip_id)).eq("user_id", user_id).single().execute()
        if not trip.data:
            raise HTTPException(status_code=404, detail="Trip not found")

        result = (
            supabase.table("risk_predictions")
            .select("*")
            .eq("trip_id", str(trip_id))
            .order("timestamp")
            .execute()
        )
        return {"trip_id": str(trip_id), "risk_predictions": result.data or []}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch risk predictions") from exc


@router.post(
    "/{trip_id}/process",
    response_model=ProcessTripResponse,
    summary="Trigger AI analysis pipeline",
)
async def process_trip(
    trip_id: UUID,
    body: ProcessTripRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user_id),
):
    """
    Trigger the AI analysis pipeline for a trip.

    STUB — Phase 4+ will implement the actual pipeline:
    Frame extraction → YOLO → Tracking → Segmentation → Feature extraction → ANN/ML → Risk engine
    """
    try:
        supabase = get_supabase_admin()
        trip_result = (
            supabase.table("trips")
            .select("id, status")
            .eq("id", str(trip_id))
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        if not trip_result.data:
            raise HTTPException(status_code=404, detail="Trip not found")

        trip = trip_result.data
        current_status = trip["status"]

        if current_status in ("PROCESSING", "ANALYZING") and not body.force:
            return ProcessTripResponse(
                trip_id=trip_id,
                status=current_status,
                message="Trip is already being processed. Pass force=true to restart.",
            )

        if current_status == "COMPLETED" and not body.force:
            return ProcessTripResponse(
                trip_id=trip_id,
                status="COMPLETED",
                message="Trip already analyzed. Pass force=true to re-analyze.",
            )

        # Update status to PROCESSING
        supabase.table("trips").update({"status": "PROCESSING"}).eq("id", str(trip_id)).execute()

        # TODO Phase 4: background_tasks.add_task(run_analysis_pipeline, trip_id, user_id)
        logger.info("Analysis pipeline stub triggered for trip %s (Phase 4 pending)", trip_id)

        return ProcessTripResponse(
            trip_id=trip_id,
            status="PROCESSING",
            message="Analysis pipeline will be implemented in Phase 4 (Video Processing).",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error triggering processing for trip %s: %s", trip_id, exc)
        raise HTTPException(status_code=500, detail="Failed to trigger processing") from exc
