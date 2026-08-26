"""
Health-check route — GET /health
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns `{status: ok}` when the API is running.",
)
async def health_check() -> HealthResponse:
    """Lightweight liveness probe."""
    return HealthResponse(status="ok")
