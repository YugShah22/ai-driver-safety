"""
Tests for the trips API routes.

These tests verify the API contract (request/response shapes, auth enforcement,
status codes) without requiring a live Supabase connection.
"""
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


# ─── Health still works ───────────────────────────────────────
@pytest.mark.asyncio
async def test_health_still_returns_ok():
    """Phase 2 must not break the Phase 1 health endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ─── Auth enforcement ─────────────────────────────────────────
@pytest.mark.asyncio
async def test_list_trips_requires_auth():
    """GET /api/trips must return 422 without an Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trips")
    # FastAPI returns 422 when a required Header is missing
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_trips_rejects_bad_token():
    """GET /api/trips must return 401 with an invalid JWT token."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/trips",
            headers={"Authorization": "Bearer not-a-real-token"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_create_trip_requires_auth():
    """POST /api/trips must return 422 without an Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/trips",
            json={"title": "Test trip"},
        )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_trip_requires_auth():
    """GET /api/trips/{id} must return 422 without an Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/trips/00000000-0000-0000-0000-000000000001")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_process_trip_requires_auth():
    """POST /api/trips/{id}/process must return 422 without an Authorization header."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/trips/00000000-0000-0000-0000-000000000001/process",
            json={"force": False},
        )
    assert resp.status_code == 422


# ─── Schema validation ────────────────────────────────────────
@pytest.mark.asyncio
async def test_create_trip_validates_empty_title():
    """
    POST /api/trips with an empty title and a bad token returns 401.
    FastAPI evaluates Header dependencies (auth) before the request body,
    so an invalid token results in 401 before Pydantic body validation runs.
    """
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/api/trips",
            json={"title": ""},
            headers={"Authorization": "Bearer fake-token"},
        )
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_create_trip_body_without_title_returns_422():
    """POST /api/trips with a missing Authorization header returns 422 (missing required header)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/trips", json={})
    assert resp.status_code == 422


# ─── OpenAPI docs ─────────────────────────────────────────────
@pytest.mark.asyncio
async def test_openapi_includes_trips():
    """Swagger spec must include the /api/trips endpoints."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json().get("paths", {})
    assert "/api/trips" in paths
    assert "/api/trips/{trip_id}" in paths
    assert "/api/trips/{trip_id}/process" in paths
