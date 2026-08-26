import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    """GET /health should return HTTP 200 with {status: ok}."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {"status": "ok"}


def test_health_content_type():
    """GET /health should return JSON content type."""
    response = client.get("/health")
    assert "application/json" in response.headers["content-type"]


def test_health_response_schema():
    """Response body must only contain the 'status' key."""
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert isinstance(data["status"], str)
