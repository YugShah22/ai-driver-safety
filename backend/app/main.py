"""
AI Driver Safety & Intelligence Platform — FastAPI Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health
from app.api.routes import trips
from app.core.config import settings

app = FastAPI(
    title="AI Driver Safety & Intelligence Platform",
    description=(
        "Analyzes dashcam driving footage using computer vision, deep learning "
        "and machine learning to generate a comprehensive driving-risk assessment."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ─── CORS ────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(trips.router)   # Phase 2: Trip CRUD + pipeline stubs

# Future routers (Phase 4+):
# app.include_router(analyze.router, prefix="/api/v1", tags=["Analysis"])
