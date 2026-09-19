"""
End-to-end video processing pipeline.

Orchestrates:
  1. Download the video from Supabase Storage to a temp dir.
  2. Extract metadata via VideoProcessor.
  3. Extract + save frames.
  4. Generate thumbnails.
  5. Upload frames and thumbnails to Supabase Storage.
  6. Insert frame records into the PostgreSQL `frames` table.
  7. Update trip status (PROCESSING → PROCESSED / FAILED).

All heavy work runs in a background thread via FastAPI BackgroundTasks.
"""
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from supabase import Client

from .exceptions import (
    CorruptVideoError,
    DatabaseError,
    InvalidMetadataError,
    StorageUploadError,
    UnsupportedFormatError,
    VideoNotFoundError,
    VideoProcessingError,
)
from .models import ProcessingConfig, ResizeMode
from .processor import VideoProcessor
from .storage import upload_frames, upload_thumbnails

logger = logging.getLogger(__name__)

VIDEOS_BUCKET = "videos"

# Default extraction config — 1 fps, max 720p, JPEG output
DEFAULT_CONFIG = ProcessingConfig(
    target_fps=1.0,
    resize_mode=ResizeMode.MAX_DIM,
    max_dimension=720,
    max_frames=0,           # no limit
    output_format="jpg",
    jpeg_quality=85,
    thumbnail_width=320,
    thumbnail_height=180,
)


def run_processing_pipeline(
    trip_id: str,
    supabase: Client,
    config: Optional[ProcessingConfig] = None,
) -> None:
    """
    Full end-to-end processing pipeline for a single trip.

    Called from FastAPI BackgroundTasks — must NOT raise; instead it logs
    errors and sets the trip status to FAILED.

    Args:
        trip_id:  UUID of the trip to process.
        supabase: Supabase admin client (service-role, never exposed to frontend).
        config:   Optional processing configuration; defaults to DEFAULT_CONFIG.
    """
    cfg = config or DEFAULT_CONFIG
    work_dir: Optional[Path] = None

    logger.info("[Pipeline] Started processing for trip %s", trip_id)

    try:
        # ── 1. Fetch the trip record ─────────────────────────────────────────
        trip_result = (
            supabase.table("trips")
            .select("video_path, status")
            .eq("id", trip_id)
            .single()
            .execute()
        )
        if not trip_result.data:
            raise VideoNotFoundError(f"Trip {trip_id} not found in database")

        trip        = trip_result.data
        video_path  = trip.get("video_path")

        if not video_path:
            raise VideoNotFoundError(
                f"Trip {trip_id} has no video_path — cannot process"
            )

        # ── 2. Download the video from Supabase Storage to a temp file ───────
        work_dir = Path(tempfile.mkdtemp(prefix=f"driveai_{trip_id}_"))
        ext = Path(video_path).suffix or ".mp4"
        local_video = work_dir / f"video{ext}"

        logger.info(
            "[Pipeline] Downloading video from Storage: %s → %s",
            video_path, local_video,
        )
        video_bytes = supabase.storage.from_(VIDEOS_BUCKET).download(video_path)
        local_video.write_bytes(video_bytes)
        logger.info("[Pipeline] Video downloaded (%d bytes)", len(video_bytes))

        # ── 3. Extract metadata ───────────────────────────────────────────────
        processor = VideoProcessor(local_video, cfg)
        metadata  = processor.get_video_metadata()

        # Update trip with duration now that we know it
        if metadata.duration:
            supabase.table("trips").update(
                {"duration": round(metadata.duration, 3)}
            ).eq("id", trip_id).execute()

        # ── 4. Extract + save frames ──────────────────────────────────────────
        trip_work_dir = work_dir / "processing" / trip_id
        frames_gen = processor.extract_frames(metadata)
        paths = processor.save_frames(frames_gen, base_dir=trip_work_dir)
        frames_dir = paths["frames_dir"]
        thumbs_dir = paths["thumbs_dir"]

        saved_frames = paths["saved_frames"]

        # ── 5. Generate thumbnails ────────────────────────────────────────────
        processor.create_thumbnails(frames_dir, thumbs_dir)

        # ── 6. Upload frames to Supabase Storage ──────────────────────────────
        frame_storage_paths = upload_frames(supabase, trip_id, frames_dir)
        logger.info(
            "[Pipeline] Uploaded %d frames to Storage for trip %s",
            len(frame_storage_paths), trip_id,
        )

        # ── 7. Upload thumbnails to Supabase Storage ──────────────────────────
        thumb_storage_paths = upload_thumbnails(supabase, trip_id, thumbs_dir)
        logger.info(
            "[Pipeline] Uploaded %d thumbnails to Storage for trip %s",
            len(thumb_storage_paths), trip_id,
        )

        # ── 8. Insert frame records into the database ─────────────────────────
        _insert_frame_records(supabase, trip_id, saved_frames, frame_storage_paths)

        # ── 9. Mark trip as PROCESSED ─────────────────────────────────────────
        from datetime import datetime, timezone
        supabase.table("trips").update({
            "status": "PROCESSED",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", trip_id).execute()

        logger.info("[Pipeline] Successfully completed processing for trip %s", trip_id)

    except (
        VideoNotFoundError,
        CorruptVideoError,
        UnsupportedFormatError,
        InvalidMetadataError,
        StorageUploadError,
        DatabaseError,
        VideoProcessingError,
    ) as exc:
        logger.error("[Pipeline] Processing failed for trip %s: %s", trip_id, exc)
        _mark_failed(supabase, trip_id)

    except Exception as exc:
        logger.exception(
            "[Pipeline] Unexpected error during processing for trip %s: %s",
            trip_id, exc,
        )
        _mark_failed(supabase, trip_id)

    finally:
        # Always clean up temporary files
        if work_dir and work_dir.exists():
            VideoProcessor.cleanup_dir(work_dir)
            logger.info("[Pipeline] Cleaned up temp directory for trip %s", trip_id)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _insert_frame_records(
    supabase: Client,
    trip_id: str,
    saved_frames: list[dict[str, object]],
    storage_paths: list[str],
) -> None:
    """
    Insert a row into the ``frames`` table for each saved frame.
    """
    if len(saved_frames) != len(storage_paths):
        raise DatabaseError(
            f"Saved frame count ({len(saved_frames)}) does not match "
            f"storage path count ({len(storage_paths)}) for trip {trip_id}"
        )

    records = []
    # Sort saved_frames by local_path name to match storage_paths order
    saved_frames_sorted = sorted(saved_frames, key=lambda f: Path(str(f["local_path"])).name)

    for frame_meta, storage_path in zip(saved_frames_sorted, storage_paths):
        records.append({
            "trip_id":      trip_id,
            "frame_number": frame_meta["frame_number"],
            "timestamp":    frame_meta["timestamp"],
            "image_path":   storage_path,
        })

    try:
        supabase.table("frames").insert(records).execute()
        logger.info(
            "[DB] Inserted %d frame records for trip %s", len(records), trip_id
        )
    except Exception as exc:
        raise DatabaseError(
            f"Failed to insert frame records for trip {trip_id}: {exc}"
        ) from exc


def _mark_failed(supabase: Client, trip_id: str) -> None:
    """Set trip status to FAILED — best-effort, never raises."""
    try:
        supabase.table("trips").update({"status": "FAILED"}).eq("id", trip_id).execute()
    except Exception as exc:
        logger.error(
            "[Pipeline] Could not mark trip %s as FAILED: %s", trip_id, exc
        )
