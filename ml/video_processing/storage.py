"""
Supabase Storage integration for the video processing pipeline.

Uploads frames and thumbnails to:
  trips/{trip_id}/frames/frame_XXXXXX.jpg
  trips/{trip_id}/thumbnails/thumb_XXXXXX.jpg

Returns the Supabase Storage path for each uploaded file.
"""
from __future__ import annotations

import logging
from pathlib import Path

from supabase import Client

from .exceptions import StorageUploadError

logger = logging.getLogger(__name__)

# Bucket name — must match the bucket created in the SQL migration
VIDEOS_BUCKET = "videos"


def upload_frames(
    client: Client,
    trip_id: str,
    frames_dir: Path,
) -> list[str]:
    """
    Upload all frame images from ``frames_dir`` to Supabase Storage.

    Storage path pattern: ``trips/{trip_id}/frames/{filename}``

    Args:
        client:     Supabase admin client.
        trip_id:    UUID string of the trip being processed.
        frames_dir: Local directory containing frame image files.

    Returns:
        Ordered list of Supabase Storage paths.

    Raises:
        :class:`StorageUploadError` on any failure.
    """
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in exts)
    storage_paths: list[str] = []

    logger.info(
        "[Storage] Uploading %d frames for trip %s", len(files), trip_id
    )

    for local_path in files:
        storage_path = f"trips/{trip_id}/frames/{local_path.name}"
        _upload_single(client, local_path, storage_path)
        storage_paths.append(storage_path)

    logger.info(
        "[Storage] Uploaded %d frames for trip %s", len(storage_paths), trip_id
    )
    return storage_paths


def upload_thumbnails(
    client: Client,
    trip_id: str,
    thumbs_dir: Path,
) -> list[str]:
    """
    Upload all thumbnail images from ``thumbs_dir`` to Supabase Storage.

    Storage path pattern: ``trips/{trip_id}/thumbnails/{filename}``

    Returns:
        Ordered list of Supabase Storage paths.

    Raises:
        :class:`StorageUploadError` on any failure.
    """
    exts = {".jpg", ".jpeg", ".png"}
    files = sorted(p for p in thumbs_dir.iterdir() if p.suffix.lower() in exts)
    storage_paths: list[str] = []

    logger.info(
        "[Storage] Uploading %d thumbnails for trip %s", len(files), trip_id
    )

    for local_path in files:
        storage_path = f"trips/{trip_id}/thumbnails/{local_path.name}"
        _upload_single(client, local_path, storage_path)
        storage_paths.append(storage_path)

    logger.info(
        "[Storage] Uploaded %d thumbnails for trip %s", len(storage_paths), trip_id
    )
    return storage_paths


def _upload_single(client: Client, local_path: Path, storage_path: str) -> None:
    """Upload a single file; raise StorageUploadError on failure."""
    try:
        with open(local_path, "rb") as fh:
            content = fh.read()

        # Determine MIME type
        suffix = local_path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"

        client.storage.from_(VIDEOS_BUCKET).upload(
            path=storage_path,
            file=content,
            file_options={"content-type": mime, "upsert": "true"},
        )
    except StorageUploadError:
        raise
    except Exception as exc:
        raise StorageUploadError(
            f"Failed to upload {local_path.name} to Supabase Storage "
            f"path '{storage_path}': {exc}"
        ) from exc
