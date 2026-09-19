"""
Core VideoProcessor implementation.

Responsibilities:
  - get_video_metadata()  — extract file / codec / frame information via OpenCV
  - extract_frames()      — incrementally sample frames from the video
  - save_frames()         — write sampled frames to a temporary working directory
  - create_thumbnails()   — generate smaller JPEG copies of saved frames

No AI inference is performed here.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Generator, Iterator, Optional

import cv2
import numpy as np
from PIL import Image

from .exceptions import (
    CorruptVideoError,
    InvalidMetadataError,
    UnsupportedFormatError,
    VideoNotFoundError,
)
from .models import ExtractedFrame, ProcessingConfig, ResizeMode, VideoMetadata

logger = logging.getLogger(__name__)

# ─── Supported video extensions ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v", ".3gp", ".wmv",
}


class VideoProcessor:
    """
    Generic dashcam video preprocessor.

    Usage::

        config = ProcessingConfig(target_fps=2.0, resize_mode=ResizeMode.MAX_DIM,
                                   max_dimension=640)
        proc = VideoProcessor("/tmp/video.mp4", config)
        meta = proc.get_video_metadata()
        frames = list(proc.extract_frames())
        paths  = proc.save_frames(frames, base_dir=Path("/tmp/processing/trip-123"))
        thumbs = proc.create_thumbnails(paths["frames_dir"], paths["thumbs_dir"])
    """

    def __init__(
        self,
        video_path: str | Path,
        config: Optional[ProcessingConfig] = None,
    ) -> None:
        self._video_path = Path(video_path)
        self.config = config or ProcessingConfig()
        self._metadata: Optional[VideoMetadata] = None

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def get_video_metadata(self) -> VideoMetadata:
        """
        Extract metadata from the video file.

        Returns a :class:`VideoMetadata` instance.
        Raises :class:`VideoNotFoundError`, :class:`CorruptVideoError`,
        :class:`UnsupportedFormatError`, or :class:`InvalidMetadataError`.
        """
        path = self._video_path

        if not path.exists():
            raise VideoNotFoundError(f"Video file not found: {path}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise UnsupportedFormatError(
                f"Unsupported video extension '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        file_size = path.stat().st_size
        if file_size == 0:
            raise CorruptVideoError(f"Video file is empty (0 bytes): {path}")

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            cap.release()
            raise CorruptVideoError(
                f"OpenCV could not open the video file. "
                f"The file may be corrupt or in an unsupported codec: {path}"
            )

        try:
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            raw_fps     = cap.get(cv2.CAP_PROP_FPS)
            raw_count   = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fourcc_int  = int(cap.get(cv2.CAP_PROP_FOURCC))

            if width <= 0 or height <= 0:
                raise InvalidMetadataError(
                    f"Video reports invalid dimensions ({width}×{height}): {path}"
                )

            # Validate FPS — some containers report 0 or NaN
            fps: Optional[float] = None
            if raw_fps and raw_fps > 0 and not np.isnan(raw_fps):
                fps = float(raw_fps)
            else:
                logger.warning(
                    "[VideoProcessor] Video reports invalid FPS (%s) — "
                    "timestamps will be estimated from frame positions: %s",
                    raw_fps, path.name,
                )

            # Frame count — some containers report -1 or 0
            frame_count: Optional[int] = None
            if raw_count and raw_count > 0:
                frame_count = int(raw_count)

            # Duration — derive from fps + count if available
            duration: Optional[float] = None
            if fps and frame_count:
                duration = frame_count / fps

            # Codec / container
            codec: Optional[str] = None
            if fourcc_int:
                codec = "".join([
                    chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
                ]).strip()

            container = ext.lstrip(".")

        finally:
            cap.release()

        meta = VideoMetadata(
            filename=path.name,
            file_size=file_size,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration=duration,
            codec=codec or None,
            container=container,
        )
        self._metadata = meta
        logger.info(
            "[VideoProcessor] Metadata extracted | file=%s size=%d fps=%s "
            "dims=%dx%d frames=%s duration=%ss codec=%s",
            meta.filename, meta.file_size, meta.fps,
            meta.width, meta.height, meta.frame_count,
            f"{meta.duration:.2f}" if meta.duration else "unknown",
            meta.codec,
        )
        return meta

    # ─────────────────────────────────────────────────────────────────────────

    def extract_frames(
        self,
        metadata: Optional[VideoMetadata] = None,
    ) -> Generator[ExtractedFrame, None, None]:
        """
        Incrementally yield :class:`ExtractedFrame` objects from the video.

        Frames are NOT loaded all at once — the generator reads forward through
        the file using ``cap.grab()`` / ``cap.retrieve()``, keeping only one
        decoded frame in memory at a time.

        Args:
            metadata: Pre-computed metadata (to avoid re-reading the file).
                      If None, :meth:`get_video_metadata` is called first.

        Yields:
            :class:`ExtractedFrame` instances in ascending timestamp order.

        Raises:
            :class:`CorruptVideoError` if the file cannot be read mid-stream.
        """
        if metadata is None:
            metadata = self.get_video_metadata()

        cfg = self.config
        interval = cfg.effective_sample_interval(metadata.fps)

        cap = cv2.VideoCapture(str(self._video_path))
        if not cap.isOpened():
            raise CorruptVideoError(
                f"OpenCV lost access to video file during extraction: {self._video_path}"
            )

        extracted_count = 0
        source_frame_idx = 0        # which source frame we are positioned at
        next_target_ts   = 0.0      # next timestamp we want to capture

        try:
            while True:
                if cfg.max_frames and extracted_count >= cfg.max_frames:
                    break

                # Compute the target source frame number for next_target_ts
                if metadata.fps:
                    target_source_frame = int(next_target_ts * metadata.fps)
                else:
                    # No reliable FPS — seek by position index using interval
                    target_source_frame = int(next_target_ts * 25)  # assume 25 fps for seek

                # Skip (grab without decode) frames we don't need
                while source_frame_idx < target_source_frame:
                    ret = cap.grab()
                    if not ret:
                        return  # end of video
                    source_frame_idx += 1

                # Decode the frame we actually want
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break

                # Compute accurate timestamp
                raw_ts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if raw_ts > 0:
                    timestamp = raw_ts / 1000.0
                elif metadata.fps:
                    timestamp = source_frame_idx / metadata.fps
                else:
                    timestamp = next_target_ts

                source_frame_idx += 1
                extracted_count  += 1

                # Resize if needed
                bgr, out_w, out_h = self._resize_frame(bgr, cfg)

                yield ExtractedFrame(
                    frame_number=extracted_count,
                    timestamp=round(timestamp, 6),
                    width=out_w,
                    height=out_h,
                    data=bgr,
                )

                next_target_ts += interval

        finally:
            cap.release()

        logger.info(
            "[VideoProcessor] Extracted %d frames from %s (interval=%.3fs)",
            extracted_count, self._video_path.name, interval,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def save_frames(
        self,
        frames: Iterator[ExtractedFrame],
        base_dir: Path,
    ) -> dict[str, object]:
        """
        Save extracted frames to ``base_dir/frames/`` using deterministic names.

        Naming: ``frame_000001.jpg`` (or ``.png`` based on config).

        Args:
            frames:    Iterable of :class:`ExtractedFrame` (from :meth:`extract_frames`).
            base_dir:  Working directory root for this trip (``processing/{trip_id}/``).

        Returns:
            dict containing ``"frames_dir"``, ``"thumbs_dir"``, and ``"saved_frames"`` (list of dicts).
        """
        cfg = self.config
        frames_dir = base_dir / "frames"
        thumbs_dir = base_dir / "thumbnails"
        frames_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        ext = "jpg" if cfg.output_format.lower() in ("jpg", "jpeg") else "png"
        saved_frames_meta: list[dict[str, object]] = []

        encode_params: list[int] = []
        if ext == "jpg":
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]

        for frame in frames:
            name = f"frame_{frame.frame_number:06d}.{ext}"
            dest = frames_dir / name
            ok = cv2.imwrite(str(dest), frame.data, encode_params)
            if not ok:
                raise OSError(f"cv2.imwrite failed for {dest}")
            saved_frames_meta.append({
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp,
                "local_path": dest
            })

        logger.info(
            "[VideoProcessor] Saved %d frames to %s",
            len(saved_frames_meta), frames_dir,
        )
        return {
            "frames_dir": frames_dir,
            "thumbs_dir": thumbs_dir,
            "saved_frames": saved_frames_meta
        }

    # ─────────────────────────────────────────────────────────────────────────

    def create_thumbnails(
        self,
        frames_dir: Path,
        thumbs_dir: Path,
    ) -> list[Path]:
        """
        Generate JPEG thumbnails from already-saved frames.

        Reads each frame file from ``frames_dir``, resizes to thumbnail dimensions
        while preserving aspect ratio, and writes to ``thumbs_dir``.

        Args:
            frames_dir: Directory containing saved frame images.
            thumbs_dir: Directory where thumbnails will be written.

        Returns:
            List of thumbnail :class:`Path` objects in frame order.
        """
        cfg = self.config
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        # Collect frame files in deterministic order
        exts = {".jpg", ".jpeg", ".png"}
        frame_files = sorted(
            p for p in frames_dir.iterdir() if p.suffix.lower() in exts
        )

        thumb_paths: list[Path] = []

        for frame_file in frame_files:
            # Build matching thumbnail name
            thumb_name = "thumb_" + frame_file.name.removeprefix("frame_")
            # Always write as JPEG regardless of source format
            thumb_name = Path(thumb_name).with_suffix(".jpg").name
            thumb_dest = thumbs_dir / thumb_name

            # Use PIL for efficient thumbnail creation (no full decode to numpy needed)
            with Image.open(frame_file) as img:
                # thumbnail() resizes IN PLACE preserving aspect ratio,
                # shrinking only — never upscaling.
                img_copy = img.copy()

            img_copy.thumbnail(
                (cfg.thumbnail_width, cfg.thumbnail_height),
                Image.LANCZOS,
            )
            img_copy.save(str(thumb_dest), format="JPEG", quality=80)
            thumb_paths.append(thumb_dest)

        logger.info(
            "[VideoProcessor] Generated %d thumbnails in %s",
            len(thumb_paths), thumbs_dir,
        )
        return thumb_paths

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resize_frame(
        bgr: "np.ndarray",
        cfg: ProcessingConfig,
    ) -> tuple["np.ndarray", int, int]:
        """
        Return (resized_bgr, width, height) according to the config.

        - ResizeMode.ORIGINAL  → returns the frame unchanged.
        - ResizeMode.TARGET    → resizes to (target_width, target_height) exactly.
        - ResizeMode.MAX_DIM   → scales down so max(w, h) == max_dimension,
                                 preserving aspect ratio.  Never upscales.
        """
        h, w = bgr.shape[:2]

        if cfg.resize_mode == ResizeMode.ORIGINAL:
            return bgr, w, h

        if cfg.resize_mode == ResizeMode.TARGET:
            tw = cfg.target_width or w
            th = cfg.target_height or h
            if tw == w and th == h:
                return bgr, w, h
            resized = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
            return resized, tw, th

        if cfg.resize_mode == ResizeMode.MAX_DIM:
            max_d = cfg.max_dimension or max(w, h)
            if max(w, h) <= max_d:
                return bgr, w, h
            scale = max_d / max(w, h)
            nw = max(1, int(w * scale))
            nh = max(1, int(h * scale))
            resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            return resized, nw, nh

        return bgr, w, h  # fallback — should never reach here

    @staticmethod
    def cleanup_dir(directory: Path) -> None:
        """Remove a working directory and all its contents."""
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
            logger.debug("[VideoProcessor] Cleaned up temp directory: %s", directory)
