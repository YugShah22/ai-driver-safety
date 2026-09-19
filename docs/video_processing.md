# Video Processing Architecture

The video processing pipeline handles dashcam footage immediately after a user uploads it. It runs entirely asynchronously in the background.

## Overview
1. **Triggered via API**: Client calls `POST /api/trips/{trip_id}/process`
2. **Status Update**: Trip is marked as `PROCESSING`
3. **Download**: Video is downloaded from Supabase Storage (`videos` bucket) to a local temporary directory.
4. **Metadata Extraction**: OpenCV extracts duration, FPS, codecs, and dimensions.
5. **Frame Extraction**: OpenCV samples frames (incrementally, without loading the full video into memory).
6. **Resizing & Thumbnails**: Frames are optionally resized (preserving aspect ratio) and smaller JPEG thumbnails are generated using Pillow.
7. **Storage Upload**: Frames and thumbnails are uploaded back to Supabase Storage.
8. **Database Update**: Frame metadata (frame number, precise timestamp, storage paths) are inserted into the `frames` table.
9. **Status Completion**: Trip is marked as `PROCESSED` (or `FAILED` if errors occurred).

## Configuration
The pipeline is highly configurable via `ProcessingConfig`:
- **Sampling Behavior**: Extracts frames based on a specified interval (e.g., `target_fps = 1.0` for 1 frame per second).
- **Resizing**: Supports keeping the original resolution, resizing to exact target dimensions, or clamping by maximum dimension (e.g., `max_dimension=720`) to save processing time later.
- **Output Quality**: Output format defaults to JPEG with a configurable quality (default `85`).

## Storage Paths
All assets related to a single trip are stored hierarchically:
* **Video**: `trips/{trip_id}/{video_filename}`
* **Extracted Frames**: `trips/{trip_id}/frames/frame_000001.jpg`
* **Thumbnails**: `trips/{trip_id}/thumbnails/thumb_000001.jpg`

## API Endpoint
* `POST /api/trips/{trip_id}/process`
  * Requires authentication (Bearer Token).
  * Enforces RLS via user_id matching before triggering the job.
  * Returns immediately with `HTTP 200` while the background task runs.

## Trip Status Lifecycle
1. `UPLOADED` - Trip record created, video uploaded but unprocessed.
2. `PROCESSING` - Background pipeline has started.
3. `PROCESSED` - Frames extracted, DB populated, ready for AI Inference (Phase 4).
4. `FAILED` - Any step of the processing pipeline raised an exception. Can be restarted with `?force=true`.

## System Requirements
* Python 3.10+
* **OpenCV** (`opencv-python-headless` or `opencv-python`)
* Underlying OS must support OpenCV video codecs (usually provided natively, but FFmpeg backend bindings are used implicitly by cv2.VideoCapture on most systems).
