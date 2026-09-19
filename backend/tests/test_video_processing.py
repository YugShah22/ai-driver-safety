import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import cv2
import numpy as np
import pytest

from ml.video_processing.exceptions import (
    VideoNotFoundError,
    UnsupportedFormatError,
    CorruptVideoError,
)
from ml.video_processing.models import ProcessingConfig, ResizeMode
from ml.video_processing.processor import VideoProcessor
from ml.video_processing.pipeline import _insert_frame_records


@pytest.fixture
def synthetic_video():
    """Create a small synthetic video file for testing."""
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    
    # 10 frames at 10 fps (1 second duration)
    fps = 10.0
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    
    for i in range(10):
        # Create a dummy frame (gray background with white text)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 128
        cv2.putText(frame, f"Frame {i}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)
        
    out.release()
    yield path
    os.remove(path)


def test_metadata_extraction(synthetic_video):
    proc = VideoProcessor(synthetic_video)
    meta = proc.get_video_metadata()
    
    assert meta.filename == Path(synthetic_video).name
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 10.0
    assert meta.frame_count == 10
    assert meta.duration == 1.0


def test_frame_extraction(synthetic_video):
    # Extract at 5 fps (should get ~5 frames from a 1s 10fps video)
    cfg = ProcessingConfig(target_fps=5.0)
    proc = VideoProcessor(synthetic_video, cfg)
    
    frames = list(proc.extract_frames())
    assert len(frames) == 5
    
    # Check timestamps: 0.0, 0.2, 0.4, 0.6, 0.8
    timestamps = [f.timestamp for f in frames]
    assert timestamps == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_frame_resizing(synthetic_video):
    cfg = ProcessingConfig(resize_mode=ResizeMode.MAX_DIM, max_dimension=160)
    proc = VideoProcessor(synthetic_video, cfg)
    
    frames = list(proc.extract_frames())
    # Original is 320x240, max_dim=160. So width should become 160, height 120.
    assert frames[0].width == 160
    assert frames[0].height == 120
    assert frames[0].data.shape[:2] == (120, 160)


def test_save_frames_and_thumbnails(synthetic_video, tmp_path):
    cfg = ProcessingConfig(target_fps=2.0, thumbnail_width=100, thumbnail_height=100)
    proc = VideoProcessor(synthetic_video, cfg)
    frames = list(proc.extract_frames())
    
    trip_dir = tmp_path / "trip123"
    paths = proc.save_frames(frames, trip_dir)
    
    saved_frames = paths["saved_frames"]
    assert len(saved_frames) == 2
    
    frames_dir = paths["frames_dir"]
    thumbs_dir = paths["thumbs_dir"]
    
    assert frames_dir.exists()
    assert len(list(frames_dir.glob("*.jpg"))) == 2
    
    # Check thumbnails
    proc.create_thumbnails(frames_dir, thumbs_dir)
    assert thumbs_dir.exists()
    thumbs = list(thumbs_dir.glob("*.jpg"))
    assert len(thumbs) == 2
    
    # Verify thumbnail dimensions
    from PIL import Image
    with Image.open(thumbs[0]) as img:
        assert img.size[0] <= 100
        assert img.size[1] <= 100


def test_invalid_video():
    with pytest.raises(VideoNotFoundError):
        proc = VideoProcessor("nonexistent.mp4")
        proc.get_video_metadata()


def test_unsupported_format(tmp_path):
    bad_file = tmp_path / "video.txt"
    bad_file.write_text("not a video")
    
    with pytest.raises(UnsupportedFormatError):
        proc = VideoProcessor(bad_file)
        proc.get_video_metadata()


def test_database_insertion():
    supabase_mock = MagicMock()
    trip_id = str(uuid4())
    saved_frames = [
        {"frame_number": 1, "timestamp": 0.0, "local_path": Path("frame_000001.jpg")},
        {"frame_number": 2, "timestamp": 0.5, "local_path": Path("frame_000002.jpg")}
    ]
    storage_paths = [
        "trips/123/frames/frame_000001.jpg",
        "trips/123/frames/frame_000002.jpg"
    ]
    
    _insert_frame_records(supabase_mock, trip_id, saved_frames, storage_paths)
    
    supabase_mock.table.assert_called_with("frames")
    supabase_mock.table().insert.assert_called_once()
    
    args = supabase_mock.table().insert.call_args[0][0]
    assert len(args) == 2
    assert args[0]["timestamp"] == 0.0
    assert args[1]["timestamp"] == 0.5
