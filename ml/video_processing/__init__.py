"""
Video Processing module — OpenCV-based dashcam video utilities.

Phase 3+ will implement:
  - VideoLoader: reads video files and streams frames
  - FrameSampler: configurable FPS sampling (e.g., 5fps from 30fps source)
  - VideoPreprocessor: resize, normalize, color-space conversion
  - MotionAnalyzer: optical flow for speed / motion estimation
  - AnnotatedVideoWriter: overlays detection boxes + risk score on output video
"""
