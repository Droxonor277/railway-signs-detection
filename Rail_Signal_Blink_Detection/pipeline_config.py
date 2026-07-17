"""
shared dataclasses and configuration for the detection pipeline.
"""

from dataclasses import dataclass, field

import numpy as np


# --- pipeline dataclasses

# fps the config defaults are calibrated for; frame-count parameters
# (min_detection_frames, track_max_lost, etc.) are auto-scaled when reader_fps differs
_REFERENCE_FPS: int = 30

# 20 fps with 200 buffer ~ 3.3 GB RAM

@dataclass
class PipelineConfig:
    reader_fps: int | str = "full"      # target output fps from VideoReader; "full" = up to _REFERENCE_FPS (capped by source fps)
    detection_fps: int | str = 5        # frames per second sent to YOLO; or "full"
    reader_buffer_size: int = 20       # queue size between reader and detector threads
    detection_confidence: float = 0.5
    imgsz: int = 1920                   # yolo inference image size, set to 960 or 640 for faster inference on CPU
    device: str = "cuda:0"              # e.g. 'cuda:0' or 'cpu'
    corner_cover: bool = True           # paint over signal visualization in top-right corner of videos
    # --- segment detection parameters ---
    ring_buffer_size: int = 300         # max reader-fps frames held in detector ring buffer
    segment_offset_s: float = 0.2       # seconds (~ 6 frames) of context before/after a segment (for re-pass window and clips)
    min_detection_frames: int = 7        # min reader-fps frames with detection in the re-pass window for significance (auto-scaled with reader_fps)
    min_segment_confidence: float = 0.75 # additional confidence threshold for segment significance (0.0 = disabled)
    min_confident_frames: int = 2       # min frames with conf >= min_segment_confidence (auto-scaled with reader_fps; 0 = disabled)
    max_detection_gap_s: float = 0.5    # seconds (~ 6 frames of gaps allowed) of no phase-1 detection before closing a segment
    classify_classes: list[str] = field(default_factory=lambda: ["signal"]) # this classes will be passed to classifier 
    #   add "dwarf_signal" and "sign_stripes", "sign_triangle"?
    box_relative: bool = True           # if True, bbox coords in csv are relative (0-1) instead of pixels
    # --- tracking (phase 2) ---
    tracking_enabled: bool = True       # enable centre-distance tracking in phase 2
    track_max_dist_limits: tuple[float, float] = (0.05, 0.15) # (min, max) normalised centre distance; auto-scaled with reader_fps
    # NOTE: defaults are calibrated for _REFERENCE_FPS (30 fps); auto-scaled proportionally when reader_fps differs.
    track_max_lost: int = 5             # frames a track can be absent before being discarded (auto-scaled with reader_fps)
    # --- track postprocessing (phase 2) ---
    segment_postprocess: bool = True   # enable per-segment track postprocessing (side filter, area filter, center-proximity)
    min_track_max_area: float = 0.0001     # drop tracks whose bbox never reached this relative area (0-1); 0.0 = disabled
    detection_side: str = "both"        # keep only tracks on this side of center_line_x: "left", "right", or "both"
    center_line_x: float = 0.0         # reference vertical line in [-1, 1]; 0 = image center, -1 = left edge, 1 = right edge
    postprocess_last_frames: int = 5    # last N detected frames per track used for side filter and center-proximity selection
    # --- debug ---
    debug: bool = False                  # print extra phase-1/phase-2 diagnostic information
    debug_phase1: bool = False           # print every phase-1 detection (class, frame, time) at detection_fps


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in relative (0-1) or absolute pixel coords
    class_id: int
    class_name: str
    confidence: float
    track_id: int | None = None
    timestamp_ms: float | None = None       # timestamp of the frame this detection belongs to


@dataclass
class FramePacket:
    frame_no: int         # original frame index in the source video
    timestamp_ms: float   # timestamp in milliseconds from the source video
    image: np.ndarray
    detections: list[Detection] = field(default_factory=list)


# --- corner cover constants - same as used during dataset preparation

# regions in YOLO format (cx, cy, w, h) - relative to frame dimensions
# source: upper-left reference area used to sample the fill colour
COVER_SOURCE_BOX: tuple[float, float, float, float] = (0.014, 0.059, 0.028, 0.117)
# target: upper-right area (in-cab traffic light display) to paint over
COVER_TARGET_BOX: tuple[float, float, float, float] = (0.982, 0.065, 0.035, 0.130)