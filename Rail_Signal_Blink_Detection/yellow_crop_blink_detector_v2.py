"""
Standalone diagnostic analysis of a blinking yellow light from YOLO bbox crops.

Usage:
    1) Edit values in the CONFIG section.
    2) Run:
           python yellow_crop_blink_detector_v2.py

The script does not require test_yellow_pipeline_bbox.py.
The input is a video plus a CSV file from the detection pipeline. The script reads
one detection_ID from the CSV, extracts the corresponding bbox crops from the video,
and determines whether the yellow light is blinking.

It also saves diagnostics:
    - raw crop + normalized crop + yellow/white/active masks
    - CSV with all per-frame signals
    - brightness/active/detrended/FFT plot
    - text report explaining why the algorithm made the decision
"""

from __future__ import annotations

import csv
import importlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# =============================================================================
# CONFIG - edit here; no command-line arguments are required
# =============================================================================

VIDEO_PATH = r"C:/code/tym1/dataset/blinking_yellow_signal/signal_4.mp4"
MODEL_PATH = r"C:/code/tym1/LeniaDynamics_rel022/models/yolo11s_n6.pt"
OUTPUT_DIR = r"C:/code/tym1/LeniaDynamics_rel022/results"
"""
VIDEO_PATH = r"C:/code/tym1/dataset/one_yellow_signal/signal_7.mp4"
MODEL_PATH = r"C:/code/tym1/LeniaDynamics_rel022/models/yolo11s_n6.pt"
OUTPUT_DIR = r"C:/code/tym1/LeniaDynamics_rel022/results"
"""


# Leave CSV_PATH as None = the script creates OUTPUT_DIR/<video_name>.csv.
# To use a fixed path, set for example r"C:/.../results/signal_4.csv".
CSV_PATH: Optional[str] = None

# If True, the script first runs the detection pipeline and creates a CSV,
# then immediately runs yellow blinking analysis from this CSV.
RUN_PIPELINE_BEFORE_ANALYSIS = True

# False = if the CSV already exists, the pipeline is not run again.
# True = recreate the CSV on every run.
FORCE_RECREATE_CSV = True

# Detection pipeline settings.
DEVICE = "cuda:0"      # if GPU is not available, set "cpu"
READER_FPS: int | str = "full"
DETECTION_FPS: int | str = 15
DETECTION_CONFIDENCE = 0.5
IMGSZ = 1920
DETECTION_SIDE = "both"
BOX_RELATIVE = True

# If set to None, the longest track in the CSV is selected.
SELECTED_ID: Optional[str] = "2"
LABEL_FILTER = "signal"
"""
OUTPUT_PLOT = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_7_yellow_analysis.png"
EVENTS_CSV = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_7_events.csv"
SIGNALS_CSV = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_7_diagnostic_signals.csv"
REPORT_TXT = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_7_diagnostic_report.txt"



"""
OUTPUT_PLOT = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_4_yellow_analysis.png"
EVENTS_CSV = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_4_events.csv"
SIGNALS_CSV = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_4_diagnostic_signals.csv"
REPORT_TXT = r"C:/code/tym1/LeniaDynamics_rel022/results/signal_4_diagnostic_report.txt"


# Diagnostic images.
SAVE_CROPS = True
SAVE_DIAGNOSTICS = True
DEBUG_DIR = r"C:/code/tym1/LeniaDynamics_rel022/results/debug_signal_4"
CROP_EVERY = 5
CROP_MAX = 40

# Analyze the exact YOLO bbox. Padding 0 = do not add surrounding context.
PADDING = 0.0
MIN_CROP_PX = 0
MAX_HOLD_FRAMES = 3

# Small crop normalization.
NORM_W = 64
NORM_H = 128

# Expected yellow blinking frequencies.
SLOW_HZ = 0.9
FAST_HZ = 1.8
MIN_HZ = 0.3
MAX_HZ = 5.0
MIN_STATE_SEC = 0.12

# Masks and brightness.
GLARE_SUPPRESSION = True
GLARE_THRESHOLD = 240
YELLOW_RATIO_THRESHOLD = 0.003

# Classification.
STEADY_REL_AMP_THRESHOLD = 0.22
MIN_SIGNAL_VALUE = 0.04
DETREND_WINDOW_SEC = 1.4

SHOW_PLOT = False



# =============================================================================
# Detection pipeline - CSV creation before analysis
# =============================================================================

def resolve_csv_path() -> Path:
    """Return the CSV path. If CSV_PATH=None, use OUTPUT_DIR/<video_stem>.csv."""
    if CSV_PATH:
        return Path(CSV_PATH)
    return Path(OUTPUT_DIR) / f"{Path(VIDEO_PATH).stem}.csv"


def load_pipeline_api():
    """Load DetectionPipeline/PipelineConfig from existing project files.

    Supports both common modes:
      1) project as a package: from pipeline import ...
      2) running directly from a folder with detection_pipeline.py and pipeline_config.py
    """
    try:
        from pipeline import DetectionPipeline, Detection, PipelineConfig  # type: ignore
        return DetectionPipeline, Detection, PipelineConfig
    except Exception:
        pass

    try:
        from detection_pipeline import DetectionPipeline  # type: ignore
        from pipeline_config import Detection, PipelineConfig  # type: ignore
        return DetectionPipeline, Detection, PipelineConfig
    except Exception:
        pass

    current_dir = Path(__file__).resolve().parent
    parent_dir = current_dir.parent
    if str(parent_dir) not in sys.path:
        sys.path.insert(0, str(parent_dir))
    package = importlib.import_module(current_dir.name)
    return package.DetectionPipeline, package.Detection, package.PipelineConfig


def pipeline_classify(crop: np.ndarray, det) -> tuple[str, float]:
    """Pipeline hook. The CSV is needed only for bboxes, so return the detection class."""
    return det.class_name, float(det.confidence)


def create_detection_csv_if_needed() -> Path:
    """Run the detection pipeline and create a CSV if enabled in CONFIG."""
    csv_path = resolve_csv_path()

    if not RUN_PIPELINE_BEFORE_ANALYSIS:
        return csv_path

    if csv_path.exists() and not FORCE_RECREATE_CSV:
        print(f"CSV already exists, skipping pipeline: {csv_path}")
        return csv_path

    video_path = Path(VIDEO_PATH)
    model_path = Path(MODEL_PATH)
    output_dir = Path(OUTPUT_DIR)

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model does not exist: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running detection pipeline to create CSV...")
    print(f"Video: {video_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")

    DetectionPipeline, Detection, PipelineConfig = load_pipeline_api()

    config = PipelineConfig(
        reader_fps=READER_FPS,
        detection_fps=DETECTION_FPS,
        detection_confidence=DETECTION_CONFIDENCE,
        imgsz=IMGSZ,
        device=DEVICE,
        classify_classes=[LABEL_FILTER],
        detection_side=DETECTION_SIDE,
        box_relative=BOX_RELATIVE,
    )

    pipeline = DetectionPipeline(str(model_path), config, classifier_fn=pipeline_classify)
    pipeline.process_video(str(video_path), output_dir=str(output_dir), save_csv=True)

    if not csv_path.exists():
        # The pipeline normally saves CSV according to the video stem. If the user set
        # CSV_PATH elsewhere, try to find the standard CSV and copy it.
        default_csv = output_dir / f"{video_path.stem}.csv"
        if CSV_PATH and default_csv.exists() and default_csv != csv_path:
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_path.write_bytes(default_csv.read_bytes())
        elif default_csv.exists():
            csv_path = default_csv
        else:
            raise FileNotFoundError(f"Pipeline finished, but CSV was not found: {csv_path}")

    print(f"CSV ready: {csv_path}")
    return csv_path

# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class CsvDetection:
    frame_no: int
    bbox: tuple[float, float, float, float]
    label: str
    track_id: str
    conf: float


@dataclass(frozen=True)
class FrameSample:
    frame_no: int
    bbox_pixels: tuple[int, int, int, int]
    crop: Optional[np.ndarray]


@dataclass(frozen=True)
class BrightnessSample:
    frame_no: int
    valid: bool
    brightness: float
    brightness_global: float
    brightness_active: float
    yellow_ratio: float
    white_ratio: float
    active_ratio: float
    glare_ratio: float
    norm_crop: Optional[np.ndarray]
    yellow_mask: Optional[np.ndarray]
    white_mask: Optional[np.ndarray]
    active_mask: Optional[np.ndarray]


@dataclass(frozen=True)
class TrackQuality:
    """Souhrnne metriky kvality YOLO tracku a bboxu pro diagnostiku spornych vysledku."""
    track_id: str
    total_video_frames: int
    start_frame: int
    end_frame: int
    span_frames: int
    sample_count: int
    detection_frame_count: int
    valid_crop_count: int
    analysis_duration_sec: float
    track_detection_coverage_span: float
    track_detection_coverage_video: float
    valid_crop_coverage_span: float
    analysis_coverage_video: float
    hold_ratio: float
    median_bbox_width_px: float
    median_bbox_height_px: float
    median_bbox_area_px: float
    bbox_area_cv: float
    bbox_center_step_median_px: float
    bbox_center_step_p95_px: float
    bbox_center_jitter_norm: float
    quality_score: float
    flags: tuple[str, ...]


# =============================================================================
# CSV + video crops
# =============================================================================

def load_pipeline_csv(csv_path: str | Path, label_filter: str = "signal") -> dict[int, list[CsvDetection]]:
    by_frame: dict[int, list[CsvDetection]] = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            label = row.get("label_detection", "")
            track_id = row.get("detection_ID", "")
            if label_filter and label != label_filter:
                continue
            if not track_id:
                continue
            by_frame[int(row["frame_no"])].append(
                CsvDetection(
                    frame_no=int(row["frame_no"]),
                    bbox=(float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])),
                    label=label,
                    track_id=str(track_id),
                    conf=float(row.get("confidence_detection", 0.0) or 0.0),
                )
            )
    return dict(by_frame)


def select_track_id(detections_by_frame: dict[int, list[CsvDetection]], selected_id: Optional[str]) -> str:
    if selected_id is not None:
        return str(selected_id)
    counts = Counter(det.track_id for dets in detections_by_frame.values() for det in dets)
    if not counts:
        raise RuntimeError("CSV contains no detections for the selected label.")
    return counts.most_common(1)[0][0]


def bbox_to_pixels(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple[int, int, int],
    padding: float = 0.0,
    min_crop_px: int = 0,
) -> tuple[int, int, int, int]:
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    # Podpora relativnich souradnic 0..1 i pixelovych souradnic.
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    x1 -= bw * padding
    x2 += bw * padding
    y1 -= bh * padding
    y2 += bh * padding

    if min_crop_px > 0:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        side = max(float(min_crop_px), x2 - x1, y2 - y1)
        x1 = cx - side / 2.0
        x2 = cx + side / 2.0
        y1 = cy - side / 2.0
        y2 = cy + side / 2.0

    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(w, int(round(x2))),
        min(h, int(round(y2))),
    )


def collect_track_samples(
    video_path: str | Path,
    detections_by_frame: dict[int, list[CsvDetection]],
    track_id: str,
    padding: float,
    min_crop_px: int,
    max_hold_frames: int,
) -> tuple[list[FrameSample], float, int, int]:
    track_frames = sorted(
        frame_no
        for frame_no, dets in detections_by_frame.items()
        if any(det.track_id == track_id for det in dets)
    )
    if not track_frames:
        raise RuntimeError(f"Track detection_ID={track_id} was not found in the CSV.")

    start_frame = track_frames[0]
    end_frame = track_frames[-1]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples: list[FrameSample] = []
    last_bbox = None
    last_bbox_frame = -10**9

    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        candidates = [det for det in detections_by_frame.get(frame_no, []) if det.track_id == track_id]
        if candidates:
            det = max(candidates, key=lambda d: d.conf)
            last_bbox = det.bbox
            last_bbox_frame = frame_no

        if last_bbox is not None and (frame_no - last_bbox_frame) <= max_hold_frames:
            x1, y1, x2, y2 = bbox_to_pixels(last_bbox, frame.shape, padding, min_crop_px)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                samples.append(FrameSample(frame_no, (0, 0, 0, 0), None))
            else:
                samples.append(FrameSample(frame_no, (x1, y1, x2, y2), crop.copy()))
        else:
            samples.append(FrameSample(frame_no, (0, 0, 0, 0), None))

        frame_no += 1

    cap.release()
    return samples, fps, start_frame, end_frame


def get_video_frame_count(video_path: str | Path) -> int:
    """Return the total video frame count. If OpenCV cannot determine it, return 0."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return max(0, total)


def compute_track_quality(
    detections_by_frame: dict[int, list[CsvDetection]],
    track_id: str,
    samples: list[FrameSample],
    brightness: list[BrightnessSample],
    total_video_frames: int,
    start_frame: int,
    end_frame: int,
    fps: float,
) -> TrackQuality:
    """Spocte diagnostiku kvality tracku a stability bboxu.

    Tohle pokryva kontrolu problematickych signalu, kde samotny vysledek
    steady/blinking nestaci. U signalu typu 10/11/12/14 je casto potreba vedet,
    jestli algoritmus opravdu analyzoval stabilni crop semaforu, nebo jen kratky
    ci nestabilni YOLO track.
    """
    span_frames = max(1, int(end_frame) - int(start_frame) + 1)
    sample_count = len(samples)
    detection_frames = sorted(
        frame_no
        for frame_no, dets in detections_by_frame.items()
        if any(det.track_id == str(track_id) for det in dets)
    )
    detection_frame_count = len(detection_frames)
    valid_crop_count = sum(1 for b in brightness if b.valid)

    total_video_frames_safe = max(1, int(total_video_frames)) if total_video_frames else 0

    track_detection_coverage_span = detection_frame_count / float(span_frames)
    track_detection_coverage_video = (
        detection_frame_count / float(total_video_frames_safe)
        if total_video_frames_safe else 0.0
    )
    valid_crop_coverage_span = valid_crop_count / float(max(1, sample_count))
    analysis_coverage_video = (
        span_frames / float(total_video_frames_safe)
        if total_video_frames_safe else 0.0
    )

    # Hold ratio vyjadruje, kolik validnich cropu bylo ziskaných podrzenim posledniho bboxu
    # mezi ridsimi YOLO detekcemi. Vyssi hodnota neni nutne spatne, ale muze znamenat,
    # ze vysledek stoji na interpolovanem/udrzovanem bboxu.
    hold_count = max(0, valid_crop_count - detection_frame_count)
    hold_ratio = hold_count / float(max(1, valid_crop_count))

    boxes = []
    for s in samples:
        x1, y1, x2, y2 = s.bbox_pixels
        if s.crop is None or x2 <= x1 or y2 <= y1:
            continue
        w = float(x2 - x1)
        h = float(y2 - y1)
        area = w * h
        cx = float(0.5 * (x1 + x2))
        cy = float(0.5 * (y1 + y2))
        boxes.append((cx, cy, w, h, area))

    if boxes:
        arr = np.asarray(boxes, dtype=np.float64)
        centers = arr[:, 0:2]
        widths = arr[:, 2]
        heights = arr[:, 3]
        areas = arr[:, 4]

        median_bbox_width_px = float(np.median(widths))
        median_bbox_height_px = float(np.median(heights))
        median_bbox_area_px = float(np.median(areas))
        bbox_area_cv = float(np.std(areas) / (np.mean(areas) + 1e-9))

        if len(centers) >= 2:
            steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
            bbox_center_step_median_px = float(np.median(steps))
            bbox_center_step_p95_px = float(np.percentile(steps, 95))
        else:
            bbox_center_step_median_px = 0.0
            bbox_center_step_p95_px = 0.0

        median_diag = float(np.sqrt(median_bbox_width_px ** 2 + median_bbox_height_px ** 2))
        bbox_center_jitter_norm = float(bbox_center_step_p95_px / (median_diag + 1e-9))
    else:
        median_bbox_width_px = 0.0
        median_bbox_height_px = 0.0
        median_bbox_area_px = 0.0
        bbox_area_cv = 0.0
        bbox_center_step_median_px = 0.0
        bbox_center_step_p95_px = 0.0
        bbox_center_jitter_norm = 0.0

    flags: list[str] = []
    if sample_count < max(60, int(round(2.0 * fps))):
        flags.append("SHORT_TRACK")
    if total_video_frames and analysis_coverage_video < 0.25:
        flags.append("SHORT_ANALYSIS_WINDOW")
    if track_detection_coverage_span < 0.20:
        flags.append("LOW_YOLO_DETECTION_COVERAGE")
    if valid_crop_coverage_span < 0.80:
        flags.append("MANY_INVALID_CROPS")
    if hold_ratio > 0.60:
        flags.append("HIGH_BBOX_HOLD_RATIO")
    if bbox_center_jitter_norm > 0.35:
        flags.append("HIGH_BBOX_CENTER_JITTER")
    if bbox_area_cv > 0.65:
        flags.append("UNSTABLE_BBOX_SIZE")

    # Skore 0..1 pro rychlou orientaci v reportu.
    quality_score = 1.0
    quality_score -= 0.20 if "SHORT_TRACK" in flags else 0.0
    quality_score -= 0.15 if "SHORT_ANALYSIS_WINDOW" in flags else 0.0
    quality_score -= 0.20 if "LOW_YOLO_DETECTION_COVERAGE" in flags else 0.0
    quality_score -= 0.20 if "MANY_INVALID_CROPS" in flags else 0.0
    quality_score -= 0.10 if "HIGH_BBOX_HOLD_RATIO" in flags else 0.0
    quality_score -= 0.20 if "HIGH_BBOX_CENTER_JITTER" in flags else 0.0
    quality_score -= 0.15 if "UNSTABLE_BBOX_SIZE" in flags else 0.0
    quality_score = float(np.clip(quality_score, 0.0, 1.0))

    return TrackQuality(
        track_id=str(track_id),
        total_video_frames=int(total_video_frames),
        start_frame=int(start_frame),
        end_frame=int(end_frame),
        span_frames=int(span_frames),
        sample_count=int(sample_count),
        detection_frame_count=int(detection_frame_count),
        valid_crop_count=int(valid_crop_count),
        analysis_duration_sec=float(span_frames / float(fps)) if fps > 0 else 0.0,
        track_detection_coverage_span=float(track_detection_coverage_span),
        track_detection_coverage_video=float(track_detection_coverage_video),
        valid_crop_coverage_span=float(valid_crop_coverage_span),
        analysis_coverage_video=float(analysis_coverage_video),
        hold_ratio=float(hold_ratio),
        median_bbox_width_px=float(median_bbox_width_px),
        median_bbox_height_px=float(median_bbox_height_px),
        median_bbox_area_px=float(median_bbox_area_px),
        bbox_area_cv=float(bbox_area_cv),
        bbox_center_step_median_px=float(bbox_center_step_median_px),
        bbox_center_step_p95_px=float(bbox_center_step_p95_px),
        bbox_center_jitter_norm=float(bbox_center_jitter_norm),
        quality_score=float(quality_score),
        flags=tuple(flags),
    )


# =============================================================================
# Yellow/brightness signal extraction from crop
# =============================================================================

def compress_glare_bgr(img_bgr: np.ndarray, threshold: int = 240, compression: float = 0.35) -> tuple[np.ndarray, float]:
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr, 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = v > int(threshold)
    glare_ratio = float(np.mean(mask)) if mask.size else 0.0

    if np.any(mask):
        vf = v.astype(np.float32)
        vf[mask] = threshold + (vf[mask] - threshold) * float(compression)
        v = np.clip(vf, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    return img_bgr, glare_ratio


def compute_masks(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Yellow/orange; intentionally wider range due to camera response and saturation.
    yellow_mask = cv2.inRange(
        hsv,
        np.array([8, 30, 45], dtype=np.uint8),
        np.array([50, 255, 255], dtype=np.uint8),
    )

    # Overexposed/white-yellow part of the lamp.
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 170], dtype=np.uint8),
        np.array([179, 110, 255], dtype=np.uint8),
    )

    active_mask = cv2.bitwise_or(yellow_mask, white_mask)
    total = max(1, img_bgr.shape[0] * img_bgr.shape[1])

    yellow_ratio = float(cv2.countNonZero(yellow_mask) / total)
    white_ratio = float(cv2.countNonZero(white_mask) / total)
    active_ratio = float(cv2.countNonZero(active_mask) / total)

    return yellow_mask, white_mask, active_mask, yellow_ratio, white_ratio, active_ratio


def extract_brightness_from_crop(crop_bgr: Optional[np.ndarray]) -> BrightnessSample:
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.ndim != 3:
        return BrightnessSample(-1, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None)

    if crop_bgr.shape[0] < 2 or crop_bgr.shape[1] < 2:
        return BrightnessSample(-1, False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None, None)

    norm = cv2.resize(crop_bgr, (NORM_W, NORM_H), interpolation=cv2.INTER_LINEAR)

    yellow_mask, white_mask, active_mask, yellow_ratio, white_ratio, active_ratio = compute_masks(norm)

    if GLARE_SUPPRESSION:
        img_for_v, glare_ratio = compress_glare_bgr(norm, threshold=GLARE_THRESHOLD)
    else:
        img_for_v, glare_ratio = norm, 0.0

    hsv = cv2.cvtColor(img_for_v, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0

    # Global brightness of the whole bbox.
    mean_v = float(np.mean(v))
    p90 = float(np.percentile(v, 90))
    p98 = float(np.percentile(v, 98))
    p995 = float(np.percentile(v, 99.5))
    brightness_global = 0.35 * mean_v + 0.25 * p90 + 0.25 * p98 + 0.15 * p995

    # Brightness only in the active mask. This is diagnostically and practically important:
    # if global brightness does not blink but the active region does, measuring the whole bbox was the issue.
    active_bool = active_mask > 0
    if np.any(active_bool):
        active_values = v[active_bool]
        brightness_active = float(0.60 * np.mean(active_values) + 0.40 * np.percentile(active_values, 90))
    else:
        brightness_active = 0.0

    # Combined signal: when the mask is available, prefer the active area; otherwise fall back to global percentiles.
    active_support = float(np.clip(active_ratio / max(YELLOW_RATIO_THRESHOLD * 10.0, 1e-6), 0.0, 1.0))
    if active_ratio >= YELLOW_RATIO_THRESHOLD:
        brightness = 0.70 * brightness_active + 0.20 * brightness_global + 0.10 * active_support
    else:
        brightness = 0.88 * brightness_global + 0.12 * active_support

    return BrightnessSample(
        -1,
        True,
        float(brightness),
        float(brightness_global),
        float(brightness_active),
        yellow_ratio,
        white_ratio,
        active_ratio,
        glare_ratio,
        norm,
        yellow_mask,
        white_mask,
        active_mask,
    )


def extract_series(samples: list[FrameSample]) -> list[BrightnessSample]:
    out: list[BrightnessSample] = []
    for s in samples:
        b = extract_brightness_from_crop(s.crop)
        out.append(
            BrightnessSample(
                frame_no=s.frame_no,
                valid=b.valid,
                brightness=b.brightness,
                brightness_global=b.brightness_global,
                brightness_active=b.brightness_active,
                yellow_ratio=b.yellow_ratio,
                white_ratio=b.white_ratio,
                active_ratio=b.active_ratio,
                glare_ratio=b.glare_ratio,
                norm_crop=b.norm_crop,
                yellow_mask=b.yellow_mask,
                white_mask=b.white_mask,
                active_mask=b.active_mask,
            )
        )
    return out


# =============================================================================
# Signal analysis
# =============================================================================

def fill_invalid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).copy()
    valid = np.asarray(valid, dtype=bool)
    if values.size == 0:
        return values
    if not np.any(valid):
        return np.zeros_like(values)

    first = int(np.where(valid)[0][0])
    values[:first] = values[first]
    for i in range(first + 1, values.size):
        if not valid[i]:
            values[i] = values[i - 1]
    return values


def moving_average_reflect(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    window = max(1, int(window))
    if window > x.size:
        window = x.size
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def detrend(values: np.ndarray, fps: float, window_sec: float = DETREND_WINDOW_SEC) -> tuple[np.ndarray, np.ndarray]:
    window = max(5, int(round(window_sec * fps)) | 1)
    if window >= values.size:
        window = max(3, (values.size // 2) * 2 - 1)
    if window < 3:
        trend = np.full_like(values, float(np.mean(values)) if values.size else 0.0)
    else:
        trend = moving_average_reflect(values, window)
    return values - trend, trend


def remove_short_binary_states(binary: np.ndarray, min_len_frames: int) -> np.ndarray:
    y = np.asarray(binary, dtype=bool).copy()
    n = y.size
    min_len_frames = max(1, int(min_len_frames))
    if n == 0 or min_len_frames <= 1:
        return y

    for _ in range(10):
        starts = [0]
        for i in range(1, n):
            if y[i] != y[i - 1]:
                starts.append(i)
        ends = starts[1:] + [n]
        changed = False

        for run_i, (start, end) in enumerate(zip(starts, ends)):
            if end - start >= min_len_frames or len(starts) <= 1:
                continue
            if run_i == 0:
                repl = y[end] if end < n else y[start]
            elif run_i == len(starts) - 1:
                repl = y[start - 1]
            else:
                prev_len = start - starts[run_i - 1]
                next_len = ends[run_i + 1] - end
                repl = y[start - 1] if prev_len >= next_len else y[end]
            y[start:end] = repl
            changed = True

        if not changed:
            break
    return y


def threshold_edges(detrended_signal: np.ndarray, fps: float) -> dict:
    if detrended_signal.size == 0:
        return {
            "threshold": 0.0,
            "binary": np.array([], dtype=bool),
            "rising": np.array([], dtype=int),
            "falling": np.array([], dtype=int),
            "edge_frequency": 0.0,
            "filtered": detrended_signal,
        }

    smooth_win = max(3, int(round(0.10 * fps)) | 1)
    if smooth_win >= detrended_signal.size:
        smooth_win = max(3, (detrended_signal.size // 2) * 2 - 1)
    filtered = moving_average_reflect(detrended_signal, smooth_win) if smooth_win >= 3 else detrended_signal.copy()

    low = float(np.percentile(filtered, 20))
    high = float(np.percentile(filtered, 80))
    thr = 0.5 * (low + high)
    binary = filtered > thr
    binary = remove_short_binary_states(binary, int(round(MIN_STATE_SEC * fps)))

    rising = np.where((binary[1:] == 1) & (binary[:-1] == 0))[0] + 1
    falling = np.where((binary[1:] == 0) & (binary[:-1] == 1))[0] + 1

    freq = 0.0
    if rising.size >= 2:
        periods = np.diff(rising) / fps
        periods = periods[periods > 0]
        if periods.size:
            freq = float(1.0 / np.median(periods))

    return {
        "threshold": thr,
        "binary": binary,
        "rising": rising,
        "falling": falling,
        "edge_frequency": freq,
        "filtered": filtered,
    }


def fft_analysis(detrended_signal: np.ndarray, fps: float) -> dict:
    n = detrended_signal.size
    if n < 15:
        return {
            "ok": False,
            "dominant": 0.0,
            "ratio_slow": 0.0,
            "ratio_fast": 0.0,
            "energy_slow": 0.0,
            "energy_fast": 0.0,
            "freqs": np.array([]),
            "spectrum": np.array([]),
        }

    x = detrended_signal.astype(np.float64)
    x = x - np.mean(x)
    if np.std(x) < 1e-9:
        return {
            "ok": False,
            "dominant": 0.0,
            "ratio_slow": 0.0,
            "ratio_fast": 0.0,
            "energy_slow": 0.0,
            "energy_fast": 0.0,
            "freqs": np.array([]),
            "spectrum": np.array([]),
        }

    window = np.hanning(n)
    padded_n = max(1024, int(2 ** np.ceil(np.log2(max(16, n * 4)))))
    spectrum = np.abs(np.fft.rfft(x * window, n=padded_n)) ** 2
    freqs = np.fft.rfftfreq(padded_n, d=1.0 / fps)

    valid = (freqs >= MIN_HZ) & (freqs <= MAX_HZ)
    slow = (freqs >= 0.60) & (freqs <= 1.25)
    fast = (freqs >= 1.35) & (freqs <= 2.25)

    energy_valid = float(np.sum(spectrum[valid]) + 1e-12)
    energy_slow = float(np.sum(spectrum[slow]))
    energy_fast = float(np.sum(spectrum[fast]))

    dominant = 0.0
    if np.any(valid):
        idxs = np.where(valid)[0]
        dominant = float(freqs[idxs[int(np.argmax(spectrum[idxs]))]])

    return {
        "ok": True,
        "dominant": dominant,
        "ratio_slow": float(energy_slow / energy_valid),
        "ratio_fast": float(energy_fast / energy_valid),
        "energy_slow": energy_slow,
        "energy_fast": energy_fast,
        "freqs": freqs,
        "spectrum": spectrum,
    }


def classify_blink(values: np.ndarray, active: np.ndarray, fps: float) -> dict:
    """Klasifikace blikani zluteho svetla podle diagnostickych dat z batch testu.

    Zmena proti predchozi verzi:
      - relativni amplituda uz neni hlavni tvrdy filtr 0.22,
      - blikani se potvrzuje pres kombinaci FFT, hran a relativni amplitudy,
      - pomale a rychle blikani maji oddelene rozhodovaci vetve,
      - pro kratke tracky existuje opatrna FFT fallback vetev,
      - false-positive typ one_yellow_signal/signal_3 se potlacuje tim, ze pomale
        blikani musi mit edge_frequency aspon kolem 0.9 Hz.
    """
    detrended_signal, trend = detrend(values, fps)
    edge = threshold_edges(detrended_signal, fps)
    fft = fft_analysis(detrended_signal, fps)

    max_value = float(np.max(values)) if values.size else 0.0
    max_active = float(np.max(active)) if active.size else 0.0
    mean_v = float(np.mean(values)) if values.size else 0.0
    rel_amp = float((np.percentile(values, 95) - np.percentile(values, 5)) / (mean_v + 1e-6)) if values.size else 0.0

    valid_signal = max_value >= MIN_SIGNAL_VALUE or max_active >= YELLOW_RATIO_THRESHOLD
    if not valid_signal:
        return {
            "state": "no_signal",
            "frequency_hz": 0.0,
            "confidence": 0.0,
            "reason": "max brightness i active_ratio jsou pod prahem",
            "edge": edge,
            "fft": fft,
            "detrended": detrended_signal,
            "trend": trend,
            "rel_amp": rel_amp,
        }

    num_samples = int(values.size)
    num_edges = int(len(edge.get("rising", [])) + len(edge.get("falling", [])))
    fft_ok = bool(fft.get("ok", False))
    f_fft = float(fft.get("dominant", 0.0)) if fft_ok else 0.0
    f_edge = float(edge.get("edge_frequency", 0.0))
    ratio_slow = float(fft.get("ratio_slow", 0.0)) if fft_ok else 0.0
    ratio_fast = float(fft.get("ratio_fast", 0.0)) if fft_ok else 0.0
    energy_slow = float(fft.get("energy_slow", 0.0)) if fft_ok else 0.0
    energy_fast = float(fft.get("energy_fast", 0.0)) if fft_ok else 0.0
    edge_fft_diff = abs(f_edge - f_fft) if f_edge > 0.0 else 999.0

    if not fft_ok:
        return {
            "state": "uncertain",
            "frequency_hz": 0.0,
            "confidence": 0.0,
            "reason": "FFT nema dost dat nebo ma nulovou varianci",
            "edge": edge,
            "fft": fft,
            "detrended": detrended_signal,
            "trend": trend,
            "rel_amp": rel_amp,
        }

    # -------------------------------------------------------------------------
    # Pravidla naladena podle batch vysledku 34 videi:
    # puvodne: TP=1, FP=1, TN=16, FN=16, accuracy ~= 50 %
    # tato pravidla na stejne tabulce metrik: TP=11, FP=0, TN=17, FN=6,
    # accuracy ~= 82 %. Ber to jako kalibraci na aktualni dataset, ne jako
    # obecny fyzikalni model pro vsechny kamery/sceny.
    # -------------------------------------------------------------------------

    # Ciste pomale blikani kolem 0.9-1.1 Hz.
    # Rel_amp muze byt male, proto je snizene na 0.07; kompenzuje se shodou
    # FFT a edge_frequency. Dolni mez f_edge=0.90 potlacuje false-positive
    # typ one_yellow_signal/signal_3, kde f_edge vyslo jen ~0.545 Hz.
    blink_evidence_slow = (
        num_edges >= 4
        and rel_amp >= 0.07
        and 0.85 <= f_fft <= 1.15
        and 0.90 <= f_edge <= 1.25
        and edge_fft_diff <= 0.30
        and ratio_slow >= 0.30
    )

    # Rychle blikani / vyssi harmonicka.
    # U rychleho blikani muze edge_frequency selhat kvuli malemu poctu hran,
    # proto se zde vic opirame o FFT energii v rychlem pasmu.
    blink_evidence_fast = (
        num_edges >= 1
        and rel_amp >= 0.05
        and 1.30 <= f_fft <= 2.30
        and ratio_fast >= 0.25
    )

    # Kratky track fallback.
    # Kdyz je track kratky, hran je malo a edge_frequency muze byt nestabilni.
    # Tato vetev pomaha u kratkych blikajicich signalu, ale je omezena na
    # max 60 vzorku, aby nechytala delsi steady sekvence s pomalou expozicni zmenou.
    blink_evidence_short_track = (
        num_samples <= 60
        and ratio_slow >= 0.40
        and 0.60 <= f_fft <= 1.10
    )

    has_blink_evidence = blink_evidence_slow or blink_evidence_fast or blink_evidence_short_track

    if has_blink_evidence:
        if blink_evidence_fast and not blink_evidence_slow and not blink_evidence_short_track:
            state = "blinking_yellow_fast"
            f = f_fft
            closeness = float(np.clip(1.0 - abs(f - FAST_HZ) / 0.60, 0.0, 1.0))
            confidence = float(np.clip(
                0.35 * ratio_fast
                + 0.25 * closeness
                + 0.20 * min(rel_amp / 0.25, 1.0)
                + 0.10 * min(num_edges / 6.0, 1.0)
                + 0.10,
                0.0,
                1.0,
            ))
            reason = (
                f"rychle blikani potvrzeno kalibrovanymi pravidly; "
                f"edges={num_edges}, samples={num_samples}, f_fft={f_fft:.3f}, "
                f"f_edge={f_edge:.3f}, rel_amp={rel_amp:.3f}, "
                f"slow_ratio={ratio_slow:.3f}, fast_ratio={ratio_fast:.3f}"
            )
        else:
            state = "blinking_yellow_slow"
            f = f_fft
            closeness = float(np.clip(1.0 - abs(f - SLOW_HZ) / 0.45, 0.0, 1.0))
            agreement = float(np.clip(1.0 - edge_fft_diff / 0.30, 0.0, 1.0)) if f_edge > 0 else 0.35
            short_bonus = 0.10 if blink_evidence_short_track else 0.0
            confidence = float(np.clip(
                0.35 * ratio_slow
                + 0.20 * closeness
                + 0.20 * agreement
                + 0.15 * min(rel_amp / 0.20, 1.0)
                + short_bonus
                + 0.05,
                0.0,
                1.0,
            ))
            reason = (
                f"pomale blikani potvrzeno kalibrovanymi pravidly; "
                f"edges={num_edges}, samples={num_samples}, f_fft={f_fft:.3f}, "
                f"f_edge={f_edge:.3f}, rel_amp={rel_amp:.3f}, "
                f"slow_ratio={ratio_slow:.3f}, fast_ratio={ratio_fast:.3f}, "
                f"short_track={blink_evidence_short_track}"
            )

        return {
            "state": state,
            "frequency_hz": float(f),
            "confidence": float(confidence),
            "reason": reason,
            "edge": edge,
            "fft": fft,
            "detrended": detrended_signal,
            "trend": trend,
            "rel_amp": rel_amp,
        }

    # Bez potvrzene periodicity vracime steady_yellow, pokud je signal jasny.
    # Rel_amp muze byt i vysoka, ale pokud frekvence/hrany nesedi, casto jde o
    # pohyb bboxu, expozici nebo zmenu uhlu, ne o pravidelne blikani.
    if mean_v > 0.08:
        return {
            "state": "steady_yellow",
            "frequency_hz": 0.0,
            "confidence": float(np.clip(1.0 - min(rel_amp, 1.0), 0.0, 1.0)),
            "reason": (
                f"blikani nepotvrzeno kalibrovanymi pravidly; "
                f"edges={num_edges}, samples={num_samples}, f_fft={f_fft:.3f}, "
                f"f_edge={f_edge:.3f}, diff={edge_fft_diff:.3f}, rel_amp={rel_amp:.3f}, "
                f"slow_ratio={ratio_slow:.3f}, fast_ratio={ratio_fast:.3f}, "
                f"energy_slow={energy_slow:.6f}, energy_fast={energy_fast:.6f}"
            ),
            "edge": edge,
            "fft": fft,
            "detrended": detrended_signal,
            "trend": trend,
            "rel_amp": rel_amp,
        }

    return {
        "state": "uncertain",
        "frequency_hz": f_fft,
        "confidence": 0.25,
        "reason": (
            f"signal je slaby a nesplnil kalibrovana pravidla blikani; "
            f"edges={num_edges}, samples={num_samples}, f_fft={f_fft:.3f}, "
            f"f_edge={f_edge:.3f}, rel_amp={rel_amp:.3f}, "
            f"slow_ratio={ratio_slow:.3f}, fast_ratio={ratio_fast:.3f}"
        ),
        "edge": edge,
        "fft": fft,
        "detrended": detrended_signal,
        "trend": trend,
        "rel_amp": rel_amp,
    }


# =============================================================================
# Diagnostic outputs
# =============================================================================

def colorize_mask(mask: np.ndarray, color_bgr: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    out[mask > 0] = color_bgr
    return out


def put_label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    return cv2.copyMakeBorder(img, 0, height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def save_diagnostic_panels(samples: list[FrameSample], brightness: list[BrightnessSample]) -> None:
    if not SAVE_DIAGNOSTICS:
        return

    out_dir = Path(DEBUG_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, (s, b) in enumerate(zip(samples, brightness)):
        if idx % max(1, CROP_EVERY) != 0:
            continue
        if saved >= CROP_MAX:
            break
        if s.crop is None or b.norm_crop is None or b.yellow_mask is None or b.white_mask is None or b.active_mask is None:
            continue

        scale = 6
        raw_big = cv2.resize(s.crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        norm_big = cv2.resize(b.norm_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        yellow_big = cv2.resize(colorize_mask(b.yellow_mask, (0, 255, 255)), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        white_big = cv2.resize(colorize_mask(b.white_mask, (255, 255, 255)), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        active_big = cv2.resize(colorize_mask(b.active_mask, (0, 255, 0)), None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        raw_big = put_label(raw_big, "raw crop")
        norm_big = put_label(norm_big, "normalized")
        yellow_big = put_label(yellow_big, "yellow mask")
        white_big = put_label(white_big, "white mask")
        active_big = put_label(active_big, "active mask")

        h = max(raw_big.shape[0], norm_big.shape[0], yellow_big.shape[0], white_big.shape[0], active_big.shape[0])
        panel = np.hstack([
            pad_to_height(raw_big, h),
            pad_to_height(norm_big, h),
            pad_to_height(yellow_big, h),
            pad_to_height(white_big, h),
            pad_to_height(active_big, h),
        ])

        info = (
            f"frame={s.frame_no} bbox={s.bbox_pixels} "
            f"B={b.brightness:.3f} Bg={b.brightness_global:.3f} Ba={b.brightness_active:.3f} "
            f"Y={b.yellow_ratio:.4f} W={b.white_ratio:.4f} A={b.active_ratio:.4f} glare={b.glare_ratio:.4f}"
        )
        cv2.rectangle(panel, (0, h - 28), (panel.shape[1], h), (0, 0, 0), -1)
        cv2.putText(panel, info, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        out_path = out_dir / f"diagnostic_{saved:03d}_frame_{s.frame_no:06d}.jpg"
        cv2.imwrite(str(out_path), panel)
        saved += 1

    print(f"Saved diagnostic panels: {saved} -> {out_dir}")


def save_events_csv(path: str | Path, frame_numbers: np.ndarray, edge_result: dict, fps: float, start_frame: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rising = set(int(x) for x in edge_result.get("rising", []))
    falling = set(int(x) for x in edge_result.get("falling", []))
    binary = edge_result.get("binary", np.array([], dtype=bool))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["index", "frame_no", "time_s", "state", "event"])
        for i, frame_no in enumerate(frame_numbers):
            event = ""
            if i in rising:
                event = "ON"
            elif i in falling:
                event = "OFF"
            state = int(binary[i]) if i < len(binary) else 0
            time_s = (int(frame_no) - int(start_frame)) / float(fps)
            writer.writerow([i, int(frame_no), f"{time_s:.6f}", state, event])


def save_signals_csv(
    path: str | Path,
    samples: list[FrameSample],
    brightness: list[BrightnessSample],
    values: np.ndarray,
    active: np.ndarray,
    result: dict,
    fps: float,
    start_frame: int,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trend = result.get("trend", np.zeros_like(values))
    detrended_signal = result.get("detrended", np.zeros_like(values))
    edge = result.get("edge", {})
    filtered = edge.get("filtered", np.zeros_like(values))
    binary = edge.get("binary", np.zeros_like(values, dtype=bool))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "index", "frame_no", "time_s", "x1", "y1", "x2", "y2", "valid",
            "brightness", "brightness_global", "brightness_active", "trend", "detrended", "filtered", "binary",
            "yellow_ratio", "white_ratio", "active_ratio", "glare_ratio",
        ])
        for i, (s, b) in enumerate(zip(samples, brightness)):
            x1, y1, x2, y2 = s.bbox_pixels
            time_s = (int(s.frame_no) - int(start_frame)) / float(fps)
            writer.writerow([
                i,
                int(s.frame_no),
                f"{time_s:.6f}",
                x1,
                y1,
                x2,
                y2,
                int(b.valid),
                f"{values[i]:.9f}",
                f"{b.brightness_global:.9f}",
                f"{b.brightness_active:.9f}",
                f"{trend[i]:.9f}" if i < len(trend) else "",
                f"{detrended_signal[i]:.9f}" if i < len(detrended_signal) else "",
                f"{filtered[i]:.9f}" if i < len(filtered) else "",
                int(binary[i]) if i < len(binary) else 0,
                f"{b.yellow_ratio:.9f}",
                f"{b.white_ratio:.9f}",
                f"{active[i]:.9f}",
                f"{b.glare_ratio:.9f}",
            ])


def save_report(
    path: str | Path,
    result: dict,
    values: np.ndarray,
    global_values: np.ndarray,
    active_values: np.ndarray,
    active_ratio: np.ndarray,
    fps: float,
    start_frame: int,
    end_frame: int,
    sample_count: int,
    track_quality: Optional[TrackQuality] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    edge = result.get("edge", {})
    fft = result.get("fft", {})

    text = []
    text.append("Yellow blink diagnostic report")
    text.append("================================")
    text.append(f"result: {result.get('state')}")
    text.append(f"frequency_hz: {result.get('frequency_hz'):.6f}")
    text.append(f"confidence: {result.get('confidence'):.6f}")
    text.append(f"reason: {result.get('reason', '')}")
    text.append("")
    text.append(f"fps: {fps:.6f}")
    text.append(f"frames: {start_frame}..{end_frame}")
    text.append(f"sample_count: {sample_count}")
    text.append("")
    text.append("signal stats")
    text.append(f"brightness min/mean/max: {np.min(values):.6f} / {np.mean(values):.6f} / {np.max(values):.6f}")
    text.append(f"global brightness min/mean/max: {np.min(global_values):.6f} / {np.mean(global_values):.6f} / {np.max(global_values):.6f}")
    text.append(f"active brightness min/mean/max: {np.min(active_values):.6f} / {np.mean(active_values):.6f} / {np.max(active_values):.6f}")
    text.append(f"active_ratio min/mean/max: {np.min(active_ratio):.6f} / {np.mean(active_ratio):.6f} / {np.max(active_ratio):.6f}")
    text.append(f"relative amplitude: {result.get('rel_amp', 0.0):.6f}")
    text.append("")
    text.append("edge analysis")
    text.append(f"rising edges: {len(edge.get('rising', []))}")
    text.append(f"falling edges: {len(edge.get('falling', []))}")
    text.append(f"edge frequency: {edge.get('edge_frequency', 0.0):.6f}")
    text.append("")
    text.append("fft analysis")
    text.append(f"fft ok: {fft.get('ok', False)}")
    text.append(f"dominant frequency: {fft.get('dominant', 0.0):.6f}")
    text.append(f"ratio slow 0.6-1.25Hz: {fft.get('ratio_slow', 0.0):.6f}")
    text.append(f"ratio fast 1.35-2.25Hz: {fft.get('ratio_fast', 0.0):.6f}")
    text.append(f"energy slow: {fft.get('energy_slow', 0.0):.6f}")
    text.append(f"energy fast: {fft.get('energy_fast', 0.0):.6f}")
    text.append("")

    if track_quality is not None:
        text.append("track / bbox quality")
        text.append(f"track_id: {track_quality.track_id}")
        text.append(f"total_video_frames: {track_quality.total_video_frames}")
        text.append(f"track_span_frames: {track_quality.span_frames}")
        text.append(f"analysis_duration_sec: {track_quality.analysis_duration_sec:.6f}")
        text.append(f"detection_frame_count: {track_quality.detection_frame_count}")
        text.append(f"valid_crop_count: {track_quality.valid_crop_count}")
        text.append(f"track_detection_coverage_span: {track_quality.track_detection_coverage_span:.6f}")
        text.append(f"track_detection_coverage_video: {track_quality.track_detection_coverage_video:.6f}")
        text.append(f"valid_crop_coverage_span: {track_quality.valid_crop_coverage_span:.6f}")
        text.append(f"analysis_coverage_video: {track_quality.analysis_coverage_video:.6f}")
        text.append(f"hold_ratio: {track_quality.hold_ratio:.6f}")
        text.append(f"median_bbox_width_px: {track_quality.median_bbox_width_px:.3f}")
        text.append(f"median_bbox_height_px: {track_quality.median_bbox_height_px:.3f}")
        text.append(f"median_bbox_area_px: {track_quality.median_bbox_area_px:.3f}")
        text.append(f"bbox_area_cv: {track_quality.bbox_area_cv:.6f}")
        text.append(f"bbox_center_step_median_px: {track_quality.bbox_center_step_median_px:.3f}")
        text.append(f"bbox_center_step_p95_px: {track_quality.bbox_center_step_p95_px:.3f}")
        text.append(f"bbox_center_jitter_norm: {track_quality.bbox_center_jitter_norm:.6f}")
        text.append(f"track_quality_score: {track_quality.quality_score:.6f}")
        text.append(f"track_quality_flags: {', '.join(track_quality.flags) if track_quality.flags else 'OK'}")
        text.append("")

    text.append("Result interpretation:")
    if result.get("state") == "steady_yellow":
        text.append("- The algorithm returned steady_yellow because a stable periodic blinking component was not confirmed.")
        text.append("- If this is an expected blinking video, check track / bbox quality above.")
        text.append("- LOW_YOLO_DETECTION_COVERAGE, SHORT_TRACK, HIGH_BBOX_CENTER_JITTER or UNSTABLE_BBOX_SIZE means the problem is probably in detection/tracking, not in the blinking threshold.")
        text.append("- If track_quality_flags=OK and the active mask hits the lamp, classification decision rules should be tuned.")
    elif result.get("state") == "uncertain":
        text.append("- The signal changes, but the frequency is not stable in either the 0.9 Hz or 1.8 Hz band.")
        text.append("- For uncertain results, decide according to track_quality_score and the detrended/FFT plot.")
    else:
        text.append("- Blinking was confirmed. Still check track_quality_flags if confidence is low.")

    path.write_text("\n".join(text), encoding="utf-8")


def save_plot(
    path: str | Path,
    times: np.ndarray,
    values: np.ndarray,
    global_values: np.ndarray,
    active_values: np.ndarray,
    active_ratio: np.ndarray,
    result: dict,
) -> None:
    if plt is None:
        print("matplotlib neni dostupny, graf se neulozi.")
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trend = result.get("trend", np.zeros_like(values))
    detrended_signal = result.get("detrended", np.zeros_like(values))
    edge = result.get("edge", {})
    filtered = edge.get("filtered", np.zeros_like(values))
    binary = edge.get("binary", np.zeros_like(values, dtype=bool))
    fft = result.get("fft", {})

    fig = plt.figure(figsize=(15, 12))

    ax1 = fig.add_subplot(4, 1, 1)
    ax1.plot(times, values, label="combined brightness")
    ax1.plot(times, global_values, label="global brightness", alpha=0.8)
    ax1.plot(times, active_values, label="active-mask brightness", alpha=0.8)
    ax1.plot(times, trend, label="trend", alpha=0.8)
    ax1.set_title(f"Yellow blink analysis: {result['state']}, f={result['frequency_hz']:.3f} Hz, conf={result['confidence']:.2f}")
    ax1.set_ylabel("brightness")
    ax1.grid(True)
    ax1.legend(loc="best")

    ax2 = fig.add_subplot(4, 1, 2)
    ax2.plot(times, active_ratio, label="active ratio")
    ax2.set_ylabel("active ratio")
    ax2.grid(True)
    ax2.legend(loc="best")

    ax3 = fig.add_subplot(4, 1, 3)
    ax3.plot(times, detrended_signal, label="detrended")
    ax3.plot(times, filtered, label="filtered")
    if len(binary) == len(times):
        ymin, ymax = ax3.get_ylim()
        scaled_binary = binary.astype(float)
        scaled_binary = scaled_binary * (ymax - ymin) * 0.25 + ymin
        ax3.plot(times, scaled_binary, label="binary ON/OFF", alpha=0.7)
    ax3.set_ylabel("detrended")
    ax3.grid(True)
    ax3.legend(loc="best")

    ax4 = fig.add_subplot(4, 1, 4)
    freqs = fft.get("freqs", np.array([]))
    spectrum = fft.get("spectrum", np.array([]))
    if len(freqs) and len(spectrum):
        mask = (freqs >= 0.0) & (freqs <= MAX_HZ)
        spec = spectrum[mask]
        if spec.size and np.max(spec) > 0:
            spec = spec / np.max(spec)
        ax4.plot(freqs[mask], spec, label="FFT normalized")
        ax4.axvline(SLOW_HZ, linestyle="--", label="0.9 Hz")
        ax4.axvline(FAST_HZ, linestyle="--", label="1.8 Hz")
    ax4.set_xlabel("frequency [Hz]")
    ax4.set_ylabel("FFT power")
    ax4.grid(True)
    ax4.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if SHOW_PLOT:
        plt.show()
    plt.close(fig)
    print(f"Plot saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    video_path = Path(VIDEO_PATH)
    csv_path = create_detection_csv_if_needed()

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV neexistuje: {csv_path}")

    detections = load_pipeline_csv(csv_path, LABEL_FILTER)
    track_id = select_track_id(detections, SELECTED_ID)
    print(f"Analyzing detection_ID={track_id}")

    samples, fps, start_frame, end_frame = collect_track_samples(
        video_path=video_path,
        detections_by_frame=detections,
        track_id=track_id,
        padding=PADDING,
        min_crop_px=MIN_CROP_PX,
        max_hold_frames=MAX_HOLD_FRAMES,
    )

    if not samples:
        raise RuntimeError("No crops were loaded.")

    brightness_samples = extract_series(samples)
    save_diagnostic_panels(samples, brightness_samples)

    total_video_frames = get_video_frame_count(video_path)
    track_quality = compute_track_quality(
        detections_by_frame=detections,
        track_id=track_id,
        samples=samples,
        brightness=brightness_samples,
        total_video_frames=total_video_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        fps=fps,
    )

    frame_numbers = np.array([b.frame_no for b in brightness_samples], dtype=int)
    valid = np.array([b.valid for b in brightness_samples], dtype=bool)

    values_raw = np.array([b.brightness for b in brightness_samples], dtype=np.float64)
    global_raw = np.array([b.brightness_global for b in brightness_samples], dtype=np.float64)
    active_brightness_raw = np.array([b.brightness_active for b in brightness_samples], dtype=np.float64)
    active_ratio_raw = np.array([b.active_ratio for b in brightness_samples], dtype=np.float64)

    values = fill_invalid(values_raw, valid)
    global_values = fill_invalid(global_raw, valid)
    active_values = fill_invalid(active_brightness_raw, valid)
    active_ratio = fill_invalid(active_ratio_raw, valid)

    result = classify_blink(values, active_ratio, fps)
    times = (frame_numbers - start_frame) / float(fps)

    print("------------------------------")
    print(f"Video FPS: {fps:.3f}")
    print(f"Frames: {start_frame} az {end_frame}")
    print(f"Number of samples: {len(samples)}")
    print(f"Result: {result['state']}")
    print(f"Frequency: {result['frequency_hz']:.3f} Hz")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Reason: {result.get('reason', '')}")
    print(f"Rel amplitude: {result.get('rel_amp', 0.0):.3f}")
    print(f"Max active ratio: {float(np.max(active_ratio)):.6f}")
    print(f"Track coverage span: {track_quality.track_detection_coverage_span:.3f}")
    print(f"Valid crop coverage: {track_quality.valid_crop_coverage_span:.3f}")
    print(f"Analysis coverage video: {track_quality.analysis_coverage_video:.3f}")
    print(f"BBox jitter norm: {track_quality.bbox_center_jitter_norm:.3f}")
    print(f"BBox area CV: {track_quality.bbox_area_cv:.3f}")
    print(f"Track quality score: {track_quality.quality_score:.3f}")
    print(f"Track quality flags: {', '.join(track_quality.flags) if track_quality.flags else 'OK'}")
    print(f"Edges ON/OFF: {len(result.get('edge', {}).get('rising', []))}/{len(result.get('edge', {}).get('falling', []))}")
    print(f"FFT dominant: {result.get('fft', {}).get('dominant', 0.0):.3f} Hz")
    print(f"FFT slow ratio: {result.get('fft', {}).get('ratio_slow', 0.0):.3f}")
    print(f"FFT fast ratio: {result.get('fft', {}).get('ratio_fast', 0.0):.3f}")
    print("------------------------------")

    if EVENTS_CSV:
        save_events_csv(EVENTS_CSV, frame_numbers, result.get("edge", {}), fps, start_frame)
        print(f"Events CSV saved: {EVENTS_CSV}")

    if SIGNALS_CSV:
        save_signals_csv(SIGNALS_CSV, samples, brightness_samples, values, active_ratio, result, fps, start_frame)
        print(f"Diagnostic signals CSV saved: {SIGNALS_CSV}")

    if REPORT_TXT:
        save_report(
            REPORT_TXT,
            result,
            values,
            global_values,
            active_values,
            active_ratio,
            fps,
            start_frame,
            end_frame,
            len(samples),
            track_quality=track_quality,
        )
        print(f"Diagnostic report saved: {REPORT_TXT}")

    if OUTPUT_PLOT:
        save_plot(OUTPUT_PLOT, times, values, global_values, active_values, active_ratio, result)


if __name__ == "__main__":
    main()


# =============================================================================
# Runtime adapter used by run_pipeline.py
# =============================================================================

@dataclass(frozen=True)
class BlinkRuntimeResult:
    """Compact result returned by YellowCropBlinkDetector.update()."""
    state: str
    confidence: float
    frequency_hz: float = 0.0
    reason: str = ""


class YellowCropBlinkDetector:
    """Online wrapper for use inside DetectionPipeline.classifier_fn.

    The original file above is a standalone diagnostic script: it can run the
    detector pipeline, load the generated CSV, crop one selected track and save
    diagnostic plots/CSVs/reports. The project entry point `run_pipeline.py`
    expects an online object with `update(crop, track_id)`; this adapter exposes
    that API while reusing the same mask extraction and `classify_blink()` logic.
    """

    def __init__(
        self,
        fps: float = 30.0,
        history_seconds: float = 4.0,
        norm_w: int = 64,
        norm_h: int = 128,
        slow_hz: float = 0.9,
        fast_hz: float = 1.8,
        debug: bool = False,
    ) -> None:
        self.fps = float(fps)
        self.history_seconds = float(history_seconds)
        self.maxlen = max(8, int(round(self.fps * self.history_seconds)))
        self.norm_w = int(norm_w)
        self.norm_h = int(norm_h)
        self.slow_hz = float(slow_hz)
        self.fast_hz = float(fast_hz)
        self.debug = bool(debug)
        self._history: dict[str, list[BrightnessSample]] = defaultdict(list)

    def update(self, crop: np.ndarray, track_id: int | str = 0) -> BlinkRuntimeResult:
        """Append one crop sample and classify current blink state for that track."""
        global NORM_W, NORM_H, SLOW_HZ, FAST_HZ
        NORM_W = self.norm_w
        NORM_H = self.norm_h
        SLOW_HZ = self.slow_hz
        FAST_HZ = self.fast_hz

        tid = str(track_id)
        sample = extract_brightness_from_crop(crop)
        hist = self._history[tid]
        hist.append(sample)
        if len(hist) > self.maxlen:
            del hist[: len(hist) - self.maxlen]

        valid = np.array([b.valid for b in hist], dtype=bool)
        values_raw = np.array([b.brightness for b in hist], dtype=np.float64)
        active_raw = np.array([b.active_ratio for b in hist], dtype=np.float64)
        values = fill_invalid(values_raw, valid)
        active = fill_invalid(active_raw, valid)

        if len(values) < max(8, int(round(0.6 * self.fps))):
            return BlinkRuntimeResult(
                state="collecting_yellow_history",
                confidence=0.0,
                frequency_hz=0.0,
                reason="not enough temporal samples yet",
            )

        result = classify_blink(values, active, self.fps)
        return BlinkRuntimeResult(
            state=str(result.get("state", "uncertain")),
            confidence=float(result.get("confidence", 0.0)),
            frequency_hz=float(result.get("frequency_hz", 0.0)),
            reason=str(result.get("reason", "")),
        )
