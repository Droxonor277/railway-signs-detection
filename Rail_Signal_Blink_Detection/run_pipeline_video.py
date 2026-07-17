"""
Run detection pipeline on a single video file, write pipeline CSV, optionally
write an annotated video, and optionally run yellow-blinking analysis on a
selected detected signal bounding box.

Typical usage:
    python run_pipeline_video.py --video test_videos/double_yellow2.mp4 --model models/yolo11s_n6.pt --blink --id 1

If --blink is used, this script analyzes only the selected detection_ID bbox
from the pipeline CSV, not the whole frame.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def _load_pipeline_api():
    """Import the pipeline API both when this folder is named `pipeline` and
    when the files are executed directly from a flat extracted ZIP folder.
    """
    try:
        from pipeline import DetectionPipeline, Detection, PipelineConfig  # type: ignore
        return DetectionPipeline, Detection, PipelineConfig
    except ModuleNotFoundError:
        current_dir = Path(__file__).resolve().parent
        parent_dir = current_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        package = importlib.import_module(current_dir.name)
        return package.DetectionPipeline, package.Detection, package.PipelineConfig


try:
    from test_yellow_pipeline_bbox import analyze_pipeline_bbox
except ModuleNotFoundError:
    analyze_pipeline_bbox = None


# --- annotation style ---
BOX_THICKNESS: int = 3
FONT_SCALE: float = 1.0

# BGR colors per detection class; fallback to white for unknown classes.
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "signal": (0, 0, 255),
    "dwarf_signal": (0, 165, 255),
    "distant_sign": (255, 0, 255),
    "sign_stripes": (255, 255, 0),
    "sign_triangle": (255, 255, 0),
}
DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)


def classify(crop: np.ndarray, det) -> tuple[str, float]:
    """Placeholder classifier hook.

    The detection pipeline calls this function for crops whose detection class
    is configured in PipelineConfig.classify_classes. For now it returns the
    detection label and confidence. The yellow blinking test is run later from
    the CSV bbox track.
    """
    return det.class_name, det.confidence


def load_csv_detections(csv_path: str | Path) -> dict[int, list[dict]]:
    """Load pipeline CSV and return frame_no -> list of detection row dicts."""
    detections: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            frame_no = int(row["frame_no"])
            detections[frame_no].append(row)
    return dict(detections)


def draw_annotations(
    frame: np.ndarray,
    rows: list[dict],
    box_relative: bool = False,
    thickness: int = BOX_THICKNESS,
    font_scale: float = FONT_SCALE,
) -> None:
    """Draw bounding boxes and labels on frame in-place from CSV row dicts."""
    h, w = frame.shape[:2]
    for row in rows:
        if box_relative:
            x1 = int(float(row["x1"]) * w)
            y1 = int(float(row["y1"]) * h)
            x2 = int(float(row["x2"]) * w)
            y2 = int(float(row["y2"]) * h)
        else:
            x1, y1 = int(float(row["x1"])), int(float(row["y1"]))
            x2, y2 = int(float(row["x2"])), int(float(row["y2"]))

        label_det = row.get("label_detection", "")
        conf_det = row.get("confidence_detection", "")
        label_cls = row.get("label_classification", "")
        conf_cls = row.get("confidence_classification", "")
        det_id = row.get("detection_ID", "")

        color = CLASS_COLORS.get(label_det, DEFAULT_COLOR)
        label = f"{label_cls} ({conf_cls})" if label_cls else f"{label_det} ({conf_det})"
        if det_id:
            label = f"#{det_id} {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
        )


def write_annotated_video(
    video_path: str | Path,
    csv_path: str | Path,
    output_path: str | Path,
    box_relative: bool,
    reader_fps: int | str = "full",
) -> None:
    """Load video and CSV, draw bounding boxes on each frame, save as AVI."""
    detections = load_csv_detections(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if reader_fps == "full":
        out_fps = fps
        frame_step = 1
    else:
        frame_step = max(1, round(fps / float(reader_fps)))
        out_fps = fps / frame_step

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % frame_step == 0:
                if frame_no in detections:
                    draw_annotations(frame, detections[frame_no], box_relative=box_relative)
                vw.write(frame)

            frame_no += 1
    finally:
        vw.release()
        cap.release()

    print(f"Annotated video saved: {output_path}")


def build_config(args: argparse.Namespace, PipelineConfig):
    return PipelineConfig(
        reader_fps=args.reader_fps,
        detection_fps=args.detection_fps,
        detection_confidence=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
        detection_side=args.detection_side,
        box_relative=not args.pixel_boxes,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run detection pipeline and optionally analyze yellow blinking in selected bbox."
    )
    parser.add_argument("--video", default="test_videos/double_yellow2.mp4", help="Input video path.")
    parser.add_argument("--model", default="models/yolo11s_n6.pt", help="YOLO model path.")
    parser.add_argument("--output-dir", default="results/kaggle/n6/pipeline", help="Output directory for CSV/results.")
    parser.add_argument("--device", default="cuda:0", help="Inference device, e.g. cuda:0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=1920, help="YOLO inference image size.")
    parser.add_argument("--reader-fps", default="full", help='Reader FPS or "full".')
    parser.add_argument("--detection-fps", default=5, help='Detection FPS or "full".')
    parser.add_argument("--confidence", type=float, default=0.5, help="YOLO confidence threshold.")
    parser.add_argument("--detection-side", default="both", choices=["left", "right", "both"], help="Track postprocess side filter.")
    parser.add_argument("--pixel-boxes", action="store_true", help="Write/read CSV bbox as pixel coordinates instead of relative coords.")

    parser.add_argument("--skip-pipeline", action="store_true", help="Do not run YOLO; use an existing CSV.")
    parser.add_argument("--no-annotated", action="store_true", help="Do not write annotated AVI video.")

    parser.add_argument("--blink", action="store_true", help="Analyze yellow blinking in selected pipeline bbox after CSV is available.")
    parser.add_argument("--id", default=None, help="Selected detection_ID for blink analysis. If omitted, longest track is used.")
    parser.add_argument("--label", default="signal", help="CSV label_detection to analyze, usually signal.")
    parser.add_argument("--min-hz", type=float, default=0.1, help="Minimum blink frequency.")
    parser.add_argument("--max-hz", type=float, default=5.0, help="Maximum blink frequency.")
    parser.add_argument("--hold", type=int, default=3, help="Hold last bbox for this many frames during short detection dropout.")
    parser.add_argument("--padding", type=float, default=0.0, help="Relative bbox expansion for blink crop. Default 0.0 = use the exact small YOLO bbox from CSV.")
    parser.add_argument("--min-crop-px", type=int, default=0, help="Minimum square crop size in pixels. Default 0 = do not enlarge; use the exact YOLO bbox.")
    parser.add_argument("--debug-bbox", action="store_true", help="Show selected bbox and analyzed crop preview during blink analysis.")
    parser.add_argument("--save-crops", action="store_true", help="Save debug images showing the exact crop used for blink analysis.")
    parser.add_argument("--crop-dir", default=None, help="Optional directory for saved crop debug images. If omitted, a folder is created next to the CSV.")
    parser.add_argument("--crop-every", type=int, default=10, help="Save every N-th analyzed crop when --save-crops is used.")
    parser.add_argument("--crop-max", type=int, default=30, help="Maximum number of saved crop debug images.")
    parser.add_argument("--lamp-roi", action=argparse.BooleanOptionalAction, default=True, help="Analyze a small lamp ROI found inside the YOLO bbox. Default: enabled. Use --no-lamp-roi to analyze the full YOLO bbox.")
    parser.add_argument("--lamp-roi-size", type=int, default=12, help="Square lamp ROI size in pixels inside the YOLO bbox.")
    parser.add_argument("--lamp-search", choices=["black-bright", "yellow"], default="black-bright", help="How to find lamp ROI. black-bright = find black signal body first, then brightest pixels inside it. yellow = original yellow/orange score.")
    parser.add_argument("--black-threshold", type=int, default=70, help="HSV V-channel threshold for the black signal body mask used by --lamp-search black-bright.")
    parser.add_argument("--plot", default=None, help="Optional PNG path for yellow blinking plot. If omitted, it is saved next to the CSV.")
    parser.add_argument("--events-csv", default=None, help="Optional CSV path for detected ON/OFF blink event times.")
    parser.add_argument("--min-state-sec", type=float, default=0.12, help="Ignore ON/OFF states shorter than this duration in seconds.")
    parser.add_argument("--glare-suppression", action=argparse.BooleanOptionalAction, default=True, help="Compress overexposed/glare pixels before brightness extraction. Use --no-glare-suppression to disable.")
    parser.add_argument("--glare-threshold", type=int, default=240, help="HSV Value threshold for glare compression.")
    parser.add_argument("--yellow-ratio-threshold", type=float, default=0.003, help="Minimum yellow/white active ratio used as color-support confidence.")
    parser.add_argument("--no-show", action="store_true", help="Save the yellow blinking plot without opening a matplotlib window.")
    parser.add_argument("--color-analysis", action="store_true", help="Save pixel color analysis from the selected crop to CSV.")
    parser.add_argument("--color-csv", default=None, help="Optional CSV path for pixel color analysis. If omitted, it is saved next to the pipeline CSV.")
    parser.add_argument("--color-top-n", type=int, default=8, help="Number of most frequent quantized BGR colors to save per analyzed frame.")
    parser.add_argument("--color-bin-size", type=int, default=16, help="BGR quantization bin size for the top color table, e.g. 16.")
    parser.add_argument("--color-roi", choices=["body", "bbox"], default="body", help="Crop used for color analysis: black signal body or whole YOLO bbox.")
    return parser.parse_args()


def _normalize_fps_value(value: str | int | float):
    if isinstance(value, str) and value.lower() == "full":
        return "full"
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"FPS must be a number or 'full', got: {value!r}")
    return int(as_float) if as_float.is_integer() else as_float


def main() -> None:
    args = parse_args()
    args.reader_fps = _normalize_fps_value(args.reader_fps)
    args.detection_fps = _normalize_fps_value(args.detection_fps)

    video_path = Path(args.video)
    output_dir = Path(args.output_dir)
    stem = video_path.stem
    csv_path = output_dir / f"{stem}.csv"
    annotated_video_path = output_dir / f"{stem}_annotated.avi"

    config = None

    if not args.skip_pipeline:
        DetectionPipeline, _Detection, PipelineConfig = _load_pipeline_api()
        config = build_config(args, PipelineConfig)
        print("Running detection pipeline...")
        pipeline = DetectionPipeline(args.model, config, classifier_fn=classify)
        pipeline.process_video(str(video_path), output_dir=str(output_dir), save_csv=True)
        print(f"Pipeline CSV saved/expected at: {csv_path}")
    else:
        print(f"Skipping detection pipeline, using existing CSV: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV was not found: {csv_path}. Run without --skip-pipeline first or check --output-dir."
        )

    if not args.no_annotated:
        print("Writing annotated video...")
        write_annotated_video(
            video_path=video_path,
            csv_path=csv_path,
            output_path=annotated_video_path,
            box_relative=not args.pixel_boxes,
            reader_fps=args.reader_fps,
        )

    if args.blink or args.color_analysis:
        if analyze_pipeline_bbox is None:
            raise RuntimeError("test_yellow_pipeline_bbox.py was not found next to run_pipeline_video.py")
        print("Running bbox analysis on selected track...")
        analyze_pipeline_bbox(
            video_path=video_path,
            csv_path=csv_path,
            selected_id=args.id,
            label_filter=args.label,
            min_hz=args.min_hz,
            max_hz=args.max_hz,
            max_hold_frames=args.hold,
            padding=args.padding,
            min_crop_px=args.min_crop_px,
            show_debug_window=args.debug_bbox,
            save_crops=args.save_crops,
            crop_dir=args.crop_dir,
            crop_every=args.crop_every,
            crop_max=args.crop_max,
            use_lamp_roi=args.lamp_roi,
            lamp_roi_size=args.lamp_roi_size,
            lamp_search=args.lamp_search,
            black_threshold=args.black_threshold,
            plot_path=args.plot,
            events_csv_path=args.events_csv,
            min_state_sec=args.min_state_sec,
            glare_suppression=args.glare_suppression,
            glare_threshold=args.glare_threshold,
            yellow_ratio_threshold=args.yellow_ratio_threshold,
            show_plot=not args.no_show,
            color_analysis=args.color_analysis,
            color_csv_path=args.color_csv,
            color_top_n=args.color_top_n,
            color_bin_size=args.color_bin_size,
            color_roi=args.color_roi,
        )


if __name__ == "__main__":
    main()
