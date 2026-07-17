"""
run detection pipeline on a single video file and produce an annotated video.

runs the detection pipeline (csv output), then loads the video again together
with the generated csv and draws bounding boxes with labels onto every frame,
saving the result as a full annotated video.
"""

import csv
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

from pipeline import DetectionPipeline, Detection, PipelineConfig

# VIDEO = sys.argv[1]
# MODEL = sys.argv[2]

# set video and model paths
VIDEO = 'test_videos/double_yellow2.mp4'
# VIDEO = 'test_videos/one_red.mp4'
MODEL = 'models/yolo11s_n6.pt'
OUTPUT_DIR = "results/kaggle/n6/pipeline"

# --- annotation style ---
BOX_THICKNESS: int = 3
FONT_SCALE: float = 1

# bgr colors per detection class - fallback to white for unknown classes
CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "signal":       (0,   0,   255),    # red
    "dwarf_signal": (0,   165, 255),    # orange
    "distant_sign": (255, 0,   255),    # magenta
    "sign_stripes": (255, 255, 0),      # cyan
    "sign_triangle":(255, 255, 0),      # cyan
}
DEFAULT_COLOR: tuple[int, int, int] = (255, 255, 255)  # white


# --- classifier integration ---
def classify(crop: np.ndarray, det: Detection) -> tuple[str, float]:
    return det.class_name, det.confidence


config = PipelineConfig(
    reader_fps="full",
    detection_fps=5,
    detection_confidence=0.5,
    imgsz=1920, # 960,
    device="cuda:0",
    detection_side = "both",
)


# --------------------------------------------------
# annotated video writer
# --------------------------------------------------

def load_csv_detections(csv_path: str) -> dict[int, list[dict]]:
    """load pipeline csv and return a dict mapping frame_no to list of detection dicts."""
    detections: dict[int, list[dict]] = defaultdict(list)
    with open(csv_path, "r") as f:
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
    """draw bounding boxes and labels on frame in-place from csv row dicts."""
    h, w = frame.shape[:2]
    for row in rows:
        if box_relative:
            x1 = int(float(row["x1"]) * w)
            y1 = int(float(row["y1"]) * h)
            x2 = int(float(row["x2"]) * w)
            y2 = int(float(row["y2"]) * h)
        else:
            x1, y1 = int(row["x1"]), int(row["y1"])
            x2, y2 = int(row["x2"]), int(row["y2"])

        label_det = row["label_detection"]
        conf_det = row["confidence_detection"]
        label_cls = row.get("label_classification", "")
        conf_cls = row.get("confidence_classification", "")
        det_id = row.get("detection_ID", "")

        color = CLASS_COLORS.get(label_det, DEFAULT_COLOR)

        # use classification label if available, otherwise detection label
        if label_cls:
            label = f"{label_cls} ({conf_cls})"
        else:
            label = f"{label_det} ({conf_det})"

        if det_id:
            label = f"#{det_id} {label}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            frame, label, (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
        )


def write_annotated_video(
    video_path: str,
    csv_path: str,
    output_path: str,
) -> None:
    """load video and csv, draw bounding boxes on each frame, save as avi."""
    detections = load_csv_detections(csv_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # use reader_fps for output if configured, source fps otherwise
    if config.reader_fps == "full":
        out_fps = fps
        frame_step = 1
    else:
        frame_step = max(1, round(fps / config.reader_fps))
        out_fps = fps / frame_step

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vw = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_no % frame_step == 0:
                if frame_no in detections:
                    draw_annotations(frame, detections[frame_no], box_relative=config.box_relative)
                vw.write(frame)

            frame_no += 1
    finally:
        vw.release()
        cap.release()

    print(f"annotated video saved: {output_path}")


# --------------------------------------------------
# main
# --------------------------------------------------
if __name__ == "__main__":
    stem = os.path.splitext(os.path.basename(VIDEO))[0]
    csv_path = os.path.join(OUTPUT_DIR, f"{stem}.csv")
    video_out = os.path.join(OUTPUT_DIR, f"{stem}_annotated.avi")

    # run detection pipeline -> csv
    pipeline = DetectionPipeline(MODEL, config, classifier_fn=classify)
    pipeline.process_video(VIDEO, output_dir=OUTPUT_DIR, save_csv=True)
    print("processing done")

    # annotate video from csv
    print("writing annotated video...")
    write_annotated_video(VIDEO, csv_path, video_out)
    print("annotated video saved")
