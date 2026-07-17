"""
run detection pipeline on a single video and save cropped bounding box images
for use as classifier training data.

crops are saved to <output_dir>/ID_<track_id>_<class_name>_<video_stem>_<idx:06d>.jpg

usage:
    python run_pipeline_create_dataset.py <video> <output_dir>
"""

import os
import sys

import cv2
import numpy as np

from pipeline import DetectionPipeline, Detection, PipelineConfig

if len(sys.argv) < 3:
    raise SystemExit(
        "usage: python run_pipeline_create_dataset.py <video> <output_dir>"
    )

VIDEO = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

# --- config

MODEL = "models/yolo11s_n6.pt"
# target fps delivered by the video reader (lower = faster, fewer crops)
READER_FPS = 10
# fps at which yolo inference runs in phase 1 (lower = faster)
DETECTION_FPS = 5
DEVICE = "cuda:0"   # "cuda:0" or "cpu"
# crops from these detection classes will be saved to OUTPUT_DIR/<class_name>/
DETECTION_CLASSES = ["signal"]
# also write a detection csv alongside the crops
SAVE_CSV = False


# ------

video_stem = os.path.splitext(os.path.basename(VIDEO))[0]
_crop_counter = [0]  # mutable list used as a counter inside the closure

# pre-create output folder
os.makedirs(OUTPUT_DIR, exist_ok=True)

# use the classification function to save the crops to disk during the pipeline run
def classify(crop: np.ndarray, det: Detection) -> tuple[str, float]:
    """save the detected crop to disk and return the class label."""
    filename = f"ID_{det.track_id}_{det.class_name}_{video_stem}_{_crop_counter[0]:06d}.jpg"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), crop)
    _crop_counter[0] += 1
    return det.class_name, det.confidence


config = PipelineConfig(
    reader_fps=READER_FPS,
    detection_fps=DETECTION_FPS,
    detection_confidence=0.5,
    imgsz=1920,
    device=DEVICE,
    classify_classes=DETECTION_CLASSES,
    detection_side = "both",
)

# the pipeline only calls classify (and thus saves crops) when csv_writer is
# active - so we always enable csv writing internally and remove the file
# afterward if SAVE_CSV is False
pipeline = DetectionPipeline(MODEL, config, classifier_fn=classify)
pipeline.process_video(VIDEO, output_dir=OUTPUT_DIR, save_csv=True)

if not SAVE_CSV:
    csv_path = os.path.join(OUTPUT_DIR, f"{video_stem}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
