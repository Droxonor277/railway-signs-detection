"""
run detection pipeline on a single video file.
"""

import numpy as np

from pipeline import DetectionPipeline, Detection, PipelineConfig
from yellow_crop_blink_detector_v2 import YellowCropBlinkDetector


VIDEO = "../../videos/video6/video6-Zilina-Bratislava-4-00-30-00.mp4"
MODEL = "models/yolo11s_n6.pt"
OUTPUT_DIR = "results/kaggle/n6/pipeline"
SAVE_CSV = True


blink_detector = YellowCropBlinkDetector(
    fps=30.0,
    history_seconds=4.0,
    norm_w=64,
    norm_h=128,
    slow_hz=0.9,
    fast_hz=1.8,
    debug=False,
)


def classify(crop: np.ndarray, det: Detection) -> tuple[str, float]:
    """
    crop = small signal bbox crop from the detection pipeline
    det = detection metadata, including class_name, confidence and track_id
    """

    if det.class_name != "signal":
        return det.class_name, det.confidence

    track_id = det.track_id if det.track_id is not None else 0

    result = blink_detector.update(crop, track_id=track_id)

    return result.state, result.confidence


config = PipelineConfig(
    reader_fps="full",
    detection_fps=5,
    detection_confidence=0.5,
    imgsz=1920,
    device="cuda:0",
    classify_classes=["signal"],
    detection_side="both",
)

pipeline = DetectionPipeline(MODEL, config, classifier_fn=classify)
pipeline.process_video(VIDEO, output_dir=OUTPUT_DIR, save_csv=SAVE_CSV)