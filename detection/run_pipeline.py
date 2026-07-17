"""
run detection pipeline on a single video file.
"""

import numpy as np

from pipeline import DetectionPipeline, Detection, PipelineConfig

# VIDEO = sys.argv[1]
# MODEL = sys.argv[2]

# set video and model paths
VIDEO = '../../videos/video6/video6-Zilina-Bratislava-4-00-30-00.mp4'
MODEL = 'models/yolo11s_n6.pt'
OUTPUT_DIR = "results/kaggle/n6/pipeline"
SAVE_CSV = True


# --- classifier integration ---
# replace this function with the real classifier when available.
# crop: bgr numpy array of the detected bounding box region
# det:  Detection dataclass with class_name, confidence, bbox, class_id
# must return (label: str, confidence: float)
def classify(crop: np.ndarray, det: Detection) -> tuple[str, float]:
    return det.class_name, det.confidence


config = PipelineConfig(
    reader_fps="full",
    detection_fps=5,
    detection_confidence=0.5,
    imgsz=1920,
    device="cuda:0",
    detection_side = "both",
)

pipeline = DetectionPipeline(MODEL, config, classifier_fn=classify)
pipeline.process_video(VIDEO, output_dir=OUTPUT_DIR, save_csv=SAVE_CSV)
