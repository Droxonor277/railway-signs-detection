"""
batch_run_yellow_blink_analysis.py

Batch runner for yellow_crop_blink_detector_v2.py over all videos in:
  - C:/code/tym1/dataset/blinking_yellow_signal/signal_X.mp4
  - C:/code/tym1/dataset/one_yellow_signal/signal_X.mp4

For each signal, the script creates a separate output folder:
  <dataset_folder>/_yellow_blink_outputs/signal_X/

It stores:
  - CSV from the detection pipeline
  - events CSV
  - diagnostic signals CSV
  - diagnostic TXT report
  - PNG plot
  - diagnostic crop/panel images in debug/

Python 3.12 note:
  The module must be inserted into sys.modules before spec.loader.exec_module(module),
  otherwise @dataclass may fail during dynamic import.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


# =============================================================================
# PATHS - adjust this section as needed
# =============================================================================

DETECTOR_SCRIPT = Path(r"C:/code/tym1/LeniaDynamics_rel022/yellow_crop_blink_detector_v2.py")

MODEL_PATH = Path(r"C:/code/tym1/LeniaDynamics_rel022/models/yolo11s_n6.pt")

DATASET_DIRS = {
    "blinking_yellow_signal": Path(r"C:/code/tym1/dataset/blinking_yellow_signal"),
    "one_yellow_signal": Path(r"C:/code/tym1/dataset/one_yellow_signal"),
}

# True = run YOLO pipeline for every video and create a new CSV.
# False = use an already existing CSV if available.
RUN_PIPELINE_BEFORE_ANALYSIS = False

# True = always overwrite the detection-pipeline CSV.
# False = skip the pipeline if the CSV already exists.
FORCE_RECREATE_CSV = False

# None = automatically select the longest detection_ID track in the CSV.
# Recommended for batch processing, because a fixed SELECTED_ID may not match every signal.
SELECTED_ID = None

# If CUDA/GPU is not available, change this to "cpu".
DEVICE = "cuda:0"


# =============================================================================
# Helper functions
# =============================================================================

def natural_signal_key(path: Path) -> tuple[int, str]:
    """Sort signal_1, signal_2, ..., signal_10 instead of lexicographic order."""
    stem = path.stem
    try:
        return int(stem.split("_")[-1]), stem
    except ValueError:
        return 10**9, stem


def load_detector_module(script_path: Path):
    """Load the detector as a module without executing its main block.

    This allows the batch runner to override CONFIG variables and then call main() manually.
    For Python 3.12, the module must be inserted into sys.modules before exec_module().
    """
    if not script_path.exists():
        raise FileNotFoundError(f"Detector does not exist: {script_path}")

    module_name = f"yellow_crop_blink_detector_v2_batch_{id(script_path)}"

    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load detector as a module: {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module


def configure_detector(detector, video_path: Path, output_dir: Path) -> None:
    """Override CONFIG variables in yellow_crop_blink_detector_v2.py for one video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    signal_name = video_path.stem
    debug_dir = output_dir / "debug"

    detector.VIDEO_PATH = str(video_path)
    detector.MODEL_PATH = str(MODEL_PATH)
    detector.OUTPUT_DIR = str(output_dir)
    detector.CSV_PATH = str(output_dir / f"{signal_name}.csv")

    detector.RUN_PIPELINE_BEFORE_ANALYSIS = RUN_PIPELINE_BEFORE_ANALYSIS
    detector.FORCE_RECREATE_CSV = FORCE_RECREATE_CSV
    detector.DEVICE = DEVICE
    detector.SELECTED_ID = SELECTED_ID

    detector.OUTPUT_PLOT = str(output_dir / f"{signal_name}_yellow_analysis.png")
    detector.EVENTS_CSV = str(output_dir / f"{signal_name}_events.csv")
    detector.SIGNALS_CSV = str(output_dir / f"{signal_name}_diagnostic_signals.csv")
    detector.REPORT_TXT = str(output_dir / f"{signal_name}_diagnostic_report.txt")

    detector.SAVE_CROPS = True
    detector.SAVE_DIAGNOSTICS = True
    detector.DEBUG_DIR = str(debug_dir)
    detector.SHOW_PLOT = False


def process_one_video(video_path: Path, output_dir: Path) -> dict:
    """Process one video and return a compact status record for the final report."""
    detector = load_detector_module(DETECTOR_SCRIPT)
    configure_detector(detector, video_path, output_dir)

    detector.main()

    return {
        "video": str(video_path),
        "output_dir": str(output_dir),
        "status": "OK",
        "error": "",
    }


def write_batch_summary(summary_path: Path, rows: list[dict]) -> None:
    """Save a processing summary to CSV."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        f.write("status;video;output_dir;error\n")
        for row in rows:
            f.write(
                f"{row['status']};"
                f"{row['video']};"
                f"{row['output_dir']};"
                f"{row['error'].replace(chr(10), ' | ')}\n"
            )


def main() -> None:
    if not DETECTOR_SCRIPT.exists():
        raise FileNotFoundError(f"Detector does not exist: {DETECTOR_SCRIPT}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model does not exist: {MODEL_PATH}")

    all_results: list[dict] = []

    for dataset_name, dataset_dir in DATASET_DIRS.items():
        if not dataset_dir.exists():
            print(f"[SKIP] Dataset folder does not exist: {dataset_dir}")
            continue

        output_root = dataset_dir / "_yellow_blink_outputs"
        output_root.mkdir(parents=True, exist_ok=True)

        videos = sorted(dataset_dir.glob("signal_*.mp4"), key=natural_signal_key)

        print("")
        print("=" * 80)
        print(f"Dataset: {dataset_name}")
        print(f"Folder:  {dataset_dir}")
        print(f"Number of videos: {len(videos)}")
        print(f"Outputs: {output_root}")
        print("=" * 80)

        if not videos:
            print(f"[WARN] No signal_*.mp4 videos found in folder: {dataset_dir}")
            continue

        dataset_results: list[dict] = []

        for video_path in videos:
            signal_name = video_path.stem
            output_dir = output_root / signal_name

            print("")
            print("-" * 80)
            print(f"[RUN] {dataset_name}/{video_path.name}")
            print(f"[OUT] {output_dir}")
            print("-" * 80)

            try:
                result = process_one_video(video_path, output_dir)
                all_results.append(result)
                dataset_results.append(result)
                print(f"[OK] Done: {video_path.name}")

            except Exception as exc:
                error_text = "".join(traceback.format_exception_only(type(exc), exc)).strip()
                result = {
                    "video": str(video_path),
                    "output_dir": str(output_dir),
                    "status": "ERROR",
                    "error": error_text,
                }
                all_results.append(result)
                dataset_results.append(result)

                print(f"[ERROR] {video_path.name}: {error_text}")
                print("Error detail:")
                traceback.print_exc()

        write_batch_summary(output_root / "_batch_summary.csv", dataset_results)

    common_summary = DETECTOR_SCRIPT.parent / "yellow_blink_batch_summary.csv"
    write_batch_summary(common_summary, all_results)

    ok_count = sum(1 for r in all_results if r["status"] == "OK")
    err_count = sum(1 for r in all_results if r["status"] != "OK")

    print("")
    print("=" * 80)
    print("BATCH PROCESSING FINISHED")
    print(f"OK: {ok_count}")
    print(f"ERROR: {err_count}")
    print(f"Summary: {common_summary}")
    print("=" * 80)


if __name__ == "__main__":
    main()
