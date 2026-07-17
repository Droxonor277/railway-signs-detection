# Yellow signal blinking test in a selected bounding box

This document describes the bbox-based yellow blinking analysis workflow. The analysis is performed only inside a selected bounding box produced by the detection pipeline.

## What was changed

- `run_pipeline_video.py`
  - accepts command-line arguments,
  - can run the YOLO detection pipeline,
  - can write an annotated video,
  - can optionally run yellow blinking analysis directly with `--blink`.

- bbox-based analysis
  - loads the original video,
  - loads the pipeline CSV output,
  - selects a specific `detection_ID`,
  - extracts the corresponding bbox crop for each frame,
  - sends only this crop to the yellow blinking detector.

## Install dependencies

In an active Python environment, install at least:

```bash
pip install opencv-python numpy matplotlib scipy ultralytics rich
```

If NVIDIA GPU/CUDA is not available, run the script with:

```bash
--device cpu
```

## Simplest run: pipeline + bbox blinking test

From the project directory, run:

```bash
python run_pipeline_video.py \
  --video test_videos/double_yellow2.mp4 \
  --model models/yolo11s_n6.pt \
  --output-dir results/kaggle/n6/pipeline \
  --blink \
  --id 1
```

Outputs:

- detection CSV:

```text
results/kaggle/n6/pipeline/double_yellow2.csv
```

- annotated video:

```text
results/kaggle/n6/pipeline/double_yellow2_annotated.avi
```

- console output of the yellow-signal analysis,
- graph of yellow intensity and frequency spectrum.

## Create only CSV and annotated video

```bash
python run_pipeline_video.py \
  --video test_videos/double_yellow2.mp4 \
  --model models/yolo11s_n6.pt \
  --output-dir results/kaggle/n6/pipeline
```

Then select the correct `detection_ID` from the CSV file or from the annotated video.

## Run only the blinking test when CSV already exists

```bash
python run_pipeline_video.py \
  --video test_videos/double_yellow2.mp4 \
  --output-dir results/kaggle/n6/pipeline \
  --skip-pipeline \
  --no-annotated \
  --blink \
  --id 1
```

## Automatic track selection

If `--id` is omitted, the script selects the longest track available in the CSV:

```bash
python run_pipeline_video.py \
  --video test_videos/double_yellow2.mp4 \
  --output-dir results/kaggle/n6/pipeline \
  --skip-pipeline \
  --blink
```

## Useful options

- `--debug-bbox` or `--debug`: display the selected bbox during analysis.
- `--padding 0.10`: enlarge the crop by 10 percent on each side.
- `--hold 5`: keep the last bbox for 5 frames during a short detection dropout.
- `--device cpu`: run inference without CUDA.
- `--no-annotated`: do not create an annotated video.
- `--skip-pipeline`: do not run YOLO and use an existing CSV file.

## Connection principle

The original full-frame analysis is replaced by ROI analysis:

```python
crop = frame[y1:y2, x1:x2]
detector.process_frame(crop)
```

Coordinates `x1, y1, x2, y2` are read from the pipeline CSV for the selected `detection_ID`. If bbox coordinates are relative values in the range 0 to 1, the script automatically converts them to pixels.

## When `--id 1` is not present in CSV

The pipeline log may show `[signal #1]` and `[signal #2]`, while the final CSV may contain only selected tracks after post-processing. If the analysis reports that `detection_ID=1` was not found, run the command without `--id`. The script will automatically select the longest available `detection_ID` from the CSV and print which ID is analyzed.

## Saving the blinking plot

The yellow-light analysis automatically saves a PNG plot next to the CSV file. For example, for `signal_1.mp4` and automatically selected `detection_ID=2`, the output may be:

```text
results/kaggle/n6/pipeline/signal_1_yellow_blink_detection_ID_2.png
```

To choose a custom path, use `--plot`. To save the plot without displaying a window, add `--no-show`.

## Note for rel026

The package uses the exact small YOLO bounding box by default. Padding and minimum crop size are not applied unless explicitly requested. This avoids measuring pixels from surrounding poles, shadows, or background objects instead of the lamp itself.

## Displaying or saving analyzed crops

The analysis can save control images of exactly the crops that enter the blinking detector:

```bash
python run_pipeline_video.py \
  --video C:/code/tym1/dataset/blinking_yellow_signal/signal_1.mp4 \
  --model models/yolo11s_n6.pt \
  --output-dir results/kaggle/n6/pipeline \
  --blink \
  --id 2 \
  --save-crops \
  --crop-every 5 \
  --crop-max 40 \
  --no-show
```

The output folder contains individual crop-debug images and a contact sheet summarizing multiple crops.
