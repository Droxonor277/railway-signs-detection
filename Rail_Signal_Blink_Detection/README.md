# LeniaDynamics rel026 — Yellow blinking signal analysis

Git-ready English package for the yellow railway signal blinking detector and frequency analysis.

## What is included

- `run_pipeline.py` — main integrated entry point; runs the detection pipeline and calls the yellow blink detector from the classifier hook.
- `yellow_crop_blink_detector_v2.py` — diagnostic and runtime yellow-blink analysis module with `YellowCropBlinkDetector.update(crop, track_id)`.
- `original/` — exact original uploaded files preserved unchanged for traceability.
- `detection_pipeline.py`, `pipeline_config.py`, `detection_tracking.py`, `video_reader.py` — detection pipeline dependencies.
- `outputs/` — analytical outputs from blinking and steady-yellow experiments.
- `reports/` — summary CSV/JSON, figures and English report drafts.
- `docs/integration_notes.md` — notes about the integration.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Model weights

YOLO weights are not included because they are large and should not normally be committed to Git.
Place the model here:

```text
models/yolo11s_n6.pt
```

## Run the integrated detection pipeline

Edit these variables in `run_pipeline.py`:

```python
VIDEO = "path/to/video.mp4"
MODEL = "models/yolo11s_n6.pt"
OUTPUT_DIR = "results/pipeline"
```

Then run:

```bash
python run_pipeline.py
```

The classifier hook processes only detections with `det.class_name == "signal"`. For every signal crop it calls:

```python
result = blink_detector.update(crop, track_id=track_id)
```

and returns the current blink state and confidence to the pipeline.

## Run standalone diagnostic analysis

The standalone script can also run independently. Edit the CONFIG section in `yellow_crop_blink_detector_v2.py` and then run:

```bash
python yellow_crop_blink_detector_v2.py
```

It can create:

- diagnostic crop panels,
- CSV files with brightness and mask values over time,
- ON/OFF event CSV files,
- brightness + active-ratio + detrended + FFT plots,
- text reports explaining the decision.

## Main outputs for reporting

Useful outputs are in:

```text
outputs/analytical/
outputs/examples/
reports/summary_table.csv
reports/report_draft.md
reports/report_draft.tex
reports/report_full_en.tex
```
