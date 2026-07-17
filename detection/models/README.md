Trained models based on the YOLOv11s architecture:

## first iteration:

not full dataset ~ 1GB (missing video7 and video3 segments)

- `yolo11s_n3`: training imagesize: `1280x720`, separate classes for distant and main signals

## second iteration:

dataset with video7 and video3 segments ~ 1.5GB

- `yolo11s_n4`: training imagesize: `1440x810` - **faster**
- `yolo11s_n5`: training imagesize: `1920x1080`

## third iteration:

full dataset with partially annotated light signals `dataset` ~ 1.9GB. Signals annotated from greater distance to improve detection of small objects.

- `yolo11s_n6`: training imagesize: `1920x1080` - **latest**