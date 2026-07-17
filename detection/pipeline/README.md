## How the pipeline works

#### Phase 1 - segment discovery (detection_fps)

The `VideoReader` thread decodes the video, applies **corner cover** (paints over the in-cab signal display overlay in the upper-right corner with the average color of the upper-left reference region, matching the preprocessing used during dataset preparation), and pushes `FramePacket` objects into a bounded queue at `reader_fps` (default: source fps or 30 if real fps > 30). The detector thread consumes every frame from this queue but runs YOLO only every `detect_step`-th frame, achieving `detection_fps` (default: 5 fps).

Every frame is also stored in a **ring buffer** (bounded deque of `ring_buffer_size` frames) for use in phase 2.

Per detected class, a `_SegmentState` is opened tracking:
- `start_frame_no` / `end_frame_no` — extent of the segment in source frame indices
- `last_detect_reader_idx` — used to measure the gap since the last detection

A segment closes when a class is absent from phase-1 detections for more than `max_detection_gap_s` seconds or the buffer (`ring_buffer_size`) reaches its limit (`ring_buffer_size`). All open segments are also flushed at end of video.

#### Phase 2 - full-fps re-pass (reader_fps)

When a segment closes, all frames in the ring buffer with `frame_no` in `[start - offset, end + offset]` are collected. YOLO is re-run on each of these frames (only detections matching the segment class are kept).

**Significance check:** if the number of frames with at least one detection in the window is `>= min_detection_frames`, the segment is significant. Additionally, if `min_confident_frames > 0`, at least that many frames must have a detection with confidence `>= min_segment_confidence` - both conditions must pass (AND). False positives are discarded.

For significant segments: the classifier is called on each bbox crop, and results are written to CSV.

#### Parameters (PipelineConfig)

| parameter | default | description |
|-----------|---------|-------------|
| `reader_fps` | `"full"` | fps delivered by VideoReader; `"full"` = source fps |
| `detection_fps` | `5` | phase-1 YOLO inference fps |
| `detection_confidence` | `0.5` | YOLO confidence threshold |
| `imgsz` | `1920` | YOLO inference image size |
| `device` | `"cuda:0"` | inference device |
| `corner_cover` | `True` | paint over cab display overlay in top-right corner |
| `reader_buffer_size` | `20` | queue size between reader and detector threads |
| `ring_buffer_size` | `300` | max reader-fps frames in look-back ring buffer - **defines the temporal resolution of the pipeline** |
| `segment_offset_s` | `0.2` | seconds of context before/after a segment for phase-2 window, should be selected based on `detection_fps` (to cover at least the time between frames of phase-1 detections) |
| `min_detection_frames` | `7` | min reader-fps frames with detection to pass significance check* |
| `min_segment_confidence` | `0.75` | additional confidence threshold for segment significance (0.0 = disabled) |
| `min_confident_frames` | `2` | min frames with conf >= `min_segment_confidence` to pass (0 = disabled)* |
| `max_detection_gap_s` | `0.1` | seconds without phase-1 detection before closing a segment |
| `classify_classes` | `["signal"]` | classes sent to the classifier function |
| `box_relative` | `True` | if True, bbox coords in output CSV are relative (0-1) instead of pixels|
| `debug` | `False` | if True, debug information (of phase 1) is printed to the console. |
| `tracking_enabled` | `True` | enable centre-distance tracking in of objects (see additional parameters in [Tracking](#tracking) below) |

\* **Auto-scaling:** `min_detection_frames`, `min_confident_frames`, `track_max_lost`, and `track_max_dist_limits` are automatically scaled when `reader_fps` is lower than the source fps. The defaults assume full source fps (~30). When e.g. `reader_fps=15` on a 30fps video (scale 0.5), frame counts are halved and distance limits are doubled so that the pipeline behaves consistently regardless of reader fps.

#### Tracking

After the full-fps re-pass in phase 2, a centre-distance tracker assigns consistent `detection_ID` values to distinguish multiple objects of the same class in the same segment (e.g. two signals visible in the same window).

Tracking runs online across the phase-2 window frames in order. Each detection is matched to the nearest existing track by normalized centre-to-centre distance. The distance threshold scales dynamically with the detection's relative bounding box area: small far-away objects use a tight threshold to avoid cross-matching nearby objects; large close objects use a loose threshold to tolerate faster apparent motion between frames.

After tracking, IDs are reassigned so that ID 1 is the rightmost object, ID 2 the next, etc. (based on first-appearance x-centre).

> **Warning:** when objects are still far away, their bounding boxes are very small and close together on screen. The distance threshold at small areas is tight, but if two objects appear within that radius of each other, they may receive swapped or inconsistent IDs (if in one frame one of them is not detected). IDs become stable and reliable once objects are large enough to be clearly separated on screen.

| parameter | default | description |
|-----------|---------|-------------|
| `tracking_enabled` | `True` | enable centre-distance tracking in phase 2 |
| `track_max_dist_limits` | `(0.02, 0.15)` | (min, max) normalised centre distance threshold; linearly scaled by bbox area. Auto-scaled with reader_fps. |
| `track_max_lost` | `5` | frames a track can be absent before being discarded. Auto-scaled with reader_fps. The distance does scale with the lost frames - does not apply for small bounding boxes (see [`detection_tracking.py`](detection_tracking.py)). |
| `segment_postprocess` | `False` | enable per-segment track postprocessing (side filter, area filter, center-proximity); see [Track postprocessing](#track-postprocessing) |
| `min_track_max_area` | `0.0` | drop tracks whose bounding box never reached this relative area (0-1) in the segment; `0.0` = disabled |
| `detection_side` | `"both"` | keep only tracks on this side of `center_line_x`: `"left"`, `"right"`, or `"both"` |
| `center_line_x` | `0.0` | reference vertical line in [-1, 1] used for side filter and center-proximity selection; 0 = image center, -1 = left edge, 1 = right edge |
| `postprocess_last_frames` | `5` | number of last detected frames per track used for the side filter and center-proximity computation |


#### Track postprocessing

After tracking, an optional postprocessing step (enabled by `segment_postprocess=True`) filters the candidate tracks per segment before significance checking, classification, and CSV writing. Only tracks that survive all three filters are written to the output. The filters are applied in order:

1. **Side filter** (`detection_side`, `center_line_x`, `postprocess_last_frames`): drops tracks whose average center x (over their last `postprocess_last_frames` detected frames, mapped to [-1, 1]) is on the wrong side of `center_line_x`. For `"both"` (default) this step is skipped. Intended to discard signals on adjacent tracks when the relevant track is known.

2. **Area filter** (`min_track_max_area`): drops tracks whose bounding box never reached the specified relative area across the whole segment. Since the train is approaching signals, the correct detection should grow over time - tracks that stay small throughout are likely distant or irrelevant objects. Default `0.005`.

3. **Center-proximity selection**: among tracks that overlap temporally (their detected frame ranges intersect), keeps only the one whose last-N average center x is closest to `center_line_x`. This handles the case where multiple same-class objects are visible at the same time - the one closest to the track center line is preferred as the relevant signal.

> **Warning:** in segments with many simultaneously visible signals (e.g., stations), center-proximity selection eliminates all but one overlapping track per group. The surviving track may have fewer frames than the discarded ones, causing it to fail the significance check while reliable tracks were dropped. Disable postprocessing (`segment_postprocess=False`, the default) for such scenarios.

#### Ring buffer, long segments and auto-split

The ring buffer holds at most `ring_buffer_size` reader-fps frames. For a 30 fps video with `ring_buffer_size=300`, that is 10 seconds. Segments longer than this (for example in station) would have their earliest frames evicted before phase 2 can process them. To prevent frame loss, phase 1 automatically **splits** long segments when an active segment approaches the ring buffer capacity (`ring_buffer_size - 2 * offset_frames` reader frames). A split triggers phase 2 on the current buffer contents, then the segment continues from the next frame. The window bounds for each chunk are set so that consecutive splits do not overlap:

- **First chunk**: `[start - offset, split_frame]`
- **Continuation**: `[previous_split + 1, next_split]`
- **Final chunk**: `[last_split + 1, end + offset]`

The tracker instance persists across all splits of the same class segment, so track IDs remain consistent. Right-to-left ID reassignment is applied only for short segments that fit in a single buffer; split segments use raw tracker IDs which are naturally stable.
