"""
detection pipeline - runs yolo inference at detection_fps on video frames from
videoreader, groups detections into segments, then re-runs yolo on all frames
in the segment window (from the ring buffer) to produce full-fps results.

two-phase design:
    phase 1 (fast pass, detection_fps): yolo inference on every detect_step-th
    reader frame to find and track per-class segments.

    phase 2 (full pass, reader_fps): when a segment closes (gap exceeded or
    end of video), re-run yolo on every frame in the ring buffer within
    [start - offset, end + offset]. count frames with detection - if at least
    min_detection_frames, the segment is significant. classify crops and write csv.

integration:
    provide a classifier_fn at construction to plug in custom classification.
    signature: (crop: np.ndarray, det: Detection) -> (label: str, confidence: float)
    crop is a bgr numpy array of the bounding box region.
    if classifier_fn is None, no classification is run and those csv columns are empty.

note on ring buffer: the ring buffer holds up to ring_buffer_size reader-fps frames.
for segments longer than ring_buffer_size / reader_fps seconds, frames at the
start of the segment may have been evicted. significance counting still works (uses
whatever frames are in the buffer), but very long segments may lose early context.
"""

import csv
import os
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from ultralytics import YOLO

from .detection_tracking import SimpleTracker, reassign_ids_right_to_left
from .pipeline_config import Detection, FramePacket, PipelineConfig, _REFERENCE_FPS
from .video_reader import VideoReader

_console = Console()


# --------------------------------------------------
# classifier type alias
# --------------------------------------------------

ClassifierFn = Callable[[np.ndarray, Detection], tuple[str, float]]
ClassifiedRow = tuple[FramePacket, Detection, str, str]  # (packet, det, label_cls, conf_cls_str)


# --------------------------------------------------
# internal dataclass
# --------------------------------------------------

@dataclass
class _SegmentState:
    """tracks an active detection segment for one class (phase 1 only)."""
    class_name: str
    start_frame_no: int           # source video frame_no of first detection
    end_frame_no: int             # source video frame_no of latest detection
    last_detect_reader_idx: int   # reader-frame counter at last detection (for gap check)
    start_reader_idx: int         # reader_idx when this segment (or split) started
    window_start_override: int | None = None  # if set, next phase-2 window starts here (after a split)


# --------------------------------------------------
# detection pipeline
# --------------------------------------------------

class DetectionPipeline:
    """segment-based detection pipeline for railway sign detection.

    reads a video via videoreader at reader_fps, runs yolo at detection_fps
    to find segments, then re-runs yolo at reader_fps on the full segment
    window (from ring buffer) for significance filtering and csv output.

    postprocessing (yolo re-pass, classification, csv, video write) runs in
    the same thread as detection and stalls the pipeline until finished.
    the reader thread blocks on a full queue - no frames are lost.
    """

    def __init__(
        self,
        model_path: str,
        config: PipelineConfig,
        classifier_fn: ClassifierFn | None = None,
    ) -> None:
        self._config = config
        self._model = YOLO(model_path)
        self._classifier = classifier_fn

    # --------------------------------------------------
    # public interface
    # --------------------------------------------------

    def process_video(
        self,
        video_path: str,
        output_dir: str | None = None,
        save_csv: bool = True,
    ) -> None:
        """process a video file and save detection results.

        args:
            video_path: path to input video file
            output_dir: directory for output files; required when save_csv
            save_csv: write {stem}.csv with one row per detection per frame
        """
        reader_fps, detect_step, max_gap_frames, offset_frames, fps_scale, source_fps = (
            self._compute_timing(video_path)
        )

        # auto-scale frame-count parameters relative to reference_fps
        # (config defaults are calibrated for reference_fps; scale proportionally up or down)
        cfg = self._config
        self._eff_min_detection_frames = max(1, round(cfg.min_detection_frames * fps_scale))
        self._eff_min_confident_frames = (
            max(1, round(cfg.min_confident_frames * fps_scale))
            if cfg.min_confident_frames > 0 else 0
        )
        self._eff_track_max_lost = max(1, round(cfg.track_max_lost * fps_scale))
        # lower fps -> more movement between frames -> wider distance threshold
        self._eff_track_dist_limits = (
            cfg.track_max_dist_limits[0] / fps_scale,
            cfg.track_max_dist_limits[1] / fps_scale,
        )
        if fps_scale != 1.0:
            _console.print(
                f"fps scale {fps_scale:.2f} (reader {reader_fps:.1f} / reference {_REFERENCE_FPS}) - effective params: "
                f"min_det_frames={self._eff_min_detection_frames}, "
                f"min_conf_frames={self._eff_min_confident_frames}, "
                f"track_max_lost={self._eff_track_max_lost}, "
                f"track_dist=({self._eff_track_dist_limits[0]:.3f}, {self._eff_track_dist_limits[1]:.3f})",
                markup=False,
            )

        total_frames = self._video_frame_count(video_path)
        video_duration_s = total_frames / source_fps
        stem = os.path.splitext(os.path.basename(video_path))[0]

        if output_dir and save_csv:
            os.makedirs(output_dir, exist_ok=True)

        csv_file = None
        csv_writer = None
        if save_csv and output_dir:
            csv_path = os.path.join(output_dir, f"{stem}.csv")
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file, delimiter=";")
            csv_writer.writerow([
                "frame_no", "timestamp_ms",
                "x1", "y1", "x2", "y2",
                "label_detection", "detection_ID", "confidence_detection",
                "label_classification", "confidence_classification",
            ])

        reader_queue: queue.Queue[FramePacket | None] = queue.Queue(
            maxsize=self._config.reader_buffer_size
        )
        reader = VideoReader(video_path, self._config, reader_queue)
        reader.start()

        t_start = time.monotonic()
        try:
            self._run_detector(
                reader_queue=reader_queue,
                detect_step=detect_step,
                max_gap_frames=max_gap_frames,
                offset_frames=offset_frames,
                reader_fps=reader_fps,
                csv_writer=csv_writer,
                total_frames=total_frames,
            )
        finally:
            reader.stop()
            reader.join(timeout=5.0)
            if csv_file is not None:
                csv_file.close()
            elapsed = time.monotonic() - t_start
            mins, secs = divmod(int(elapsed), 60)
            vd_mins, vd_secs = divmod(int(video_duration_s), 60)
            _console.print(
                f"done in {mins:02d}:{secs:02d} ({elapsed:.1f}s) "
                f"| video {vd_mins:02d}:{vd_secs:02d} ({video_duration_s:.1f}s)",
                markup=False,
            )

    # --------------------------------------------------
    # timing helper
    # --------------------------------------------------

    def _compute_timing(self, video_path: str) -> tuple[float, int, int, int, float, float]:
        """derive frame-level timing constants from the video and config.

        returns:
            reader_fps: effective fps delivered by videoreader
            detect_step: every N-th reader frame runs phase-1 yolo
            max_gap_frames: reader frames without detection before segment closes
            offset_frames: reader frames of context before/after segment
            fps_scale: reader_fps / reference_fps (1.0 when reader_fps == reference_fps)
            source_fps: original video fps
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"cannot open video: {video_path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        cap.release()

        cfg = self._config

        if cfg.reader_fps == "full":
            frame_step_r = max(1, round(source_fps / _REFERENCE_FPS))
            reader_fps = source_fps / frame_step_r
            if source_fps > _REFERENCE_FPS:
                _console.print(
                    f"[dark_orange]warning[/dark_orange]: source fps ({source_fps:.1f}) exceeds reference fps ({_REFERENCE_FPS}); "
                    f"downsampling to {reader_fps:.1f} fps - set reader_fps explicitly to use higher fps",
                    markup=True,
                )
            elif source_fps < _REFERENCE_FPS:
                _console.print(
                    f"[dark_orange]warning[/dark_orange]: source fps ({source_fps:.1f}) is below reference fps ({_REFERENCE_FPS}); "
                    f"processing at {reader_fps:.1f} fps",
                    markup=True,
                )
        else:
            if cfg.reader_fps > _REFERENCE_FPS:
                _console.print(
                    f"[dark_orange]warning[/dark_orange]: reader_fps={cfg.reader_fps} exceeds reference fps ({_REFERENCE_FPS}); "
                    f"consider lowering it for faster processing",
                    markup=True,
                )
            if source_fps < cfg.reader_fps:
                _console.print(
                    f"[dark_orange]warning[/dark_orange]: requested reader_fps={cfg.reader_fps} exceeds source fps ({source_fps:.1f}); "
                    f"processing at source fps",
                    markup=True,
                )
            frame_step = max(1, round(source_fps / cfg.reader_fps))
            reader_fps = source_fps / frame_step
        
        if cfg.detection_fps == "full":
            detect_step = 1
        else:
            detect_step = max(1, round(reader_fps / cfg.detection_fps))

        # expressed as reader-frame counts
        max_gap_frames = max(detect_step, round(cfg.max_detection_gap_s * reader_fps))
        offset_frames = max(0, round(cfg.segment_offset_s * reader_fps))

        fps_scale = reader_fps / _REFERENCE_FPS
        return reader_fps, detect_step, max_gap_frames, offset_frames, fps_scale, source_fps

    def _video_frame_count(self, video_path: str) -> int:
        """return the total frame count of the video (used for progress bar)."""
        cap = cv2.VideoCapture(video_path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        return max(count, 1)

    # --------------------------------------------------
    # phase 1: detector loop
    # --------------------------------------------------

    def _run_detector(
        self,
        reader_queue: "queue.Queue[FramePacket | None]",
        detect_step: int,
        max_gap_frames: int,
        offset_frames: int,
        reader_fps: float,
        csv_writer: "csv.writer | None",
        total_frames: int,
    ) -> None:
        # ring buffer: all reader-fps frames for phase-2 re-pass look-back
        ring: deque[FramePacket] = deque(maxlen=self._config.ring_buffer_size)
        # active segments per class (phase 1 tracking)
        active: dict[str, _SegmentState] = {}
        # trackers persist across buffer splits for id continuity
        trackers: dict[str, SimpleTracker] = {}
        split_threshold = self._config.ring_buffer_size - 2 * offset_frames
        reader_idx = 0  # counts every reader frame received

        progress = Progress(
            TextColumn("[cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=_console,
        )
        task = progress.add_task("processing", total=total_frames)

        with progress:
            while True:
                packet = reader_queue.get()
                if packet is None:
                    progress.update(task, completed=total_frames)
                    # end of stream - flush all open segments
                    for seg in list(active.values()):
                        if self._config.debug:
                            _console.print(
                                f"  [debug] flushing [{seg.class_name}] "
                                f"phase-1 frames {seg.start_frame_no}-{seg.end_frame_no}",
                                markup=False,
                            )
                        self._close_segment(
                            seg, ring, offset_frames, csv_writer,
                            trackers=trackers,
                        )
                    break

                ring.append(packet)
                progress.update(task, completed=packet.frame_no)

                if reader_idx % detect_step == 0:
                    detections = self._infer(packet.image)
                    detected_classes = {d.class_name for d in detections}

                    if self._config.debug_phase1 and detections:
                        ts = packet.timestamp_ms
                        mm = int(ts // 60000)
                        ss = int((ts % 60000) // 1000)
                        classes_str = ", ".join(
                            f"{name}({sum(1 for d in detections if d.class_name == name)})"
                            for name in sorted(detected_classes)
                        )
                        _console.print(
                            f"  [phase1] {mm:02d}:{ss:02d} frame {packet.frame_no}: {classes_str}",
                            markup=False,
                        )

                    # open or extend a segment for each detected class
                    for cls_name in detected_classes:
                        if cls_name in active:
                            seg = active[cls_name]
                            seg.end_frame_no = packet.frame_no
                            seg.last_detect_reader_idx = reader_idx
                        else:
                            active[cls_name] = _SegmentState(
                                class_name=cls_name,
                                start_frame_no=packet.frame_no,
                                end_frame_no=packet.frame_no,
                                last_detect_reader_idx=reader_idx,
                                start_reader_idx=reader_idx,
                            )

                    # close segments where the gap has been exceeded
                    to_close = [
                        cls_name for cls_name, seg in active.items()
                        if cls_name not in detected_classes
                        and reader_idx - seg.last_detect_reader_idx >= max_gap_frames
                    ]
                    for cls_name in to_close:
                        seg = active.pop(cls_name)
                        if self._config.debug:
                            active_list = ", ".join(sorted(active.keys())) or "(none)"
                            _console.print(
                                f"  [debug] closing [{seg.class_name}] "
                                f"phase-1 frames {seg.start_frame_no}-{seg.end_frame_no}, "
                                f"remaining active: {active_list}",
                                markup=False,
                            )
                        self._close_segment(
                            seg, ring, offset_frames, csv_writer,
                            trackers=trackers,
                        )

                # --- buffer split: process long segments before ring evicts early frames
                for cls_name in list(active.keys()):
                    seg = active[cls_name]
                    if reader_idx - seg.start_reader_idx >= split_threshold:
                        if self._config.debug:
                            _console.print(
                                f"  [debug] splitting [{seg.class_name}] "
                                f"at frame {packet.frame_no} "
                                f"(segment started at {seg.start_frame_no})",
                                markup=False,
                            )
                        self._close_segment(
                            seg, ring, offset_frames, csv_writer,
                            is_split=True, split_end_frame=packet.frame_no,
                            trackers=trackers,
                        )
                        seg.window_start_override = packet.frame_no + 1
                        seg.start_reader_idx = reader_idx

                reader_idx += 1

    # --------------------------------------------------
    # phase 2: segment re-pass
    # --------------------------------------------------

    def _close_segment(
        self,
        seg: _SegmentState,
        ring: deque,
        offset_frames: int,
        csv_writer: "csv.writer | None",
        is_split: bool = False,
        split_end_frame: int | None = None,
        trackers: dict | None = None,
    ) -> None:
        """re-run yolo on all ring buffer frames in the segment window.

        collects frames from the ring buffer in [start - offset, end + offset],
        runs yolo on each, filters detections to the segment class, then checks
        significance and writes csv.

        for split segments (is_split=True): no end offset is added, the window
        ends at split_end_frame. for continuations after a split
        (window_start_override set): no start offset is added.
        """
        # --- window bounds
        if seg.window_start_override is not None:
            window_start = seg.window_start_override
        else:
            window_start = seg.start_frame_no - offset_frames

        if is_split:
            window_end = split_end_frame
        else:
            window_end = seg.end_frame_no + offset_frames

        window = sorted(
            (p for p in ring if window_start <= p.frame_no <= window_end),
            key=lambda p: p.frame_no,
        )

        if not window:
            if not is_split:
                print(
                    f"[{seg.class_name}] segment {seg.start_frame_no}-{seg.end_frame_no} "
                    f"lost from ring buffer, skipping"
                )
            return

        if self._config.debug:
            split_tag = " (split)" if is_split else ""
            cont_tag = " (cont)" if seg.window_start_override is not None else ""
            _console.print(
                f"  [debug] phase-2 [{seg.class_name}] window "
                f"frames {window[0].frame_no}-{window[-1].frame_no} "
                f"({len(window)} frames){split_tag}{cont_tag}",
                markup=False,
            )

        # re-run yolo on every frame in the window, keep only this class
        detected: list[tuple[FramePacket, list[Detection]]] = []
        cfg = self._config

        # get or create tracker (persists across splits for id continuity)
        tracker = None
        if cfg.tracking_enabled:
            if trackers is not None and seg.class_name in trackers:
                tracker = trackers[seg.class_name]
            else:
                tracker = SimpleTracker(self._eff_track_dist_limits, self._eff_track_max_lost)
                if trackers is not None:
                    trackers[seg.class_name] = tracker

        for packet in window:
            dets = self._infer(packet.image)
            h, w = packet.image.shape[:2]
            class_dets = [d for d in dets if d.class_name == seg.class_name]
            for det in class_dets:
                det.timestamp_ms = packet.timestamp_ms
            if tracker is not None:
                tracker.update(class_dets, w, h)
            if class_dets:
                detected.append((packet, class_dets))

        # --- reassign track ids right-to-left only for complete segments
        is_complete = seg.window_start_override is None and not is_split
        if cfg.tracking_enabled and is_complete:
            reassign_ids_right_to_left(detected)

        # --- segment output: significance check, postprocessing, classify, write
        self._process_segment(seg, window, detected, csv_writer)

        # cleanup tracker on final close
        if not is_split and trackers is not None and seg.class_name in trackers:
            del trackers[seg.class_name]

    def _process_segment(
        self,
        seg: _SegmentState,
        window: list[FramePacket],
        detected: list[tuple[FramePacket, list[Detection]]],
        csv_writer: "csv.writer | None",
    ) -> None:
        """orchestrate significance check, postprocessing, classification and csv write.

        order of operations:
          1. significance check per track (min frames + confidence)
          2. optional postprocessing on significant tracks (_postprocess_tracks)
          3. classify surviving detections
          4. write csv rows

        when tracking_enabled=False the whole segment is treated as one group (no
        per-track split, no postprocessing).
        """
        cfg = self._config
        start_ms = window[0].timestamp_ms
        start_min = int(start_ms // 60000)
        start_sec = int((start_ms % 60000) // 1000)

        if not cfg.tracking_enabled:
            # no tracking: significance check on the whole segment
            is_sig, conf_info = self._is_significant(detected)
            n_dets = len(detected)
            status = "significant" if is_sig else "false positive"
            _console.print(
                f"[{seg.class_name}] {start_min:02d}:{start_sec:02d} "
                f"frames {seg.start_frame_no}-{seg.end_frame_no} "
                f"({n_dets}/{len(window)} frames with detection{conf_info}) -> {status}",
                markup=False,
            )
            if is_sig and csv_writer is not None:
                rows = self._classify(detected)
                self._write_rows(csv_writer, rows)
            return

        # --- split detected by track_id (single pass)
        per_track: dict[int | None, list[tuple[FramePacket, list[Detection]]]] = {}
        for packet, dets in detected:
            by_tid: dict[int | None, list[Detection]] = {}
            for det in dets:
                by_tid.setdefault(det.track_id, []).append(det)
            for tid, tdets in by_tid.items():
                per_track.setdefault(tid, []).append((packet, tdets))

        # --- step 1: significance check per track, collect survivors
        ordered_ids: list[int | None] = sorted(k for k in per_track if k is not None)
        if None in per_track:
            ordered_ids.append(None)  # orphan detections last

        sig_per_track: dict[int | None, list[tuple[FramePacket, list[Detection]]]] = {}
        for tid in ordered_ids:
            track_data = per_track[tid]
            is_sig, conf_info = self._is_significant(track_data)
            n_dets = len(track_data)
            track_label = f" #{tid}" if tid is not None else " #?"
            status = "significant" if is_sig else "false positive"
            _console.print(
                f"[{seg.class_name}{track_label}] {start_min:02d}:{start_sec:02d} "
                f"frames {seg.start_frame_no}-{seg.end_frame_no} "
                f"({n_dets}/{len(window)} frames with detection{conf_info}) -> {status}",
                markup=False,
            )
            if is_sig:
                sig_per_track[tid] = track_data

        if not sig_per_track:
            return

        # --- step 2: optional postprocessing on significant tracks only
        sig_ids: set[int] = {tid for tid in sig_per_track if tid is not None}
        if cfg.segment_postprocess and sig_ids:
            # restrict detected to significant track ids for postprocess computations
            detected_sig = [
                (pkt, [d for d in dets if d.track_id in sig_ids])
                for pkt, dets in detected
            ]
            detected_sig = [(pkt, dets) for pkt, dets in detected_sig if dets]
            kept_ids = self._postprocess_tracks(
                detected_sig, sig_ids,
                detection_side=cfg.detection_side,
                center_line_x=cfg.center_line_x,
            )
            for tid in sig_ids - kept_ids:
                del sig_per_track[tid]

        if not sig_per_track:
            return

        # --- step 3: classify all surviving tracks and write csv
        all_sig: list[tuple[FramePacket, list[Detection]]] = [
            entry for track_data in sig_per_track.values() for entry in track_data
        ]
        all_sig.sort(key=lambda x: x[0].frame_no)

        if csv_writer is not None:
            rows = self._classify(all_sig)
            self._write_rows(csv_writer, rows)

    def _is_significant(
        self,
        detected: list[tuple[FramePacket, list[Detection]]],
    ) -> tuple[bool, str]:
        """check if a detection sequence passes the significance threshold.

        returns (is_significant, conf_info_str) where conf_info_str is appended
        to the log line (empty string if confidence check is disabled).
        """
        cfg = self._config
        n_dets = len(detected)
        is_sig = n_dets >= self._eff_min_detection_frames

        conf_info = ""
        if is_sig and self._eff_min_confident_frames > 0:
            n_confident = sum(
                1 for _, dets in detected
                if any(d.confidence >= cfg.min_segment_confidence for d in dets)
            )
            if n_confident < self._eff_min_confident_frames:
                is_sig = False
            conf_info = f", {n_confident} confident"

        return is_sig, conf_info

    def _postprocess_tracks(
        self,
        detected: list[tuple[FramePacket, list[Detection]]],
        track_ids: set[int],
        detection_side: str,
        center_line_x: float,
    ) -> set[int]:
        """filter tracks per segment before significance check and classification.

        three steps applied in order:
          1. side filter - drop tracks whose average last-N center is on the wrong side
          2. area filter - drop tracks whose max relative bbox area is below threshold
          3. center-proximity - for temporally overlapping tracks, keep only the one
             closest to center_line_x

        args:
            detected: list of (FramePacket, detections) for the segment
            track_ids: candidate track ids to filter
            detection_side: "left", "right", or "both"
            center_line_x: reference vertical line in [-1, 1]; 0 = image center

        returns:
            filtered set of track ids
        """
        if not track_ids:
            return track_ids

        cfg = self._config
        n_last = cfg.postprocess_last_frames
        remaining = set(track_ids)

        # --- build per-track appearance list: list of (frame_no, det) in order
        track_appearances: dict[int, list[tuple[int, Detection, int, int]]] = {
            tid: [] for tid in remaining
        }
        for packet, dets in detected:
            h, w = packet.image.shape[:2]
            for det in dets:
                if det.track_id in track_appearances:
                    track_appearances[det.track_id].append((packet.frame_no, det, w, h))

        def _avg_cx_rel(tid: int) -> float:
            """average center x in [-1,1] over last n_last appearances."""
            appearances = track_appearances[tid]
            last = appearances[-n_last:] if len(appearances) >= n_last else appearances
            cx_norm = sum((det.bbox[0] + det.bbox[2]) / 2 / w for _, det, w, _ in last) / len(last)
            return (cx_norm - 0.5) * 2

        # --- step a: side filter
        if detection_side != "both":
            drop_side: set[int] = set()
            for tid in remaining:
                cx_rel = _avg_cx_rel(tid)
                keep = (
                    cx_rel <= center_line_x if detection_side == "left"
                    else cx_rel >= center_line_x
                )
                if not keep:
                    drop_side.add(tid)
                    if cfg.debug:
                        _console.print(
                            f"  [debug] postprocess: drop track #{tid} "
                            f"(side filter: cx_rel={cx_rel:.3f}, side={detection_side}, "
                            f"center_line_x={center_line_x})",
                            markup=False,
                        )
            remaining -= drop_side

        if not remaining:
            return remaining

        # --- step b: area filter
        if cfg.min_track_max_area > 0.0:
            drop_area: set[int] = set()
            for tid in remaining:
                max_area = max(
                    (det.bbox[2] - det.bbox[0]) * (det.bbox[3] - det.bbox[1]) / (w * h)
                    for _, det, w, h in track_appearances[tid]
                )
                if max_area < cfg.min_track_max_area:
                    drop_area.add(tid)
                    if cfg.debug:
                        _console.print(
                            f"  [debug] postprocess: drop track #{tid} "
                            f"(area filter: max_area={max_area:.5f} < {cfg.min_track_max_area})",
                            markup=False,
                        )
            remaining -= drop_area

        if len(remaining) <= 1:
            return remaining

        # --- step c: center-proximity selection for temporally overlapping groups
        # compute frame range per track
        frame_ranges: dict[int, tuple[int, int]] = {
            tid: (
                track_appearances[tid][0][0],
                track_appearances[tid][-1][0],
            )
            for tid in remaining
        }

        # group tracks by overlap (simple connected-component on range intersections)
        tids = sorted(remaining)
        groups: list[set[int]] = []
        for tid in tids:
            t_start, t_end = frame_ranges[tid]
            merged: list[int] = []
            for i, group in enumerate(groups):
                for gtid in group:
                    g_start, g_end = frame_ranges[gtid]
                    if t_start <= g_end and t_end >= g_start:
                        merged.append(i)
                        break
            if not merged:
                groups.append({tid})
            else:
                # merge all overlapping groups plus current tid
                base = groups[merged[0]]
                base.add(tid)
                for i in reversed(merged[1:]):
                    base |= groups.pop(i)

        drop_center: set[int] = set()
        for group in groups:
            if len(group) <= 1:
                continue
            # keep track closest to center_line_x based on last-n avg cx
            best_tid = min(group, key=lambda tid: abs(_avg_cx_rel(tid) - center_line_x))
            for tid in group:
                if tid != best_tid:
                    drop_center.add(tid)
                    if cfg.debug:
                        _console.print(
                            f"  [debug] postprocess: drop track #{tid} "
                            f"(center-proximity: cx_rel={_avg_cx_rel(tid):.3f}, "
                            f"kept #{best_tid} cx_rel={_avg_cx_rel(best_tid):.3f})",
                            markup=False,
                        )
        remaining -= drop_center
        return remaining

    def _classify(
        self,
        detected: list[tuple[FramePacket, list[Detection]]],
    ) -> list[ClassifiedRow]:
        """run classifier on all detections, return flat list of rows ready to write."""
        classify_classes = self._config.classify_classes
        rows: list[ClassifiedRow] = []
        for packet, dets in detected:
            for det in dets:
                if self._classifier is not None and det.class_name in classify_classes:
                    crop = _crop(packet.image, det.bbox)
                    label_cls, conf_cls = self._classifier(crop, det)
                    conf_cls_str = f"{conf_cls:.4f}"
                else:
                    label_cls, conf_cls_str = "", ""
                rows.append((packet, det, label_cls, conf_cls_str))
        return rows

    def _write_rows(
        self,
        writer: "csv.writer",
        rows: list[ClassifiedRow],
    ) -> None:
        """write pre-classified detection rows to csv."""
        cfg = self._config
        for packet, det, label_cls, conf_cls_str in rows:
            x1, y1, x2, y2 = det.bbox
            if cfg.box_relative:
                h, w = packet.image.shape[:2]
                bx1, by1 = x1 / w, y1 / h
                bx2, by2 = x2 / w, y2 / h
                bbox_vals = [f"{bx1:.6f}", f"{by1:.6f}", f"{bx2:.6f}", f"{by2:.6f}"]
            else:
                bbox_vals = [x1, y1, x2, y2]
            tid_str = str(det.track_id) if det.track_id is not None else ""
            writer.writerow([
                packet.frame_no,
                f"{packet.timestamp_ms:.1f}",
                *bbox_vals,
                det.class_name,
                tid_str,
                f"{det.confidence:.4f}",
                label_cls,
                conf_cls_str,
            ])

    # --------------------------------------------------
    # inference
    # --------------------------------------------------

    def _infer(self, image: np.ndarray) -> list[Detection]:
        """run yolo inference on a single bgr frame."""
        cfg = self._config
        results = self._model(
            image,
            imgsz=cfg.imgsz,
            conf=cfg.detection_confidence,
            device=cfg.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = self._model.names.get(class_id, str(class_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(Detection(
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(box.conf.item()),
                ))

        if cfg.device.startswith("cuda"):
            torch.cuda.empty_cache()

        return detections


# --------------------------------------------------
# module-level helpers
# --------------------------------------------------

def _crop(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """return an independent copy of the bbox region from image."""
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    return image[max(0, y1):min(h, y2), max(0, x1):min(w, x2)].copy()
