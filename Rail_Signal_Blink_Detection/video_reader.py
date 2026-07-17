"""
video reader module - decodes video frames, downsamples to target fps,
and applies preprocessing before feeding frames into a buffer queue.
"""

import queue
import threading

import cv2
import numpy as np

from .pipeline_config import (
    COVER_SOURCE_BOX,
    COVER_TARGET_BOX,
    FramePacket,
    PipelineConfig,
    _REFERENCE_FPS,
)


# VideoReader

class VideoReader:
    """reads a video file, downsamples to reader_fps, preprocesses frames,
    and puts FramePacket items into output_queue.

    sends a None sentinel when the video is exhausted or stop() is called.
    """

    def __init__(
        self,
        video_path: str,
        config: PipelineConfig,
        output_queue: "queue.Queue[FramePacket | None]",
    ) -> None:
        self._path = video_path
        self._config = config
        self._output_queue = output_queue
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="VideoReader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    # -----

    def _worker(self) -> None:
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            self._output_queue.put(None)
            return

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        # read every N-th source frame to achieve target reader_fps;
        # "full" caps at _REFERENCE_FPS to avoid processing unnecessarily high fps
        if self._config.reader_fps == "full":
            frame_step = max(1, round(source_fps / _REFERENCE_FPS))
        else:
            frame_step = max(1, round(source_fps / self._config.reader_fps))

        source_frame_no = 0
        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                if source_frame_no % frame_step == 0:
                    timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    if self._config.corner_cover:
                        frame = self._cover_corner(frame)

                    packet = FramePacket(
                        frame_no=source_frame_no,
                        timestamp_ms=timestamp_ms,
                        image=frame,
                    )
                    self._enqueue(packet)

                source_frame_no += 1
        finally:
            cap.release()

        # sentinel - signals end of stream to the downstream consumer
        if not self._stop_event.is_set():
            try:
                self._output_queue.put(None, timeout=1.0)
            except queue.Full:
                pass

    def _enqueue(self, packet: FramePacket) -> None:
        """blocks until space is available or stop is requested."""
        while not self._stop_event.is_set():
            try:
                self._output_queue.put(packet, timeout=0.1)
                return
            except queue.Full:
                continue

    @staticmethod
    def _cover_corner(frame: np.ndarray) -> np.ndarray:
        """paints the upper-right corner with the average colour of the upper-left
        reference region - mirrors the cover_corner_region logic used during dataset
        preparation so inference sees the same input distribution.
        """
        h, w = frame.shape[:2]

        def yolo_to_pixel_box(cx: float, cy: float, bw: float, bh: float) -> tuple:
            x1 = max(0, int((cx - bw / 2) * w))
            y1 = max(0, int((cy - bh / 2) * h))
            x2 = min(w, int((cx + bw / 2) * w))
            y2 = min(h, int((cy + bh / 2) * h))
            return x1, y1, x2, y2

        sx1, sy1, sx2, sy2 = yolo_to_pixel_box(*COVER_SOURCE_BOX)
        tx1, ty1, tx2, ty2 = yolo_to_pixel_box(*COVER_TARGET_BOX)

        # average colour of the source region (BGR)
        src_region = frame[sy1:sy2, sx1:sx2]
        avg_color = src_region.mean(axis=(0, 1)).astype(np.uint8)

        frame = frame.copy()
        frame[ty1:ty2, tx1:tx2] = avg_color
        return frame
