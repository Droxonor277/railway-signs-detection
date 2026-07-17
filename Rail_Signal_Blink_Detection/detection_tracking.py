"""
simple centre-distance tracker for phase-2 detection pipeline.

assigns persistent integer ids to detections across frames based on
euclidean distance between normalised bounding box centres.

the matching distance threshold scales dynamically with bounding box area:
small (far) objects get a tight threshold to avoid cross-matching nearby
objects, large (close) objects get a loose threshold to tolerate faster
apparent motion. the threshold also scales with lost frame count, but only
for detections with area >= AREA_MIN - small/far objects skip the scaling
to avoid false matches between spatially distinct objects.
"""

import math
from dataclasses import dataclass

from .pipeline_config import Detection

# relative box area bounds for distance scaling
# boxes with area <= AREA_MIN use dist_limits[0]
# boxes with area >= AREA_MAX use dist_limits[1]
AREA_MIN: float = 0.0002
AREA_MAX: float = 0.01


def _area_to_dist(
    rel_area: float,
    dist_min: float,
    dist_max: float,
) -> float:
    """linearly interpolate distance threshold from relative bbox area."""
    if rel_area <= AREA_MIN:
        return dist_min
    if rel_area >= AREA_MAX:
        return dist_max
    t = (rel_area - AREA_MIN) / (AREA_MAX - AREA_MIN)
    return dist_min + t * (dist_max - dist_min)


@dataclass
class _Track:
    """a single tracked object."""
    track_id: int
    cx: float       # last known normalised centre x
    cy: float       # last known normalised centre y
    lost: int = 0   # consecutive frames without a match


class SimpleTracker:
    """assigns persistent integer ids to detections across frames.

    matching is based on euclidean distance between normalised bounding box
    centres. the distance threshold is dynamically computed from each
    detection's relative area and scales with the track's lost-frame count.

    update() must be called for EVERY frame in sequence - including frames
    with no detections (pass an empty list) - so that lost counters are
    properly incremented and dead tracks are cleaned up.

    usage:
        tracker = SimpleTracker(dist_limits=(0.02, 0.15), max_lost=10)
        for frame in frames:
            dets = model(frame)
            tracker.update(dets, frame_w, frame_h)
            # dets now have track_id set
    """

    def __init__(self, dist_limits: tuple[float, float], max_lost: int) -> None:
        self._dist_min = dist_limits[0]
        self._dist_max = dist_limits[1]
        self._max_lost = max_lost
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], frame_w: int, frame_h: int) -> None:
        """match detections to existing tracks, assign track_id in-place."""
        if frame_w == 0 or frame_h == 0:
            return

        total_area = frame_w * frame_h

        # compute normalised centres and per-detection distance thresholds
        det_info: list[tuple[float, float, float, float]] = []  # (cx, cy, max_dist, rel_area)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2 / frame_w
            cy = (y1 + y2) / 2 / frame_h
            rel_area = (x2 - x1) * (y2 - y1) / total_area
            max_dist = _area_to_dist(rel_area, self._dist_min, self._dist_max)
            det_info.append((cx, cy, max_dist, rel_area))

        matched_tracks: set[int] = set()  # indices into self._tracks
        matched_dets: set[int] = set()    # indices into detections

        # greedy nearest-neighbour: sort all (track, det) pairs by distance
        pairs: list[tuple[float, int, int]] = []
        for ti, track in enumerate(self._tracks):
            for di, (cx, cy, _, _) in enumerate(det_info):
                dist = math.sqrt((track.cx - cx) ** 2 + (track.cy - cy) ** 2)
                pairs.append((dist, ti, di))
        pairs.sort()

        for dist, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            track = self._tracks[ti]
            _, _, det_max_dist, rel_area = det_info[di]
            # skip lost-frame scaling for small/far objects (area < AREA_MIN)
            if rel_area < AREA_MIN:
                effective_dist = det_max_dist
            else:
                effective_dist = det_max_dist * (1 + track.lost)
            if dist > effective_dist:
                continue  # can't break early - thresholds vary per pair
            # match
            cx, cy, _, _ = det_info[di]
            track.cx = cx
            track.cy = cy
            track.lost = 0
            detections[di].track_id = track.track_id
            matched_tracks.add(ti)
            matched_dets.add(di)

        # create new tracks for unmatched detections
        for di in range(len(detections)):
            if di not in matched_dets:
                cx, cy, _, _ = det_info[di]
                new_track = _Track(track_id=self._next_id, cx=cx, cy=cy)
                self._next_id += 1
                self._tracks.append(new_track)
                detections[di].track_id = new_track.track_id

        # increment lost counter for unmatched tracks, remove dead ones
        alive: list[_Track] = []
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track.lost += 1
            if track.lost <= self._max_lost:
                alive.append(track)
        self._tracks = alive


def reassign_ids_right_to_left(
    detected: list[tuple],
) -> None:
    """reassign track ids so that id=1 is the rightmost object, id=2 next, etc.

    operates in-place on the Detection objects. uses the x-centre of each
    track's first appearance to determine right-to-left order.
    """
    # find the first x-centre for each track id
    first_cx: dict[int, float] = {}
    for _, dets in detected:
        for d in dets:
            if d.track_id is not None and d.track_id not in first_cx:
                x1, y1, x2, y2 = d.bbox
                first_cx[d.track_id] = (x1 + x2) / 2

    if not first_cx:
        return

    # sort track ids by first x-centre descending (right to left)
    sorted_ids = sorted(first_cx, key=lambda tid: first_cx[tid], reverse=True)
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted_ids, start=1)}

    # apply mapping
    for _, dets in detected:
        for d in dets:
            if d.track_id is not None:
                d.track_id = id_map[d.track_id]
