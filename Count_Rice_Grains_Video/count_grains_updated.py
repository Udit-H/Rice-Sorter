#!/usr/bin/env python3
"""
Rice grain counter — centroid tracking, unique-ID counting.

Each grain is assigned an ID the first time it is detected.
The final count is the total number of unique IDs ever assigned,
so a grain visible across N frames is counted exactly once.

v3 — Tuned for 60fps fast conveyor where grains are visible for 2-3 frames.
Key improvements over v2:
  - Refined belt mask with edge exclusion zone
  - Per-frame spike guard to reject lighting artifacts
  - Hungarian-algorithm tracker (globally optimal matching)
  - Aggressive max_missed (~3 frames) so ghost tracks don't absorb new grains
  - Dynamic split_area based on median grain size
  - Lower min_area for broken grain support
"""

import cv2
import numpy as np
import os
import sys
import argparse
from typing import Optional, List, Tuple
from collections import deque

# Try to import scipy for Hungarian algorithm; fall back to greedy if unavailable
try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Colour detection ──────────────────────────────────────────────────────────

def _auto_mode(frame: np.ndarray) -> str:
    """Infer rice type from the luminance distribution of the first frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(float)
    dark_frac   = np.mean(v < 80)
    bright_frac = np.mean(v > 150)
    # If the frame has many dark pixels that rival the bright ones → dark rice
    if dark_frac > 0.15 and dark_frac > bright_frac * 0.5:
        return "dark"
    return "white"


def _make_mask(frame: np.ndarray, mode: str, edge_margin: int = 30) -> np.ndarray:
    """Build a binary mask of grain pixels.

    Parameters
    ----------
    frame : BGR image
    mode  : "auto"/"mixed"/"universal" uses belt-background subtraction;
            "dark"/"brown"/"white" use legacy colour ranges.
    edge_margin : pixels to exclude at top/bottom edges to avoid belt-edge
                  reflections and conveyor frame false positives.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, w = frame.shape[:2]

    if mode in ("auto", "mixed", "universal"):
        # 1. Isolate the blue conveyor belt (expanded range to catch edge reflections)
        bg_belt = cv2.inRange(hsv, (85, 40, 30), (145, 255, 255))

        # 2. Isolate shadows on the belt
        bg_shadow = cv2.inRange(hsv, (85, 30, 5), (145, 255, 80))

        # 3. Everything else is foreground (grains of any colour)
        bg_combined = cv2.bitwise_or(bg_belt, bg_shadow)
        fg = cv2.bitwise_not(bg_combined)

        # 4. Filter out pure black camera noise / very dark pixels
        valid_v = cv2.inRange(hsv, (0, 0, 20), (180, 255, 255))
        m = cv2.bitwise_and(fg, valid_v)

        # 5. Edge exclusion moved to contour-level filtering in detect_grains()
        #    This is more precise than blanket pixel exclusion.

    elif mode == "dark":
        m = cv2.inRange(hsv, (0, 0, 10), (180, 255, 75))
    elif mode == "brown":
        m = cv2.inRange(hsv, (5, 20, 60), (30, 210, 210))
    else:  # white / chalky
        m  = cv2.inRange(hsv, (0, 0, 100), (180, 90, 255))
        m |= cv2.inRange(hsv, (0, 0, 75), (180, 55, 200))
        belt = cv2.inRange(hsv, (90, 70, 40), (135, 255, 255))
        m = cv2.bitwise_and(m, cv2.bitwise_not(belt))

    # Morphological cleanup
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=2)
    return m


def detect_grains(
    frame: np.ndarray,
    mode: str,
    min_area: int = 40,
    max_area: int = 20000,
    max_aspect_small: float = 6.0,
    small_threshold: int = 150,
    border_margin: int = 2,
) -> Tuple[list, np.ndarray]:
    """Detect grain contours in the frame.

    Filtering applied:
    1. Area filter: rejects contours outside [min_area, max_area].
    2. Shape filter: rejects small contours (<small_threshold area) with
       aspect ratio > max_aspect_small (noise streaks from motion blur).
    3. Border filter: rejects contours whose bounding box touches the frame
       edge (within border_margin px). Belt structure artifacts always touch
       the frame border; real grains near the edge do not.

    Returns (valid_contours, binary_mask).
    """
    mask = _make_mask(frame, mode)
    h_frame, w_frame = frame.shape[:2]
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area <= min_area or area >= max_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # Border filter: reject contours whose bounding box touches the
        # frame edge.  Belt structure / conveyor frame artifacts always
        # touch the border; real grains in the interior do not.
        if border_margin > 0:
            if (x <= border_margin or y <= border_margin or
                    x + w >= w_frame - border_margin or
                    y + h >= h_frame - border_margin):
                continue
        # Shape filter: reject high-aspect-ratio small blobs (noise streaks)
        if area < small_threshold:
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > max_aspect_small:
                continue
        valid.append(c)
    return valid, mask


def _split_blob(contour, typical_area: int = 600, max_splits: int = 5) -> List[Tuple[int, int]]:
    """Return N centroids for a large blob via distance-transform peak finding.

    Parameters
    ----------
    max_splits : hard cap on the number of centroids produced per blob.
                 Prevents a single large contour (e.g. belt reflection)
                 from generating dozens of phantom centroids.
    """
    area = cv2.contourArea(contour)
    n = max(1, round(area / typical_area))
    n = min(n, max_splits)  # hard cap
    if n == 1:
        M = cv2.moments(contour)
        if M["m00"] > 0:
            return [(int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))]
        return []
    x, y, w, h = cv2.boundingRect(contour)
    local = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.drawContours(local, [contour - np.array([[x - 1, y - 1]])], -1, 255, -1)
    dist = cv2.distanceTransform(local, cv2.DIST_L2, 5)
    radius = max(4, int((typical_area / np.pi) ** 0.5 * 0.6))
    result, d = [], dist.copy()
    for _ in range(n):
        _, max_val, _, max_loc = cv2.minMaxLoc(d)
        if max_val < 2:
            break
        result.append((x - 1 + max_loc[0], y - 1 + max_loc[1]))
        cv2.circle(d, max_loc, radius, 0, -1)
    return result or [(x + w // 2, y + h // 2)]


def centroids_of(
    contours: list,
    split_area: int = 1400,
    default_typical_area: int = 750,
) -> List[Tuple[int, int]]:
    """Extract one centroid per grain, splitting large clusters.

    Parameters
    ----------
    contours : list of contours from detect_grains
    split_area : contours larger than this are treated as clusters
    default_typical_area : fallback single-grain area for splitting
    """
    pts = []

    # Dynamically calculate the median area of single, unclustered grains
    single_areas = [
        cv2.contourArea(c) for c in contours
        if 60 < cv2.contourArea(c) < split_area
    ]
    typical_grain_area = (
        int(np.median(single_areas)) if single_areas else default_typical_area
    )

    # Dynamically adjust split_area based on typical grain size
    # A cluster should be at least ~2.5× the typical grain
    effective_split = max(split_area, int(typical_grain_area * 2.5))

    for c in contours:
        if cv2.contourArea(c) > effective_split:
            pts.extend(_split_blob(c, typical_grain_area))
        else:
            M = cv2.moments(c)
            if M["m00"] > 0:
                pts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return pts


# ── Tracker ───────────────────────────────────────────────────────────────────

class UniqueGrainTracker:
    """Track grain centroids across frames.

    Tuned for fast conveyors at 60fps where each grain is visible for only
    2-3 frames.  Key design decisions:

    - **max_dist = 45 px**: at 60fps a grain moves ~20-35 px/frame; 45 px
      prevents matching adjacent grains while accommodating movement.
    - **max_missed = 3 frames**: a grain that vanishes for 3+ frames is gone.
      Keeping ghost tracks alive longer would cause new grains in similar
      positions to be absorbed → undercounting.
    - **Hungarian matching**: globally optimal assignment avoids greedy
      conflicts in dense frames.
    """

    # Area threshold for broken vs whole grain classification.
    # Based on analysis: broken grains median ~150px², whole grains median ~580-972px².
    BROKEN_AREA_THRESHOLD = 300

    def __init__(self, max_dist: int = 45, max_missed: int = 3):
        self.max_dist   = max_dist
        self.max_missed = max_missed
        self._tracks: dict = {}   # id → {"centroid": (x,y), "missed": int, "area": float}
        self._nid = 0             # total unique grains ever assigned
        self._grain_areas: dict = {}  # id → area (stored permanently for classification)

    def _match_hungarian(self, dets: list):
        """Globally optimal matching via the Hungarian algorithm."""
        if not self._tracks or not dets:
            return {}, set()

        track_ids = list(self._tracks.keys())
        n_tracks = len(track_ids)
        n_dets   = len(dets)

        # Build cost matrix
        cost = np.full((n_tracks, n_dets), 1e9, dtype=np.float64)
        for ti, tid in enumerate(track_ids):
            tx, ty = self._tracks[tid]["centroid"]
            for di, (cx, cy) in enumerate(dets):
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d <= self.max_dist:
                    cost[ti, di] = d

        if _HAS_SCIPY:
            row_ind, col_ind = linear_sum_assignment(cost)
        else:
            # Fallback: greedy matching sorted by distance
            return self._match_greedy(dets)

        matched_t: dict = {}
        matched_d: set  = set()
        for ti, di in zip(row_ind, col_ind):
            if cost[ti, di] < 1e8:   # only valid assignments
                matched_t[track_ids[ti]] = di
                matched_d.add(di)

        return matched_t, matched_d

    def _match_greedy(self, dets: list):
        """Fallback greedy nearest-neighbour matching."""
        matched_t: dict = {}
        matched_d: set  = set()
        # Sort tracks by distance to their nearest detection for better ordering
        pairs = []
        for tid, t in self._tracks.items():
            tx, ty = t["centroid"]
            for di, (cx, cy) in enumerate(dets):
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d <= self.max_dist:
                    pairs.append((d, tid, di))
        pairs.sort()
        for d, tid, di in pairs:
            if tid in matched_t or di in matched_d:
                continue
            matched_t[tid] = di
            matched_d.add(di)
        return matched_t, matched_d

    def update(self, centroids: list, areas: Optional[list] = None) -> int:
        """Update tracker with new frame detections.

        Parameters
        ----------
        centroids : list of (cx, cy) tuples
        areas     : optional list of contour areas, same length as centroids.
                    Used for broken/whole grain classification.

        Returns current unique count.
        """
        matched_t, matched_d = self._match_hungarian(centroids)

        # Update matched tracks
        for tid, di in matched_t.items():
            self._tracks[tid]["centroid"] = centroids[di]
            self._tracks[tid]["missed"]   = 0
            # Update area with max seen (grain may be partially visible initially)
            if areas and di < len(areas):
                prev = self._grain_areas.get(tid, 0)
                self._grain_areas[tid] = max(prev, areas[di])

        # Penalise unmatched existing tracks
        for tid in list(self._tracks):
            if tid not in matched_t:
                self._tracks[tid]["missed"] += 1

        # Register new tracks — each one is a new unique grain
        for di, c in enumerate(centroids):
            if di not in matched_d:
                a = areas[di] if areas and di < len(areas) else 0
                self._tracks[self._nid] = {"centroid": c, "missed": 0}
                self._grain_areas[self._nid] = a
                self._nid += 1

        # Prune dead tracks
        self._tracks = {
            k: v for k, v in self._tracks.items()
            if v["missed"] <= self.max_missed
        }
        return self._nid

    def get_classification(self) -> dict:
        """Return grain count breakdown: total, whole, broken."""
        total = self._nid
        broken = sum(1 for a in self._grain_areas.values()
                     if a < self.BROKEN_AREA_THRESHOLD)
        whole = total - broken
        return {"total": total, "whole": whole, "broken": broken}


# ── Spike guard ───────────────────────────────────────────────────────────────

class SpikeGuard:
    """Reject frames with anomalously high detection counts.

    Maintains a rolling window of per-frame detection counts and suppresses
    frames where the count exceeds `multiplier × rolling_median`.
    This prevents lighting flicker / belt reflection artifacts from creating
    hundreds of phantom tracks (e.g. test_15_brown_200 frame 62: 189 detections).
    """

    def __init__(self, window: int = 60, multiplier: float = 3.0, min_threshold: int = 20):
        self._window = deque(maxlen=window)
        self._multiplier = multiplier
        self._min_threshold = min_threshold  # don't suppress below this absolute count

    def is_spike(self, count: int) -> bool:
        """Return True if this frame's detection count looks anomalous."""
        if len(self._window) < 10:
            # Not enough history yet — only suppress extreme spikes
            self._window.append(count)
            return count > self._min_threshold * 3

        median = float(np.median(self._window))
        # Threshold: at least min_threshold, or multiplier × running median
        threshold = max(self._min_threshold, median * self._multiplier)
        self._window.append(count)
        return count > threshold

    def record(self, count: int):
        """Record a (non-spike) frame's count for future reference."""
        self._window.append(count)


# ── Main processing function ──────────────────────────────────────────────────

def process_video(
    video_path: str,
    rice_mode: str = "auto",
    out_path: Optional[str] = None,
    verbose: bool = True,
    max_dist: int = 45,
    max_missed: Optional[int] = None,
    edge_margin: int = 30,
    spike_guard: bool = True,
) -> dict:
    """
    Count rice grains in *video_path*.

    Returns
    -------
    dict with keys:
        total  : int — total unique grains counted
        whole  : int — grains classified as whole (area >= 300px²)
        broken : int — grains classified as broken (area < 300px²)

    Parameters
    ----------
    video_path  : path to input video
    rice_mode   : "white", "dark", "brown", or "auto" (universal belt subtraction)
    out_path    : optional path to save an annotated output video
    verbose     : print per-video stats
    max_dist    : max centroid distance for track matching (px)
    max_missed  : frames before a track is pruned (default: auto from fps)
    edge_margin : pixels to exclude at top/bottom frame edges
    spike_guard : enable per-frame spike detection and suppression
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return {"total": -1, "whole": 0, "broken": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # In this updated implementation, "auto" uses the universal belt-background
    # subtraction pipeline implemented in _make_mask().
    mode = rice_mode

    # Compute max_missed from FPS if not specified.
    # At 60fps, grains are visible for 2-3 frames. Keep ghosts for ~4 frames max.
    if max_missed is None:
        max_missed = max(3, int(fps * 0.07))

    tracker = UniqueGrainTracker(max_dist=max_dist, max_missed=max_missed)
    guard   = SpikeGuard() if spike_guard else None

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    frame_idx = 0
    spike_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        contours, mask = detect_grains(frame, mode, min_area=40, max_area=5000)
        cens = centroids_of(contours)
        # Collect contour areas for broken/whole classification
        grain_areas = [cv2.contourArea(c) for c in contours]

        # Spike guard: skip frames with anomalous detection counts
        if guard and guard.is_spike(len(cens)):
            spike_count += 1
            if writer:
                vis = frame.copy()
                cls = tracker.get_classification()
                cv2.putText(
                    vis, f"Total:{cls['total']} Whole:{cls['whole']} Broken:{cls['broken']}  SPIKE",
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2,
                )
                writer.write(vis)
            frame_idx += 1
            continue

        cnt = tracker.update(cens, areas=grain_areas)

        if writer:
            vis = frame.copy()
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                area = cv2.contourArea(c)
                # Green for whole, red for broken
                color = (0, 220, 0) if area >= UniqueGrainTracker.BROKEN_AREA_THRESHOLD else (0, 80, 255)
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)
            for t in tracker._tracks.values():
                cx, cy = t["centroid"]
                cv2.circle(vis, (cx, cy), 5, (0, 220, 255), -1)
            cls = tracker.get_classification()
            cv2.putText(
                vis, f"Total:{cls['total']} Whole:{cls['whole']} Broken:{cls['broken']}  [{mode}]",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
            )
            writer.write(vis)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()

    result = tracker.get_classification()

    if verbose:
        extra = f"  spikes_skipped={spike_count}" if spike_count else ""
        print(f"  mode={mode}  total={result['total']}  "
              f"whole={result['whole']}  broken={result['broken']}  "
              f"frames={frame_idx}  max_missed={max_missed}  "
              f"max_dist={max_dist}{extra}")
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(description="Count rice grains in a video.")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "mixed", "universal", "white", "dark", "brown"],
                    help="Rice colour profile (default: auto)")
    ap.add_argument("--out", default=None,
                    help="Save annotated output video to this path")
    ap.add_argument("--max-dist", type=int, default=45,
                    help="Max centroid distance for track matching (px)")
    ap.add_argument("--max-missed", type=int, default=None,
                    help="Frames before a track is pruned (default: auto)")
    ap.add_argument("--edge-margin", type=int, default=30,
                    help="Pixels to exclude at top/bottom frame edges")
    ap.add_argument("--no-spike-guard", action="store_true",
                    help="Disable per-frame spike detection")
    args = ap.parse_args()

    result = process_video(
        args.video,
        rice_mode=args.mode,
        out_path=args.out,
        max_dist=args.max_dist,
        max_missed=args.max_missed,
        edge_margin=args.edge_margin,
        spike_guard=not args.no_spike_guard,
    )
    print(f"\nFinal count: {result['total']}  (whole: {result['whole']}, broken: {result['broken']})")
    return result


if __name__ == "__main__":
    _cli()
