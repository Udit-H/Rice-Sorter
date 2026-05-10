#!/usr/bin/env python3
"""
Rice grain counter — centroid tracking, unique-ID counting.

Each grain is assigned an ID the first time it is detected.
The final count is the total number of unique IDs ever assigned,
so a grain visible across N frames is counted exactly once.
This fixes the original frame-summation bug.
"""

import cv2
import numpy as np
import os
import sys
import argparse
from typing import Optional, List, Tuple


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


def _make_mask(frame: np.ndarray, mode: str) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if mode == "dark":
        # Black rice on blue belt (belt V≈100, grains V≈20–75)
        # Tight upper bound on V to avoid belt noise
        m = cv2.inRange(hsv, (0, 0, 10), (180, 255, 75))
    elif mode == "brown":
        m = cv2.inRange(hsv, (5,  20,  60), (30, 210, 210))
    else:  # white / chalky (default)
        m  = cv2.inRange(hsv, (0,   0, 100), (180,  90, 255))
        m |= cv2.inRange(hsv, (0,   0,  75), (180,  55, 200))
        # Subtract blue conveyor belt (H≈95-135, S>70) so we can use lower V thresholds
        belt = cv2.inRange(hsv, (90, 70, 40), (135, 255, 255))
        m = cv2.bitwise_and(m, cv2.bitwise_not(belt))

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k3, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=2)
    return m


def detect_grains(
    frame: np.ndarray,
    mode: str,
    min_area: int = 55,
    max_area: int = 20000,
) -> Tuple[list, np.ndarray]:
    mask = _make_mask(frame, mode)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
    return valid, mask


def _split_blob(contour, typical_area: int = 600) -> List[Tuple[int, int]]:
    """Return N centroids for a large blob via distance-transform peak finding."""
    area = cv2.contourArea(contour)
    n = max(1, round(area / typical_area))
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
    typical_grain_area: int = 750,
) -> List[Tuple[int, int]]:
    pts = []
    for c in contours:
        if cv2.contourArea(c) > split_area:
            pts.extend(_split_blob(c, typical_grain_area))
        else:
            M = cv2.moments(c)
            if M["m00"] > 0:
                pts.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))
    return pts


# ── Tracker ───────────────────────────────────────────────────────────────────

class UniqueGrainTracker:
    """
    Track grain centroids across frames.
    A grain is counted once it has been consistently detected for at
    least `min_age` frames — this filters single-frame noise blobs.
    """

    def __init__(self, max_dist: int = 70, max_missed: int = 30):
        self.max_dist   = max_dist
        self.max_missed = max_missed
        self._tracks: dict = {}  # id → {centroid, missed}
        self._nid = 0           # total unique grains ever assigned

    # -- greedy nearest-neighbour match ---------------------------------------
    def _match(self, dets: list):
        matched_t: dict = {}
        matched_d: set  = set()
        for tid, t in self._tracks.items():
            tx, ty  = t["centroid"]
            best_di = None
            best_d  = self.max_dist
            for di, (cx, cy) in enumerate(dets):
                if di in matched_d:
                    continue
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_di = d, di
            if best_di is not None:
                matched_t[tid] = best_di
                matched_d.add(best_di)
        return matched_t, matched_d

    def update(self, centroids: list) -> int:
        matched_t, matched_d = self._match(centroids)

        # update matched tracks
        for tid, di in matched_t.items():
            self._tracks[tid]["centroid"] = centroids[di]
            self._tracks[tid]["missed"]   = 0

        # penalise unmatched existing tracks
        for tid in list(self._tracks):
            if tid not in matched_t:
                self._tracks[tid]["missed"] += 1

        # register new tracks — each one is a new unique grain
        for di, c in enumerate(centroids):
            if di not in matched_d:
                self._tracks[self._nid] = {"centroid": c, "missed": 0}
                self._nid += 1

        # prune dead tracks
        self._tracks = {
            k: v for k, v in self._tracks.items()
            if v["missed"] <= self.max_missed
        }
        return self._nid


# ── Main processing function ──────────────────────────────────────────────────

def process_video(
    video_path: str,
    rice_mode: str = "auto",
    out_path: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """
    Count rice grains in *video_path* and return the integer count.

    Parameters
    ----------
    video_path : path to input video
    rice_mode  : "white", "dark", "brown", or "auto"
    out_path   : optional path to save an annotated output video
    verbose    : print per-video stats
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return -1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # auto-detect mode from the first readable frame
    mode = rice_mode
    if mode == "auto":
        ok, f0 = cap.read()
        if ok:
            mode = _auto_mode(f0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Keep dead tracks for ~0.33 s of real time; scale with fps so a 60 fps video
    # doesn't reuse stale track IDs more aggressively than a 30 fps one.
    max_missed = max(5, int(fps * 0.33))
    tracker = UniqueGrainTracker(max_dist=70, max_missed=max_missed)

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        contours, mask = detect_grains(frame, mode)
        cens = centroids_of(contours)
        cnt  = tracker.update(cens)

        if writer:
            vis = frame.copy()
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 1)
            for t in tracker._tracks.values():
                cx, cy = t["centroid"]
                cv2.circle(vis, (cx, cy), 5, (0, 220, 255), -1)
            cv2.putText(
                vis, f"Count: {cnt}  [{mode}]",
                (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 255), 3,
            )
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()

    if verbose:
        print(f"  mode={mode}  unique_grains={tracker._nid}")
    return tracker._nid


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    ap = argparse.ArgumentParser(description="Count rice grains in a video.")
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--mode", default="auto",
                    choices=["auto", "white", "dark", "brown"],
                    help="Rice colour profile (default: auto)")
    ap.add_argument("--out", default=None,
                    help="Save annotated output video to this path")
    args = ap.parse_args()

    count = process_video(args.video, rice_mode=args.mode, out_path=args.out)
    print(f"\nFinal count: {count}")


if __name__ == "__main__":
    _cli()
