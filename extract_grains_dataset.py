#!/usr/bin/env python3
"""Extract counted grain crops from videos into a training/testing dataset.

What it does
- Uses the *updated* counting pipeline from Count_Rice_Grains_Video/count_grains_updated.py
  (universal belt-subtraction mask + centroid splitting + unique-ID tracking).
- Saves exactly one image per uniquely-counted grain (new track ID) by cropping around
  the centroid at the moment the track is created.
- Writes crops into an output dataset layout:

  <out_dir>/training/<class_name>/*.png
  <out_dir>/testing/<class_name>/*.png

Classes
- Creates 5 class folders: chalky, white, black, brown, yellow.
- For videos whose filenames include one of these class tokens, all extracted grains
  from that video are placed into that class folder.
- For "mixed" videos, a lightweight HSV heuristic is used per-crop to assign one of
  the 5 classes (best-effort).

Notes
- This script does NOT require TensorFlow.
- It is intended to generate an initial dataset quickly; if you later want higher
  fidelity labeling (especially for mixed videos), we can plug in your ML classifier.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Reuse the updated detection/splitting logic
from Count_Rice_Grains_Video import count_grains_updated as cgu


CLASS_NAMES: Tuple[str, ...] = ("chalky", "white", "black", "brown", "yellow")


@dataclass(frozen=True)
class VideoJob:
    split: str  # "training" or "testing"
    video_path: Path


class ExtractingUniqueGrainTracker:
    """Unique-ID centroid tracker that also reports newly created track IDs."""

    def __init__(self, max_dist: int, max_missed: int):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self._tracks: Dict[int, Dict[str, object]] = {}  # id -> {centroid, missed}
        self._nid = 0

    def _match(self, dets: List[Tuple[int, int]]):
        matched_t: Dict[int, int] = {}
        matched_d: set[int] = set()

        for tid, t in self._tracks.items():
            tx, ty = t["centroid"]  # type: ignore[misc]
            best_di = None
            best_d = self.max_dist
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

    def update_and_get_new(self, centroids: List[Tuple[int, int]]) -> List[Tuple[int, Tuple[int, int]]]:
        matched_t, matched_d = self._match(centroids)

        # Update matched tracks
        for tid, di in matched_t.items():
            self._tracks[tid]["centroid"] = centroids[di]
            self._tracks[tid]["missed"] = 0

        # Penalize unmatched existing tracks
        for tid in list(self._tracks):
            if tid not in matched_t:
                self._tracks[tid]["missed"] = int(self._tracks[tid]["missed"]) + 1

        new_tracks: List[Tuple[int, Tuple[int, int]]] = []

        # Register new tracks (new unique grains)
        for di, c in enumerate(centroids):
            if di not in matched_d:
                new_id = self._nid
                self._tracks[new_id] = {"centroid": c, "missed": 0}
                self._nid += 1
                new_tracks.append((new_id, c))

        # Prune dead tracks
        self._tracks = {
            k: v for k, v in self._tracks.items() if int(v["missed"]) <= self.max_missed
        }

        return new_tracks

    @property
    def total_unique(self) -> int:
        return self._nid


def _ensure_dirs(out_dir: Path, splits: Sequence[str]) -> None:
    for split in splits:
        for cls in CLASS_NAMES:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)


def _infer_video_label(video_name: str) -> Optional[str]:
    lower = video_name.lower()
    for cls in CLASS_NAMES:
        if cls in lower:
            return cls
    if "mixed" in lower:
        return None
    return None


def _crop_around(frame_bgr: np.ndarray, cx: int, cy: int, crop_size: int) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    half = crop_size // 2
    h, w = frame_bgr.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)

    crop = frame_bgr[y1:y2, x1:x2]

    # Pad to fixed size if we hit boundaries
    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        # Fill with the dominant belt-like blue for nicer visuals
        padded[:, :] = (255, 0, 0)
        padded[0 : crop.shape[0], 0 : crop.shape[1]] = crop
        crop = padded

    return crop, (x1, y1, x2, y2)


def _classify_crop_hsv(crop_bgr: np.ndarray, crop_mask: Optional[np.ndarray] = None) -> str:
    """Best-effort 5-way classification using HSV statistics.

    This is primarily used for "mixed" videos.
    """

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

    if crop_mask is not None and crop_mask.shape[:2] == crop_bgr.shape[:2]:
        fg = crop_mask > 0
        if int(np.sum(fg)) > 50:
            h = hsv[:, :, 0][fg].astype(np.float32)
            s = hsv[:, :, 1][fg].astype(np.float32)
            v = hsv[:, :, 2][fg].astype(np.float32)
        else:
            h = hsv[:, :, 0].astype(np.float32).ravel()
            s = hsv[:, :, 1].astype(np.float32).ravel()
            v = hsv[:, :, 2].astype(np.float32).ravel()
    else:
        h = hsv[:, :, 0].astype(np.float32).ravel()
        s = hsv[:, :, 1].astype(np.float32).ravel()
        v = hsv[:, :, 2].astype(np.float32).ravel()

    h_mean = float(np.mean(h))
    s_mean = float(np.mean(s))
    v_mean = float(np.mean(v))
    v_std = float(np.std(v))

    # Black: low luminance foreground
    if v_mean < 75:
        return "black"

    # Yellow/Brown: higher saturation and warm hues
    # OpenCV Hue is 0..179 where ~30 is yellow.
    if s_mean > 45:
        if 15 <= h_mean <= 45 and v_mean > 110:
            return "yellow"
        if 5 <= h_mean <= 35:
            return "brown"

    # White vs chalky: both low saturation; chalky tends to have more texture/variance
    if s_mean < 35:
        # Higher variance or slightly dimmer tends to be chalky
        if v_std > 28 or v_mean < 165:
            return "chalky"
        return "white"

    # Fallbacks
    if v_mean > 170:
        return "white"
    return "chalky"


def extract_from_video(
    job: VideoJob,
    out_dir: Path,
    crop_size: int,
    max_grains: Optional[int],
    write_debug_overlays: bool,
) -> Dict[str, int]:
    """Extract one crop per counted grain into split/class folders."""

    cap = cv2.VideoCapture(str(job.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {job.video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_missed = max(5, int(fps * 0.33))
    tracker = ExtractingUniqueGrainTracker(max_dist=80, max_missed=max_missed)

    video_label = _infer_video_label(job.video_path.name)

    debug_writer = None
    if write_debug_overlays:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        dbg_path = out_dir / job.split / f"_debug_{job.video_path.stem}.mp4"
        debug_writer = cv2.VideoWriter(str(dbg_path), fourcc, fps, (w, h))

    counts: Dict[str, int] = {cls: 0 for cls in CLASS_NAMES}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        contours, mask = cgu.detect_grains(frame, mode="universal", min_area=40, max_area=20000)
        centroids = cgu.centroids_of(contours)
        new_tracks = tracker.update_and_get_new(centroids)

        if debug_writer is not None:
            vis = frame.copy()
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 220, 0), 1)
            for tid, t in tracker._tracks.items():
                cx, cy = t["centroid"]  # type: ignore[misc]
                cv2.circle(vis, (int(cx), int(cy)), 5, (0, 220, 255), -1)
            cv2.putText(
                vis,
                f"Unique: {tracker.total_unique}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 255),
                2,
            )
            debug_writer.write(vis)

        for grain_id, (cx, cy) in new_tracks:
            crop, (x1, y1, x2, y2) = _crop_around(frame, int(cx), int(cy), crop_size)

            # For mixed videos, use local mask region to help HSV classification
            crop_mask = None
            if video_label is None:
                crop_mask = mask[y1:y2, x1:x2]
                if crop_mask.shape[0] != crop_size or crop_mask.shape[1] != crop_size:
                    padded = np.zeros((crop_size, crop_size), dtype=np.uint8)
                    padded[0 : crop_mask.shape[0], 0 : crop_mask.shape[1]] = crop_mask
                    crop_mask = padded

            cls = video_label or _classify_crop_hsv(crop, crop_mask)
            if cls not in counts:
                continue

            out_path = out_dir / job.split / cls / f"{job.video_path.stem}__id{grain_id:06d}__f{frame_idx:06d}.png"
            cv2.imwrite(str(out_path), crop)
            counts[cls] += 1

            if max_grains is not None and sum(counts.values()) >= max_grains:
                break

        if max_grains is not None and sum(counts.values()) >= max_grains:
            break

    cap.release()
    if debug_writer is not None:
        debug_writer.release()

    return counts


def build_jobs(training_dir: Path, testing_dir: Path) -> List[VideoJob]:
    jobs: List[VideoJob] = []
    for p in sorted(training_dir.glob("*.mp4")):
        jobs.append(VideoJob(split="training", video_path=p))
    for p in sorted(testing_dir.glob("*.mp4")):
        jobs.append(VideoJob(split="testing", video_path=p))
    return jobs


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Extract unique counted grain crops into class folders.")
    ap.add_argument("--training-dir", default="training_vids", help="Folder containing training videos")
    ap.add_argument("--testing-dir", default="test_vids", help="Folder containing testing videos")
    ap.add_argument("--out-dir", default="extracted_grains", help="Output dataset folder")
    ap.add_argument("--crop-size", type=int, default=96, help="Crop size (square) around centroid")
    ap.add_argument("--max-grains", type=int, default=None, help="Optional cap on total grains per video")
    ap.add_argument("--debug-overlays", action="store_true", help="Write debug overlay videos to output")
    args = ap.parse_args(list(argv) if argv is not None else None)

    training_dir = Path(args.training_dir)
    testing_dir = Path(args.testing_dir)
    out_dir = Path(args.out_dir)

    if not training_dir.exists():
        raise SystemExit(f"Training dir not found: {training_dir}")
    if not testing_dir.exists():
        raise SystemExit(f"Testing dir not found: {testing_dir}")

    _ensure_dirs(out_dir, splits=("training", "testing"))

    jobs = build_jobs(training_dir, testing_dir)
    if not jobs:
        print("No .mp4 videos found.")
        return 0

    summary_rows: List[Dict[str, object]] = []

    for job in jobs:
        print(f"\nProcessing {job.split}: {job.video_path.name}")
        counts = extract_from_video(
            job=job,
            out_dir=out_dir,
            crop_size=args.crop_size,
            max_grains=args.max_grains,
            write_debug_overlays=bool(args.debug_overlays),
        )
        total = sum(counts.values())
        print(f"  Saved {total} grain crops → {out_dir/job.split}")

        row: Dict[str, object] = {
            "split": job.split,
            "video": job.video_path.name,
            "total": total,
        }
        row.update({cls: counts[cls] for cls in CLASS_NAMES})
        summary_rows.append(row)

    csv_path = out_dir / "extraction_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "video", "total", *CLASS_NAMES],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSummary written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
