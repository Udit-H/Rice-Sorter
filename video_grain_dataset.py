"""Utilities to build a grain-crop dataset from videos.

This module is intentionally scoped to the current workspace:
- Videos are expected to be in `new_vids/`.
- Grain detection is reused from `Count/count_grains_updated.py`.

Output dataset layout (ImageFolder compatible):
  <out_dir>/train/<label>/*.png
  <out_dir>/val/<label>/*.png

Labels supported: white, brown, chalky, yellow
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np


SUPPORTED_LABELS: Tuple[str, ...] = ("white", "brown", "chalky", "yellow")


@dataclass(frozen=True)
class ExtractConfig:
    out_dir: Path
    crop_size: int = 128
    frame_stride: int = 5
    max_crops_per_video: int = 25000
    max_crops_per_class: int = 40000
    val_every_n_frames: int = 5  # frame_idx % val_every_n_frames == 0 => val
    min_area: int = 55
    max_area: int = 20000
    padding: int = 6
    seed: int = 1337


def parse_label_from_filename(name: str) -> str:
    """Infer label from the video filename.

    Expected: filenames contain one of the supported labels.
    Example: 'test_24_yellow_bunch.mp4' -> 'yellow'
    """
    lower = name.lower()
    for label in SUPPORTED_LABELS:
        if label in lower:
            return label
    raise ValueError(
        f"Unable to infer label from filename: {name}. "
        f"Expected one of: {', '.join(SUPPORTED_LABELS)}"
    )


def iter_videos(videos_dir: Path, exts: Sequence[str] = (".mp4", ".mov", ".mkv", ".avi")) -> Iterator[Path]:
    for p in sorted(videos_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def _ensure_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        for label in SUPPORTED_LABELS:
            (out_dir / split / label).mkdir(parents=True, exist_ok=True)


def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def _resize_square(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def extract_crops_from_frame(
    frame_bgr: np.ndarray,
    *,
    rice_mode: str = "universal",
    min_area: int,
    max_area: int,
    padding: int,
) -> List[Tuple[np.ndarray, Tuple[int, int, int, int], int]]:
    """Return grain crops for one frame.

    Returns list of (crop_bgr, (x, y, w, h), contour_area).
    """
    # Reuse detection from Count/ to avoid divergence.
    from Count.count_grains_updated import detect_grains

    contours, _mask = detect_grains(frame_bgr, rice_mode, min_area=min_area, max_area=max_area)
    crops: List[Tuple[np.ndarray, Tuple[int, int, int, int], int]] = []

    for c in contours:
        area = int(cv2.contourArea(c))
        x, y, w, h = cv2.boundingRect(c)
        x1 = x - padding
        y1 = y - padding
        x2 = x + w + padding
        y2 = y + h + padding
        crop = _safe_crop(frame_bgr, x1, y1, x2, y2)
        if crop is None or crop.size == 0:
            continue
        # Reject extremely tiny crops after padding.
        if min(crop.shape[0], crop.shape[1]) < 12:
            continue
        crops.append((crop, (x, y, w, h), area))

    return crops


def build_grain_dataset_from_videos(
    *,
    videos_dir: Path,
    config: ExtractConfig,
    manifest_csv: Optional[Path] = None,
) -> Dict[str, int]:
    """Extract 128x128 grain crops from videos into ImageFolder layout.

    The split is determined by frame index: every Nth frame goes to val.

    Returns counts per label (train+val combined).
    """
    _ensure_dirs(config.out_dir)

    if manifest_csv is None:
        manifest_csv = config.out_dir / "manifest.csv"

    counts: Dict[str, int] = {lbl: 0 for lbl in SUPPORTED_LABELS}

    with manifest_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "split",
                "label",
                "video",
                "frame_idx",
                "grain_idx",
                "x",
                "y",
                "w",
                "h",
                "area",
            ],
        )
        writer.writeheader()

        for video_path in iter_videos(videos_dir):
            label = parse_label_from_filename(video_path.name)

            if counts[label] >= config.max_crops_per_class:
                continue

            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")

            frame_idx = 0
            processed_frames = 0
            saved_for_video = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                if frame_idx % config.frame_stride != 0:
                    frame_idx += 1
                    continue

                # Important: choose split based on the number of processed frames,
                # not the raw frame index. Otherwise, common settings like
                # frame_stride=5 and val_every_n_frames=5 would put *all* samples
                # into the validation split.
                split = "val" if (processed_frames % config.val_every_n_frames == 0) else "train"

                crops = extract_crops_from_frame(
                    frame,
                    rice_mode="universal",
                    min_area=config.min_area,
                    max_area=config.max_area,
                    padding=config.padding,
                )

                for grain_idx, (crop, (x, y, w, h), area) in enumerate(crops):
                    if saved_for_video >= config.max_crops_per_video:
                        break
                    if counts[label] >= config.max_crops_per_class:
                        break

                    crop_sq = _resize_square(crop, config.crop_size)

                    out_name = (
                        f"{video_path.stem}_f{frame_idx:06d}_g{grain_idx:03d}.png"
                    )
                    out_path = config.out_dir / split / label / out_name
                    cv2.imwrite(str(out_path), crop_sq)

                    writer.writerow(
                        {
                            "path": str(out_path),
                            "split": split,
                            "label": label,
                            "video": video_path.name,
                            "frame_idx": frame_idx,
                            "grain_idx": grain_idx,
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "area": area,
                        }
                    )

                    counts[label] += 1
                    saved_for_video += 1

                if saved_for_video >= config.max_crops_per_video:
                    break
                if counts[label] >= config.max_crops_per_class:
                    break

                processed_frames += 1
                frame_idx += 1

            cap.release()

    return counts
