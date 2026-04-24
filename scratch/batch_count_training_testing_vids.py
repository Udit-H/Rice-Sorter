#!/usr/bin/env python3
"""Batch-count rice grains on videos in training/testing folders.

Uses the centroid-tracking unique-ID counter from:
  Count_Rice_Grains_Video/count_grains_updated.py

Outputs:
  - scratch/training_testing_counts.json
  - scratch/training_testing_counts.csv

Folders:
  - training_vids/
  - testing_vids/ (if present) else test_vids/

This script is intentionally lightweight and deterministic.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = Path(__file__).resolve().parent / "training_testing_counts.json"
OUT_CSV = Path(__file__).resolve().parent / "training_testing_counts.csv"

# Make workspace root importable when executing from scratch/
sys.path.insert(0, str(ROOT))

from Count_Rice_Grains_Video import count_grains_updated as counter


@dataclass
class VideoCount:
    split: str
    video: str
    path: str
    count: int


def _iter_videos(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts])


def main() -> int:
    training_dir = ROOT / "training_vids"
    testing_dir = ROOT / "testing_vids"
    if not testing_dir.exists():
        testing_dir = ROOT / "test_vids"

    jobs: list[tuple[str, Path]] = []
    if training_dir.exists():
        jobs += [("training", p) for p in _iter_videos(training_dir)]
    if testing_dir.exists():
        jobs += [("testing", p) for p in _iter_videos(testing_dir)]

    if not jobs:
        print("No videos found in training/testing folders.")
        print(f"Checked: {training_dir} and {testing_dir}")
        return 2

    results: list[VideoCount] = []

    for split, video_path in jobs:
        print(f"Counting {split}: {video_path.name} ...")
        count = counter.process_video(str(video_path), rice_mode="auto", out_path=None, verbose=False)
        results.append(
            VideoCount(
                split=split,
                video=video_path.name,
                path=str(video_path.relative_to(ROOT)),
                count=int(count),
            )
        )
        print(f"  -> {count}")

    OUT_JSON.write_text(json.dumps([asdict(r) for r in results], indent=2))

    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "video", "path", "count"])
        w.writeheader()
        for r in results:
            w.writerow(asdict(r))

    print("\nSummary")
    for split in ("training", "testing"):
        subset = [r for r in results if r.split == split]
        if not subset:
            continue
        print(f"- {split}: {len(subset)} videos")

    print(f"\nWrote: {OUT_JSON}")
    print(f"Wrote: {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
