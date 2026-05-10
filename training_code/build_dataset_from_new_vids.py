#!/usr/bin/env python3
"""Build a grain-crop dataset from videos in `new_vids/`.

This uses the grain detector from `Count/count_grains_updated.py` and saves
128x128 crops in an ImageFolder-compatible layout.

Example:
  python build_dataset_from_new_vids.py \
    --out-dir extracted_grains/new_vids_v1 \
    --frame-stride 5 --val-every-n-frames 5 \
    --max-crops-per-video 25000 --max-crops-per-class 40000
"""

from __future__ import annotations

import argparse
from pathlib import Path

from video_grain_dataset import ExtractConfig, build_grain_dataset_from_videos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, default=Path("new_vids"))
    ap.add_argument("--out-dir", type=Path, default=Path("extracted_grains/new_vids_v1"))
    ap.add_argument("--crop-size", type=int, default=128)
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--val-every-n-frames", type=int, default=5)
    ap.add_argument("--max-crops-per-video", type=int, default=25000)
    ap.add_argument("--max-crops-per-class", type=int, default=40000)
    ap.add_argument("--min-area", type=int, default=55)
    ap.add_argument("--max-area", type=int, default=20000)
    ap.add_argument("--padding", type=int, default=6)
    args = ap.parse_args()

    cfg = ExtractConfig(
        out_dir=args.out_dir,
        crop_size=args.crop_size,
        frame_stride=args.frame_stride,
        val_every_n_frames=args.val_every_n_frames,
        max_crops_per_video=args.max_crops_per_video,
        max_crops_per_class=args.max_crops_per_class,
        min_area=args.min_area,
        max_area=args.max_area,
        padding=args.padding,
    )

    counts = build_grain_dataset_from_videos(
        videos_dir=args.videos_dir,
        config=cfg,
        manifest_csv=args.out_dir / "manifest.csv",
    )

    total = sum(counts.values())
    print("\nDone.")
    print(f"Output: {args.out_dir}")
    print(f"Total crops: {total}")
    for k, v in counts.items():
        print(f"  {k:7}: {v}")


if __name__ == "__main__":
    main()
