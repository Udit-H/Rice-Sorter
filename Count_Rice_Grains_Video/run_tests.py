#!/usr/bin/env python3
"""
Test the rice-grain counter on all labelled videos and print an accuracy table.

Ground-truth counts come from the reference image provided with this project.
"""

import os
import sys

# Allow importing count_grains from the same folder
sys.path.insert(0, os.path.dirname(__file__))
from count_grains import process_video

VIDEOS_DIR = os.path.join(os.path.dirname(__file__), "..", "videos")

# Ground-truth grain counts (from reference image)
GT: dict = {
    "test_1_50.mp4":            50,
    "test_2_50.mp4":            50,
    "test_3_100.mp4":          100,
    "test_4_100.mp4":          100,
    "test_5_100.mp4":          100,
    "test_6_100.mp4":          100,
    "test_7_200.mp4":          200,
    "test_7_200_original.mp4": 200,
    "black_rice.mp4":           54,
    "test_set_1.mp4":          100,
}

# Explicit rice-type overrides (skip auto-detect for known types)
OVERRIDES: dict = {
    "black_rice.mp4":           "dark",
    "test_7_200_original.mp4":  "white",
}

# Optional: save annotated output videos alongside this script
SAVE_OUTPUT = False


def main():
    header = f"{'Video':<35} {'GT':>5} {'Det':>5} {'Acc%':>7} {'Mode'}"
    sep    = "-" * 60
    print(sep)
    print(header)
    print(sep)

    rows = []
    for fname in sorted(GT):
        gt    = GT[fname]
        vpath = os.path.join(VIDEOS_DIR, fname)

        if not os.path.exists(vpath):
            print(f"  [SKIP] {fname} not found")
            rows.append((fname, gt, -1, "MISSING", ""))
            continue

        mode    = OVERRIDES.get(fname, "auto")
        out_vid = None
        if SAVE_OUTPUT:
            out_vid = os.path.join(
                os.path.dirname(__file__),
                "output_" + os.path.splitext(fname)[0] + ".mp4",
            )

        print(f"\nProcessing: {fname}")
        det = process_video(vpath, rice_mode=mode, out_path=out_vid, verbose=True)

        acc = det / gt * 100 if gt > 0 else 0.0
        rows.append((fname, gt, det, f"{acc:.1f}%", mode))

    print()
    print(sep)
    print(header)
    print(sep)
    for fname, gt, det, acc, mode in rows:
        flag = " OK" if det >= 0 and abs(det - gt) / max(gt, 1) < 0.10 else ""
        print(f"{fname:<35} {gt:>5} {det:>5} {acc:>7}{flag}")
    print(sep)

    # Summary stats
    valid = [(gt, det) for _, gt, det, _, _ in rows if det >= 0]
    if valid:
        errors = [abs(d - g) / g * 100 for g, d in valid]
        print(f"\nMean absolute error: {sum(errors)/len(errors):.1f}%")
        within_10 = sum(1 for e in errors if e <= 10)
        print(f"Within ±10% accuracy: {within_10}/{len(errors)} videos")


if __name__ == "__main__":
    main()
