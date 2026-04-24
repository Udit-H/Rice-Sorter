#!/usr/bin/env python3
"""Batch-test rice grain counting on all training + testing videos.

Runs count_grains_updated.process_video() on every video, compares against
ground-truth counts parsed from filenames, and prints an accuracy table.

Usage:
    python batch_test_all.py
    python batch_test_all.py --save-annotated   # also save annotated output videos
"""

import os
import sys
import csv
import argparse

# Ensure workspace root is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from Count_Rice_Grains_Video.count_grains_updated import process_video

# ── Ground truth ──────────────────────────────────────────────────────────────
# Format: (relative_path, ground_truth_count)

VIDEOS = [
    # Training set
    ("training_vids/test_9_white_100.mp4",    100),   # Note: chalky rice
    ("training_vids/test_10_chalky_100.mp4",  100),   # Note: white rice
    ("training_vids/test_11_brown_100.mp4",   100),
    ("training_vids/test_12_black_41.mp4",     41),
    ("training_vids/test_14_yellow_100.mp4",  100),
    ("training_vids/test_15_brown_200.mp4",   200),
    ("training_vids/test_16_chalky_200.mp4",  200),
    ("training_vids/test_17_yellow_200.mp4",  200),
    ("training_vids/test_18_white_200.mp4",   200),
    ("training_vids/test_19_broken_200.mp4",  200),
    # Test set
    ("test_vids/test_13_mixed.mp4",           168),
    ("test_vids/test_20_mixed_500.mp4",       500),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-annotated", action="store_true",
                    help="Save annotated output videos")
    ap.add_argument("--max-dist", type=int, default=45)
    ap.add_argument("--max-missed", type=int, default=None)
    ap.add_argument("--edge-margin", type=int, default=30)
    ap.add_argument("--no-spike-guard", action="store_true")
    args = ap.parse_args()

    results = []
    header = f"{'Video':<40} {'GT':>4} {'Det':>4} {'Acc%':>7} {'Status'}"
    sep = "-" * 68

    for rel_path, gt in VIDEOS:
        abs_path = os.path.join(ROOT, rel_path)
        if not os.path.exists(abs_path):
            print(f"  [SKIP] {rel_path} not found")
            results.append((rel_path, gt, -1, "MISSING"))
            continue

        print(f"\nProcessing: {rel_path}")

        out_path = None
        if args.save_annotated:
            out_dir = os.path.join(ROOT, "scratch", "annotated_outputs")
            os.makedirs(out_dir, exist_ok=True)
            basename = os.path.splitext(os.path.basename(rel_path))[0]
            out_path = os.path.join(out_dir, f"{basename}_annotated.mp4")

        det = process_video(
            abs_path,
            rice_mode="auto",
            out_path=out_path,
            verbose=True,
            max_dist=args.max_dist,
            max_missed=args.max_missed,
            edge_margin=args.edge_margin,
            spike_guard=not args.no_spike_guard,
        )

        acc = det / gt * 100 if gt > 0 else 0.0
        err = abs(det - gt) / gt * 100
        status = "✅ OK" if err <= 10 else f"❌ {'+' if det > gt else ''}{det - gt}"
        results.append((rel_path, gt, det, status))

    # Print summary table
    print(f"\n{sep}")
    print(header)
    print(sep)
    for rel_path, gt, det, status in results:
        name = os.path.basename(rel_path)
        if det < 0:
            print(f"{name:<40} {gt:>4} {'?':>4} {'---':>7} {status}")
        else:
            acc = det / gt * 100 if gt > 0 else 0.0
            print(f"{name:<40} {gt:>4} {det:>4} {acc:>6.1f}% {status}")
    print(sep)

    # Summary stats
    valid = [(gt, det) for _, gt, det, _ in results if det >= 0]
    if valid:
        errors = [abs(d - g) / g * 100 for g, d in valid]
        print(f"\nMean absolute error : {sum(errors)/len(errors):.1f}%")
        within_10 = sum(1 for e in errors if e <= 10)
        print(f"Within ±10% accuracy: {within_10}/{len(errors)} videos")
        worst = max(errors)
        worst_idx = errors.index(worst)
        worst_name = os.path.basename(results[worst_idx][0])
        print(f"Worst video         : {worst_name} ({worst:.1f}% error)")

    # Save CSV
    csv_path = os.path.join(ROOT, "scratch", "accuracy_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "gt", "detected", "accuracy_pct", "error_pct", "status"])
        for rel_path, gt, det, status in results:
            if det >= 0:
                acc = det / gt * 100 if gt > 0 else 0
                err = abs(det - gt) / gt * 100
                w.writerow([os.path.basename(rel_path), gt, det,
                            f"{acc:.1f}", f"{err:.1f}", status])
    print(f"\nCSV saved: {csv_path}")


if __name__ == "__main__":
    main()
