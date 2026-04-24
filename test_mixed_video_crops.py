#!/usr/bin/env python3
"""Evaluate the trained 5-class model on the mixed test video.

Process
- Extract one crop per uniquely-counted grain from the mixed video using
  Count_Rice_Grains_Video/count_grains_updated.py (universal mode).
- Run the trained EfficientNet-B0 color classifier on each crop.
- If max softmax confidence is below a threshold, label as "broken" (optional).

Outputs
- mixed_predictions.csv (per-crop: predicted label, confidence, probs)
- Prints a table comparing predicted counts to provided GT.

GT provided by user for mixed video (Total 168):
Black 20, Brown 36, Chalky 23, White 34, Yellow 42, Broken 13

Note
- The counter may extract fewer than 168 crops (detection coverage). We report both
  extracted count and the GT total.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

from Count_Rice_Grains_Video import count_grains_updated as cgu


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CLASSES = ["black", "brown", "chalky", "white", "yellow"]

GT = {
    "black": 20,
    "brown": 36,
    "chalky": 23,
    "white": 34,
    "yellow": 42,
    "broken": 13,
    "total": 168,
}


class ExtractingUniqueGrainTracker:
    def __init__(self, max_dist: int, max_missed: int):
        self.max_dist = max_dist
        self.max_missed = max_missed
        self._tracks: Dict[int, Dict[str, object]] = {}
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
        for tid, di in matched_t.items():
            self._tracks[tid]["centroid"] = centroids[di]
            self._tracks[tid]["missed"] = 0

        for tid in list(self._tracks):
            if tid not in matched_t:
                self._tracks[tid]["missed"] = int(self._tracks[tid]["missed"]) + 1

        new_tracks: List[Tuple[int, Tuple[int, int]]] = []
        for di, c in enumerate(centroids):
            if di not in matched_d:
                new_id = self._nid
                self._tracks[new_id] = {"centroid": c, "missed": 0}
                self._nid += 1
                new_tracks.append((new_id, c))

        self._tracks = {k: v for k, v in self._tracks.items() if int(v["missed"]) <= self.max_missed}
        return new_tracks

    @property
    def total_unique(self) -> int:
        return self._nid


def crop_around(frame_bgr: np.ndarray, cx: int, cy: int, crop_size: int) -> np.ndarray:
    half = crop_size // 2
    h, w = frame_bgr.shape[:2]
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, cx + half)
    y2 = min(h, cy + half)
    crop = frame_bgr[y1:y2, x1:x2]

    if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
        padded = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        padded[:, :] = (255, 0, 0)
        padded[0 : crop.shape[0], 0 : crop.shape[1]] = crop
        crop = padded

    return crop


def build_model(num_classes: int = 5) -> models.EfficientNet:
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features  # type: ignore[index]
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Test model on mixed video by extracting unique crops.")
    ap.add_argument("--video", default="test_vids/test_13_mixed.mp4")
    ap.add_argument("--weights", default="rice_model.pth")
    ap.add_argument("--crop-size", type=int, default=96)
    ap.add_argument("--min-area", type=int, default=40)
    ap.add_argument("--max-dist", type=int, default=70, help="Tracker match radius in pixels (use 70 to match count_grains_updated)")
    ap.add_argument("--confidence-threshold", type=float, default=0.60, help="below this -> broken")
    ap.add_argument("--out-csv", default="mixed_predictions.csv")
    ap.add_argument("--max-grains", type=int, default=None)
    args = ap.parse_args(list(argv) if argv is not None else None)

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"Weights not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_missed = max(5, int(fps * 0.33))
    tracker = ExtractingUniqueGrainTracker(max_dist=args.max_dist, max_missed=max_missed)

    softmax = nn.Softmax(dim=1)

    pred_counts: Dict[str, int] = {k: 0 for k in [*CLASSES, "broken"]}

    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["grain_id", "frame", "pred", "confidence", *[f"p_{c}" for c in CLASSES]])

        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            contours, mask = cgu.detect_grains(frame, mode="universal", min_area=args.min_area, max_area=20000)
            centroids = cgu.centroids_of(contours)
            new_tracks = tracker.update_and_get_new(centroids)

            for grain_id, (cx, cy) in new_tracks:
                crop = crop_around(frame, int(cx), int(cy), args.crop_size)

                x = tfm(crop).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x)
                    probs = softmax(logits).cpu().numpy()[0]

                pred_idx = int(np.argmax(probs))
                conf = float(np.max(probs))

                if conf < args.confidence_threshold:
                    pred = "broken"
                else:
                    pred = CLASSES[pred_idx]

                pred_counts[pred] += 1

                w.writerow([grain_id, frame_idx, pred, conf, *[float(p) for p in probs.tolist()]])

                if args.max_grains is not None and sum(pred_counts.values()) >= args.max_grains:
                    cap.release()
                    break

            if args.max_grains is not None and sum(pred_counts.values()) >= args.max_grains:
                break

    cap.release()

    extracted_total = sum(pred_counts.values())

    print("\nMIXED VIDEO EVAL")
    print("=" * 56)
    print(f"Video: {video_path.name}")
    print(f"Extracted unique crops: {extracted_total}")
    print(f"GT total (provided): {GT['total']}")
    print(f"Confidence threshold -> broken: {args.confidence_threshold:.2f}")
    print("-")

    header = f"{'Class':<8} {'GT':>6} {'Pred':>6} {'Pred/GT%':>10}"
    print(header)
    print("-" * len(header))

    for cls in ["black", "brown", "chalky", "white", "yellow", "broken"]:
        gt = GT[cls]
        pred = pred_counts.get(cls, 0)
        pct = (pred / gt * 100.0) if gt else 0.0
        print(f"{cls:<8} {gt:>6} {pred:>6} {pct:>9.2f}%")

    print(f"\nPred counts sum: {extracted_total}")
    print(f"Saved: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
