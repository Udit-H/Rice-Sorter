#!/usr/bin/env python3
"""End-to-end test runner for the `new_vids/` workflow.

What it does:
1) (Optional) Extract grain crops (128x128) from videos in `new_vids/` using `Count/`.
2) Evaluate one or more classifiers on the validation split.

Supported models:
- `rice_classifier_v4.pkl` (hierarchical sklearn model in `rice_classifier_v4.py`)
- Torch checkpoint from `train_grain_classifier_new_vids.py`

This file replaces the older ad-hoc test scripts with a single, reproducible entrypoint.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


@dataclass(frozen=True)
class ManifestRow:
    path: Path
    split: str
    label: str
    video: str


def _read_manifest(path: Path) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                ManifestRow(
                    path=Path(r["path"]),
                    split=r["split"],
                    label=r["label"],
                    video=r.get("video", ""),
                )
            )
    return rows


def _labels_to_indices(labels: Sequence[str]) -> Dict[str, int]:
    return {lbl: i for i, lbl in enumerate(sorted(set(labels)))}


def _print_metrics(title: str, y_true: List[int], y_pred: List[int], class_names: List[str]) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print(f"\n=== {title} ===")
    print("Confusion matrix:\n", cm)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )


def eval_rice_classifier_v4(rows: List[ManifestRow], *, pkl_path: Path) -> None:
    import joblib

    # Backward-compat: some historical `rice_classifier_v4.pkl` files were
    # produced by running `rice_classifier_v4.py` as a script, which pickles the
    # class as `__main__.RiceClassifier`. When we load from this runner, the
    # `__main__` module is `run_all_tests.py`, so we provide the symbol.
    import sys
    import rice_classifier_v4 as rc4
    setattr(sys.modules.get("__main__"), "RiceClassifier", rc4.RiceClassifier)

    clf = joblib.load(str(pkl_path))

    class_names = ["brown", "chalky", "white", "yellow"]
    idx = {c: i for i, c in enumerate(class_names)}

    y_true: List[int] = []
    y_pred: List[int] = []

    per_video_true: Dict[str, List[int]] = defaultdict(list)
    per_video_pred: Dict[str, List[int]] = defaultdict(list)

    for r in rows:
        if r.split != "val":
            continue
        pred = clf.predict_path(str(r.path))["class"]
        if r.label not in idx or pred not in idx:
            continue
        yt = idx[r.label]
        yp = idx[pred]
        y_true.append(yt)
        y_pred.append(yp)
        per_video_true[r.video].append(yt)
        per_video_pred[r.video].append(yp)

    _print_metrics(f"rice_classifier_v4 ({pkl_path})", y_true, y_pred, class_names)

    # Majority vote per video
    v_true: List[int] = []
    v_pred: List[int] = []
    for video in sorted(per_video_true):
        vt = Counter(per_video_true[video]).most_common(1)[0][0]
        vp = Counter(per_video_pred[video]).most_common(1)[0][0]
        v_true.append(vt)
        v_pred.append(vp)

    if v_true:
        _print_metrics("rice_classifier_v4 (per-video majority)", v_true, v_pred, class_names)


def eval_torch_checkpoint(rows: List[ManifestRow], *, ckpt_path: Path, img_size: int = 128, batch_size: int = 128) -> None:
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    from torch_rice_model import get_transforms, load_checkpoint

    class ManifestDataset(Dataset):
        def __init__(self, rows: List[ManifestRow], class_to_idx: Dict[str, int]):
            self.rows = [r for r in rows if r.split == "val" and r.label in class_to_idx]
            self.class_to_idx = class_to_idx
            self.tfm = get_transforms(img_size=img_size, train=False)

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, i: int):
            r = self.rows[i]
            img = Image.open(r.path).convert("RGB")
            x = self.tfm(img)
            y = self.class_to_idx[r.label]
            return x, y, r.video

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(ckpt_path, device)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    ds = ManifestDataset(rows, class_to_idx=class_to_idx)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    y_true: List[int] = []
    y_pred: List[int] = []
    per_video_true: Dict[str, List[int]] = defaultdict(list)
    per_video_pred: Dict[str, List[int]] = defaultdict(list)

    model.eval()
    with torch.no_grad():
        for xb, yb, videos in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            yb = yb.numpy().tolist()
            y_true.extend(yb)
            y_pred.extend(preds)
            for yt, yp, v in zip(yb, preds, videos):
                per_video_true[str(v)].append(int(yt))
                per_video_pred[str(v)].append(int(yp))

    _print_metrics(f"torch ({ckpt_path})", y_true, y_pred, class_names)

    v_true: List[int] = []
    v_pred: List[int] = []
    for video in sorted(per_video_true):
        vt = Counter(per_video_true[video]).most_common(1)[0][0]
        vp = Counter(per_video_pred[video]).most_common(1)[0][0]
        v_true.append(vt)
        v_pred.append(vp)

    if v_true:
        _print_metrics("torch (per-video majority)", v_true, v_pred, class_names)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", type=Path, default=Path("new_vids"))
    ap.add_argument("--extract-dir", type=Path, default=Path("extracted_grains/new_vids_v1"))
    ap.add_argument("--rebuild-extract", action="store_true")
    ap.add_argument("--frame-stride", type=int, default=5)
    ap.add_argument("--val-every-n-frames", type=int, default=5)
    ap.add_argument("--max-crops-per-video", type=int, default=25000)
    ap.add_argument("--max-crops-per-class", type=int, default=40000)

    ap.add_argument("--eval-v4", action="store_true")
    ap.add_argument("--v4-pkl", type=Path, default=Path("rice_classifier_v4.pkl"))

    ap.add_argument("--eval-torch", action="store_true")
    ap.add_argument("--torch-ckpt", type=Path, default=Path("models/rice_resnet18_new_vids_best.pth"))
    ap.add_argument("--torch-img-size", type=int, default=128)

    args = ap.parse_args()

    manifest = args.extract_dir / "manifest.csv"
    if args.rebuild_extract or not manifest.exists():
        from video_grain_dataset import ExtractConfig, build_grain_dataset_from_videos

        cfg = ExtractConfig(
            out_dir=args.extract_dir,
            crop_size=128,
            frame_stride=args.frame_stride,
            val_every_n_frames=args.val_every_n_frames,
            max_crops_per_video=args.max_crops_per_video,
            max_crops_per_class=args.max_crops_per_class,
        )
        print("Building crop dataset from videos...")
        counts = build_grain_dataset_from_videos(videos_dir=args.videos_dir, config=cfg, manifest_csv=manifest)
        print("Counts:", counts)

    rows = _read_manifest(manifest)
    val_rows = [r for r in rows if r.split == "val"]
    print(f"Manifest: {manifest}")
    print(f"Val samples: {len(val_rows)}")
    print("Val class counts:", Counter(r.label for r in val_rows))

    # If user didn't specify, evaluate everything that exists.
    if not args.eval_v4 and not args.eval_torch:
        args.eval_v4 = args.v4_pkl.exists()
        args.eval_torch = args.torch_ckpt.exists()

    if args.eval_v4:
        if not args.v4_pkl.exists():
            raise FileNotFoundError(args.v4_pkl)
        eval_rice_classifier_v4(rows, pkl_path=args.v4_pkl)

    if args.eval_torch:
        if not args.torch_ckpt.exists():
            raise FileNotFoundError(args.torch_ckpt)
        eval_torch_checkpoint(rows, ckpt_path=args.torch_ckpt, img_size=args.torch_img_size)


if __name__ == "__main__":
    main()
