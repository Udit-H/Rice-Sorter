#!/usr/bin/env python3
"""Train a grain color classifier on crops extracted from `new_vids/`.

This script expects an ImageFolder-style dataset:
  <data-dir>/train/<class>/*.png
  <data-dir>/val/<class>/*.png

Default data-dir matches `build_dataset_from_new_vids.py`.

It saves model checkpoints at epochs 20, 25, 30 (matching the training cadence
you mentioned from earlier runs).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision.datasets import ImageFolder

from torch_rice_model import build_model, get_transforms, save_checkpoint


def _make_weighted_sampler(ds: ImageFolder) -> WeightedRandomSampler:
    # Weight each sample by inverse class frequency.
    counts = np.bincount([y for _, y in ds.samples], minlength=len(ds.classes)).astype(np.float32)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = [float(inv[y]) for _, y in ds.samples]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, List[int], List[int]]:
    model.eval()
    correct = 0
    total = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    return acc, all_labels, all_preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("extracted_grains/new_vids_v1"))
    ap.add_argument("--arch", type=str, default="resnet18", choices=["resnet18"])
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--no-weighted-sampler", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_dir}. "
            f"Run: python build_dataset_from_new_vids.py --out-dir {args.data_dir}"
        )

    tfm_train = get_transforms(img_size=args.img_size, train=True)
    tfm_val = get_transforms(img_size=args.img_size, train=False)

    train_ds = ImageFolder(str(train_dir), transform=tfm_train)
    val_ds = ImageFolder(str(val_dir), transform=tfm_val)

    class_names = train_ds.classes
    print("Classes:", class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(args.arch, num_classes=len(class_names), pretrained=not args.no_pretrained)
    model.to(device)

    if args.no_weighted_sampler:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        sampler = _make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running += float(loss.item())

        scheduler.step()

        val_acc, y_true, y_pred = evaluate(model, val_loader, device)
        train_loss = running / max(len(train_loader), 1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| train_loss={train_loss:.4f} "
            f"| val_acc={val_acc:.4f} "
            f"| lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_acc > best_val:
            best_val = val_acc
            best_path = args.save_dir / "rice_resnet18_new_vids_best.pth"
            save_checkpoint(best_path, arch=args.arch, class_names=class_names, model=model)

        if epoch in (20, 25, 30) or epoch == args.epochs:
            ckpt_path = args.save_dir / f"rice_resnet18_new_vids_e{epoch:02d}.pth"
            save_checkpoint(ckpt_path, arch=args.arch, class_names=class_names, model=model)

            cm = confusion_matrix(y_true, y_pred)
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

    print(f"Best val_acc: {best_val:.4f}")


if __name__ == "__main__":
    main()
