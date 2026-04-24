#!/usr/bin/env python3
"""Train a 5-class rice color classifier (PyTorch) per spec.

Spec highlights implemented
- Dataset root contains exactly: black/, brown/, chalky/, white/, yellow/
- ImageFolder label ordering is alphabetical.
- 80/20 random_split, and val uses val transforms (not train transforms).
- Class weights computed from training subset and passed to CrossEntropyLoss.
- EfficientNet-B0 pretrained; phase-1 trains head only; phase-2 unfreezes last 2 blocks.
- Saves best checkpoint by val accuracy: rice_model.pth (state_dict only)
- Saves training_plot.png and val_predictions.csv (per-grain softmax vector + labels)

Outputs
- rice_model.pth
- training_plot.png
- val_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class History:
    train_loss: List[float]
    train_acc: List[float]
    val_loss: List[float]
    val_acc: List[float]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    return train_tf, val_tf


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(dataset_dir: Path, seed: int) -> Tuple[Subset, Subset, List[str]]:
    train_tf, val_tf = build_transforms()

    # Two dataset objects so val subset can use val transforms.
    base_train = datasets.ImageFolder(str(dataset_dir), transform=train_tf)
    base_val = datasets.ImageFolder(str(dataset_dir), transform=val_tf)

    class_names = base_train.classes

    n_total = len(base_train)
    n_val = int(round(0.2 * n_total))
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(base_train, [n_train, n_val], generator=generator)

    # Rebuild val subset with same indices but val transforms.
    val_indices = list(val_subset.indices)  # type: ignore[attr-defined]
    val_subset = Subset(base_val, val_indices)

    return train_subset, val_subset, class_names


def compute_class_weights(train_subset: Subset, num_classes: int) -> torch.Tensor:
    # train_subset.dataset is ImageFolder
    ds: datasets.ImageFolder = train_subset.dataset  # type: ignore[assignment]

    # ds.targets aligns with ds.samples
    targets = np.array(ds.targets)
    indices = np.array(train_subset.indices)  # type: ignore[attr-defined]
    train_targets = targets[indices]

    counts = np.bincount(train_targets, minlength=num_classes).astype(np.int64)
    total = int(counts.sum())

    weights = []
    for c in counts:
        if c > 0:
            weights.append(total / (num_classes * int(c)))
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.float32)


def build_model(num_classes: int) -> models.EfficientNet:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    # Freeze whole backbone
    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features  # type: ignore[index]

    head = [
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
    ]

    # Extra block for 5+ classes
    if num_classes >= 5:
        head += [
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        ]
    else:
        head += [nn.Linear(256, num_classes)]

    model.classifier = nn.Sequential(*head)

    # Head is trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    return model


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        losses.append(float(loss.item()))
        accs.append(accuracy_from_logits(logits, y))

    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


def train_one_epoch(model: nn.Module, loader: DataLoader, device: torch.device, loss_fn: nn.Module, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    accs: List[float] = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))
        accs.append(accuracy_from_logits(logits, y))

    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


def unfreeze_last_two_blocks(model: models.EfficientNet) -> None:
    # Unfreeze last two MBConv blocks by name: features[-1] and features[-2]
    for p in model.features[-1].parameters():
        p.requires_grad = True
    for p in model.features[-2].parameters():
        p.requires_grad = True


def save_training_plot(history: History, out_path: Path, phase_boundary_epoch: int) -> None:
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.train_loss) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history.train_loss, label="train loss")
    ax1.plot(epochs, history.val_loss, label="val loss")
    ax1.axvline(x=phase_boundary_epoch, color="gray", linestyle="--", label="phase boundary")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history.train_acc, label="train acc")
    ax2.plot(epochs, history.val_acc, label="val acc")
    ax2.axvline(x=phase_boundary_epoch, color="gray", linestyle="--", label="phase boundary")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


@torch.no_grad()
def save_val_predictions_csv(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    out_path: Path,
    device: torch.device,
) -> None:
    model.eval()

    header = ["true_label", "pred_label", "confidence", *[f"p_{c}" for c in class_names]]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)

        softmax = nn.Softmax(dim=1)
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = softmax(logits).cpu().numpy()
            y_np = y.numpy()

            preds = np.argmax(probs, axis=1)
            confs = np.max(probs, axis=1)

            for i in range(len(y_np)):
                w.writerow(
                    [
                        class_names[int(y_np[i])],
                        class_names[int(preds[i])],
                        float(confs[i]),
                        *[float(p) for p in probs[i].tolist()],
                    ]
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train rice color classifier (EfficientNet-B0).")
    ap.add_argument("--dataset", default="extracted_grains/training", help="Dataset root (contains class subfolders)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs-phase1", type=int, default=10)
    ap.add_argument("--epochs-phase2", type=int, default=10)
    ap.add_argument("--lr-phase1", type=float, default=1e-3)
    ap.add_argument("--lr-phase2", type=float, default=1e-5)
    ap.add_argument("--out-model", default="rice_model.pth")
    ap.add_argument("--out-plot", default="training_plot.png")
    ap.add_argument("--out-val-csv", default="val_predictions.csv")
    args = ap.parse_args(list(argv) if argv is not None else None)

    set_seed(args.seed)

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Dataset not found: {dataset_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds, val_ds, class_names = build_datasets(dataset_dir, seed=args.seed)

    # Enforce expected class order (alphabetical via ImageFolder)
    expected = ["black", "brown", "chalky", "white", "yellow"]
    if class_names != expected:
        print(f"[WARN] ImageFolder classes are {class_names} (expected {expected}).")

    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Train samples: {len(train_ds)}  Val samples: {len(val_ds)}")

    class_weights = compute_class_weights(train_ds, num_classes=num_classes).to(device)
    print(f"Class weights: {class_weights.detach().cpu().numpy().round(3).tolist()}")

    model = build_model(num_classes=num_classes).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
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

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    history = History(train_loss=[], train_acc=[], val_loss=[], val_acc=[])

    best_val_acc = -1.0
    out_model = Path(args.out_model)

    # Phase 1
    print("\nPHASE 1: train head only")
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_phase1)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    total_epochs = args.epochs_phase1 + args.epochs_phase2
    phase_boundary_epoch = args.epochs_phase1

    for epoch in range(1, args.epochs_phase1 + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        print(
            f"Epoch {epoch:02d}/{total_epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_model)
            print(f"  ✓ saved best checkpoint (val acc={best_val_acc:.4f}) -> {out_model}")

    # Phase 2
    print("\nPHASE 2: unfreeze last two blocks + fine-tune")
    unfreeze_last_two_blocks(model)  # type: ignore[arg-type]

    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr_phase2)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    for i in range(1, args.epochs_phase2 + 1):
        epoch = args.epochs_phase1 + i
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, loss_fn, optimizer)
        va_loss, va_acc = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        history.train_loss.append(tr_loss)
        history.train_acc.append(tr_acc)
        history.val_loss.append(va_loss)
        history.val_acc.append(va_acc)

        print(
            f"Epoch {epoch:02d}/{total_epochs} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), out_model)
            print(f"  ✓ saved best checkpoint (val acc={best_val_acc:.4f}) -> {out_model}")

    # Load best model for val prediction collection
    model.load_state_dict(torch.load(out_model, map_location=device))

    out_plot = Path(args.out_plot)
    out_val_csv = Path(args.out_val_csv)

    save_training_plot(history, out_plot, phase_boundary_epoch=phase_boundary_epoch)
    save_val_predictions_csv(model, val_loader, class_names, out_val_csv, device)

    print(f"\nSaved:")
    print(f"  model: {out_model}")
    print(f"  plot:  {out_plot}")
    print(f"  val predictions: {out_val_csv}")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
