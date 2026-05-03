"""Torch utilities for training/inference on extracted rice grain crops."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T


@dataclass(frozen=True)
class Checkpoint:
    arch: str
    num_classes: int
    class_names: List[str]
    state_dict: Dict


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


def get_transforms(img_size: int = 128, train: bool = True) -> T.Compose:
    # Keep hue jitter small; too much breaks label identity.
    if train:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.2),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12, hue=0.03),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def save_checkpoint(path: Path, *, arch: str, class_names: List[str], model: nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": arch,
        "num_classes": len(class_names),
        "class_names": class_names,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, str(path))


def load_checkpoint(path: Path, device: torch.device) -> Tuple[nn.Module, List[str]]:
    payload = torch.load(str(path), map_location=device)
    arch = payload["arch"]
    class_names = list(payload["class_names"])
    num_classes = int(payload["num_classes"])

    model = build_model(arch, num_classes=num_classes, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, class_names
