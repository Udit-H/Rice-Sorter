#!/usr/bin/env python3
"""
Rice grain classifier — EfficientNet_B0 transfer learning (PyTorch).
4 classes: brown, chalky, white, yellow  (black eliminated)

Two-phase training:
  Phase 1 — freeze backbone, train custom head (10 epochs, lr=1e-3)
  Phase 2 — unfreeze last two MBConv blocks (10 epochs, lr=1e-5)

Outputs:
  rice_model.pth       — best-checkpoint weights
  training_plot.png    — accuracy + loss curves with phase-boundary line
  val_predictions.csv  — per-grain true/pred/confidence/full-prob-vector

Post-training test:
  Extracts crops from test_14_yellow_100.mp4 and runs inference to report
  class distribution (all grains should predict yellow).
"""

import csv
import cv2
import numpy as np
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from torchvision.models import EfficientNet_B0_Weights

# ── Config ────────────────────────────────────────────────────────────────────

DATASET_DIR  = Path(__file__).parent / "dataset"
TEST_VIDEO   = Path(__file__).parent / "Count_Rice_Grains_Video" / "test_14_yellow_100.mp4"
NUM_CLASSES  = 4
CLASSES      = ["brown", "chalky", "white", "yellow"]   # alphabetical = ImageFolder order
BATCH_SIZE   = 32
SEED         = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ── Transforms ────────────────────────────────────────────────────────────────

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ── Dataset split with separate transforms ────────────────────────────────────

class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


base_dataset = datasets.ImageFolder(str(DATASET_DIR), transform=None)
print(f"Dataset: {len(base_dataset)} images  |  classes: {base_dataset.classes}")
assert base_dataset.classes == CLASSES, (
    f"Expected classes {CLASSES}, got {base_dataset.classes}. "
    "Check that dataset/ subfolders match exactly."
)

n_total = len(base_dataset)
n_val   = int(n_total * 0.2)
n_train = n_total - n_val

train_sub, val_sub = random_split(
    base_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

train_dataset = TransformSubset(train_sub, train_transform)
val_dataset   = TransformSubset(val_sub,   val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_dataset)}  |  Val: {len(val_dataset)}")


# ── Class weights ─────────────────────────────────────────────────────────────

train_labels  = [base_dataset.targets[i] for i in train_sub.indices]
class_counts  = Counter(train_labels)
total_train   = len(train_labels)
class_weights = torch.tensor(
    [total_train / (NUM_CLASSES * class_counts[i]) for i in range(NUM_CLASSES)],
    dtype=torch.float32,
).to(device)
print("Class weights:", {CLASSES[i]: f"{class_weights[i].item():.3f}" for i in range(NUM_CLASSES)})

criterion = nn.CrossEntropyLoss(weight=class_weights)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int) -> nn.Module:
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    # 4 classes: 256->output jump is fine, no extra 128 block needed
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


model = build_model(NUM_CLASSES).to(device)


# ── Training helpers ──────────────────────────────────────────────────────────

def run_epoch(loader, model, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total


best_val_acc = 0.0


def train_phase(model, n_epochs: int, lr: float, label: str, history: dict):
    global best_val_acc
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    for epoch in range(1, n_epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, model, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(val_loader,   model, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        flag = ""
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), "rice_model.pth")
            flag = "  <- saved"

        print(
            f"  [{label}] Epoch {epoch:2d}/{n_epochs}"
            f"  train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}"
            f"  val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}{flag}"
        )


# ── Phase 1 ───────────────────────────────────────────────────────────────────

history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

print("\n" + "=" * 64)
print("Phase 1 -- backbone frozen, training head (lr=1e-3, 10 epochs)")
print("=" * 64)
train_phase(model, n_epochs=10, lr=1e-3, label="Phase1", history=history)


# ── Phase 2 ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 64)
print("Phase 2 -- unfreezing model.features[-2:], fine-tuning (lr=1e-5, 10 epochs)")
print("=" * 64)

for p in model.features[-1].parameters():
    p.requires_grad = True
for p in model.features[-2].parameters():
    p.requires_grad = True

train_phase(model, n_epochs=10, lr=1e-5, label="Phase2", history=history)

print(f"\nBest val accuracy overall: {best_val_acc:.4f}")


# ── Training plot ─────────────────────────────────────────────────────────────

epochs = list(range(1, 21))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, history["train_acc"], label="Train Acc")
ax1.plot(epochs, history["val_acc"],   label="Val Acc")
ax1.axvline(10, color="gray", linestyle="--", linewidth=1.2, label="Phase boundary")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs, history["train_loss"], label="Train Loss")
ax2.plot(epochs, history["val_loss"],   label="Val Loss")
ax2.axvline(10, color="gray", linestyle="--", linewidth=1.2, label="Phase boundary")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_plot.png", dpi=150)
print("Saved training_plot.png")


# ── Val inference → val_predictions.csv ──────────────────────────────────────

print("Running inference on validation set...")
model.load_state_dict(torch.load("rice_model.pth", map_location=device))
model.eval()
softmax = nn.Softmax(dim=1)

rows = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs   = imgs.to(device)
        probs  = softmax(model(imgs)).cpu().numpy()
        preds  = probs.argmax(axis=1)
        for i in range(len(labels)):
            row = {
                "true_label": CLASSES[labels[i].item()],
                "pred_label": CLASSES[preds[i]],
                "confidence": f"{probs[i, preds[i]]:.4f}",
            }
            for j, cls in enumerate(CLASSES):
                row[f"prob_{cls}"] = f"{probs[i, j]:.4f}"
            rows.append(row)

fieldnames = ["true_label", "pred_label", "confidence"] + [f"prob_{c}" for c in CLASSES]
with open("val_predictions.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved val_predictions.csv  ({len(rows)} rows)")


# ── Test on test_14_yellow_100.mp4 ────────────────────────────────────────────
# Extract grain crops from the held-out test video, run classifier on each,
# and report per-class prediction counts.  Expected: ~100% yellow.

print("\n" + "=" * 64)
print("Test evaluation: test_14_yellow_100.mp4  (GT = 100 yellow grains)")
print("=" * 64)

# Inline mask + crop helpers (same logic as extract_dataset.py)
def _test_make_mask(frame):
    hsv    = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    belt   = cv2.inRange(hsv, (90, 50,  40), (140, 255, 255))
    shadow = cv2.inRange(hsv, (90, 40,  10), (140, 255,  80))
    bg     = cv2.bitwise_or(belt, shadow)
    fg     = cv2.bitwise_not(bg)
    valid  = cv2.inRange(hsv, (0, 0, 15), (180, 255, 255))
    m = cv2.bitwise_and(fg, valid)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k3, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k5, iterations=2)
    H, W = frame.shape[:2]
    fg_ratio = np.sum(m > 0) / (H * W)
    return m, fg_ratio

def _test_centroid(c):
    M = cv2.moments(c)
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    x, y, w, h = cv2.boundingRect(c)
    return (x + w // 2, y + h // 2)

def _test_crop(frame, contour, margin=8):
    H, W = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    if x < margin or y < margin or x+w > W-margin or y+h > H-margin:
        return None
    pad = 8
    crop = frame[max(0,y-pad):min(H,y+h+pad), max(0,x-pad):min(W,x+w+pad)]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


def extract_test_crops(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return []
    tracks: dict = {}
    nid = 0
    crops = []
    MAX_DIST, MAX_MISSED, MIN_AREA, SPLIT_AREA = 70, 20, 50, 2000
    MAX_AGE_FORCED, FG_LIMIT = 5, 0.10

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mask, fg_ratio = _test_make_mask(frame)
        if fg_ratio > FG_LIMIT:
            for tid in tracks: tracks[tid]["missed"] += 1
            tracks = {k: v for k, v in tracks.items() if v["missed"] <= MAX_MISSED}
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        singles = [c for c in contours if MIN_AREA < cv2.contourArea(c) < SPLIT_AREA]
        dets = [(_test_centroid(c), c) for c in singles]

        matched_t, matched_d = {}, set()
        for tid, t in tracks.items():
            tx, ty = t["centroid"]
            best_d, best_di = MAX_DIST, None
            for di, ((cx, cy), _) in enumerate(dets):
                if di in matched_d: continue
                d = ((cx-tx)**2 + (cy-ty)**2)**0.5
                if d < best_d: best_d, best_di = d, di
            if best_di is not None:
                matched_t[tid] = best_di; matched_d.add(best_di)

        for tid, di in matched_t.items():
            cen, cnt = dets[di]
            tracks[tid]["centroid"] = cen
            tracks[tid]["missed"]   = 0
            tracks[tid]["age"]     += 1
            if not tracks[tid]["saved"]:
                margin = 8 if tracks[tid]["age"] < MAX_AGE_FORCED else 0
                crop = _test_crop(frame, cnt, margin)
                if crop is not None:
                    crops.append(crop); tracks[tid]["saved"] = True

        for tid in list(tracks.keys()):
            if tid not in matched_t: tracks[tid]["missed"] += 1

        for di, (cen, cnt) in enumerate(dets):
            if di not in matched_d:
                crop = _test_crop(frame, cnt, 8)
                saved = False
                if crop is not None:
                    crops.append(crop); saved = True
                tracks[nid] = {"centroid": cen, "missed": 0, "age": 1, "saved": saved}
                nid += 1

        tracks = {k: v for k, v in tracks.items() if v["missed"] <= MAX_MISSED}

    cap.release()
    print(f"  Extracted {len(crops)} crops from {video_path.name}  (unique tracks: {nid})")
    return crops


test_crops = extract_test_crops(TEST_VIDEO)

if test_crops:
    model.eval()
    pred_counts = Counter()
    conf_per_class = {cls: [] for cls in CLASSES}

    with torch.no_grad():
        for crop_bgr in test_crops:
            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = val_transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
            probs  = softmax(model(tensor)).cpu().numpy()[0]
            pred   = CLASSES[probs.argmax()]
            pred_counts[pred] += 1
            conf_per_class[pred].append(probs.max())

    n = len(test_crops)
    print(f"\n  Classification results ({n} grains extracted from test video):")
    print(f"  {'Class':<10}  {'Count':>6}  {'%':>7}  {'Avg conf':>9}")
    print(f"  {'-'*38}")
    for cls in CLASSES:
        cnt  = pred_counts[cls]
        pct  = 100 * cnt / n if n else 0
        conf = sum(conf_per_class[cls]) / len(conf_per_class[cls]) if conf_per_class[cls] else 0
        marker = " <-- expected" if cls == "yellow" else ""
        print(f"  {cls:<10}  {cnt:>6}  {pct:>6.1f}%  {conf:>8.4f}{marker}")

    yellow_acc = 100 * pred_counts["yellow"] / n if n else 0
    print(f"\n  Yellow recall on test video: {pred_counts['yellow']}/{n} = {yellow_acc:.1f}%")
else:
    print("  No crops extracted from test video.")

print("\nDone.")
