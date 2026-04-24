"""
RiceNet-v2: Rice Grain Classifier (PyTorch)
============================================
A ResNet-50 based CNN fine-tuned for rice grain classification.
Classifies grains into 6 categories:
  0 - Healthy
  1 - Broken
  2 - Chalky
  3 - Discolored
  4 - Immature
  5 - Long-grain

Requirements:
    pip install torch torchvision pillow numpy

Usage (training):
    python rice_model.py --mode train --data ./dataset --epochs 30

Usage (inference on a single image):
    python rice_model.py --mode infer --image grain.jpg --weights ricenet_v2.pth

Dataset folder structure expected:
    dataset/
      train/
        Healthy/      *.jpg
        Broken/       *.jpg
        Chalky/       *.jpg
        Discolored/   *.jpg
        Immature/     *.jpg
        Long-grain/   *.jpg
      val/
        (same structure)
"""

import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from PIL import Image
import numpy as np

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

CLASS_NAMES = ['Healthy', 'Broken', 'Chalky', 'Discolored', 'Immature', 'Long-grain']
NUM_CLASSES  = len(CLASS_NAMES)
IMG_SIZE     = 224          # ResNet expects 224x224
MEAN         = [0.485, 0.456, 0.406]   # ImageNet mean (transfer learning)
STD          = [0.229, 0.224, 0.225]   # ImageNet std


# ──────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────

def get_transforms(mode='train'):
    """Return augmentation transforms for train vs val/test."""
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
            transforms.RandomCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


# ──────────────────────────────────────────────
# MODEL DEFINITION
# ──────────────────────────────────────────────

class RiceNetV2(nn.Module):
    """
    ResNet-50 backbone with a custom classification head.
    The backbone is initialized with ImageNet weights (transfer learning).
    Only the final fc layer and the last ResNet layer block are unfrozen
    during initial training.
    """

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Freeze all backbone layers
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer4 (fine-tune top residual block)
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Replace the classifier head
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def unfreeze_all(self):
        """Call after initial convergence to fine-tune the full model."""
        for param in self.backbone.parameters():
            param.requires_grad = True


# ──────────────────────────────────────────────
# TRAINING
# ──────────────────────────────────────────────

def train_model(data_dir, epochs=30, batch_size=32, lr=1e-3,
                save_path='ricenet_v2.pth', device=None):
    """Full training loop with validation accuracy tracking."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[RiceNet] Device: {device}")

    # ── Datasets ──
    train_ds = datasets.ImageFolder(
        os.path.join(data_dir, 'train'), transform=get_transforms('train'))
    val_ds   = datasets.ImageFolder(
        os.path.join(data_dir, 'val'),   transform=get_transforms('val'))

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[RiceNet] Train: {len(train_ds)} images  Val: {len(val_ds)} images")
    print(f"[RiceNet] Classes: {train_ds.classes}")

    # ── Model ──
    model = RiceNetV2(num_classes=NUM_CLASSES, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Only update unfrozen parameters
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # ─ Unfreeze full model at epoch 10 ─
        if epoch == 10:
            model.unfreeze_all()
            optimizer = optim.AdamW(model.parameters(), lr=lr * 0.1, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - epoch)
            print("[RiceNet] Epoch 10: unfroze full backbone, lr reduced 10x")

        # ─ Train ─
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        t0 = time.time()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()

        train_loss = total_loss / total
        train_acc  = correct / total * 100

        # ─ Validation ─
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total   += imgs.size(0)
        val_acc = val_correct / val_total * 100

        history['train_loss'].append(round(train_loss, 4))
        history['train_acc'].append(round(train_acc, 2))
        history['val_acc'].append(round(val_acc, 2))

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.1f}%  "
              f"val_acc={val_acc:.1f}%  "
              f"({time.time()-t0:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':      epoch,
                'state_dict': model.state_dict(),
                'val_acc':    val_acc,
                'classes':    CLASS_NAMES,
            }, save_path)
            print(f"    ✓ New best model saved → {save_path}  (val_acc={val_acc:.1f}%)")

    print(f"\n[RiceNet] Training complete. Best val_acc = {best_val_acc:.1f}%")
    # Save history
    hist_path = save_path.replace('.pth', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"[RiceNet] History saved → {hist_path}")
    return model, history


# ──────────────────────────────────────────────
# INFERENCE ON A SINGLE IMAGE
# ──────────────────────────────────────────────

def load_model(weights_path, device=None):
    """Load a saved RiceNetV2 checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt  = torch.load(weights_path, map_location=device)
    model = RiceNetV2(num_classes=NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    print(f"[RiceNet] Loaded weights from {weights_path}  "
          f"(val_acc={ckpt.get('val_acc', '?'):.1f}%)")
    return model, device


def predict_image(model, image_path, device):
    """
    Run inference on a single grain image crop.
    Returns (class_name, confidence, full_probabilities)
    """
    transform = get_transforms('val')
    img = Image.open(image_path).convert('RGB')
    x   = transform(img).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx  = int(np.argmax(probs))
    pred_cls  = CLASS_NAMES[pred_idx]
    pred_conf = float(probs[pred_idx])

    print(f"[RiceNet] Prediction: {pred_cls}  ({pred_conf*100:.1f}%)")
    for i, cls in enumerate(CLASS_NAMES):
        bar = '█' * int(probs[i] * 30)
        print(f"  {cls:<12s} {probs[i]*100:5.1f}%  {bar}")

    return pred_cls, pred_conf, probs.tolist()


# ──────────────────────────────────────────────
# EVALUATION (FULL TEST SET)
# ──────────────────────────────────────────────

def evaluate_model(model, data_dir, device, batch_size=32):
    """
    Evaluate on the test/val set and print:
      - Per-class accuracy
      - Overall accuracy
      - Confusion matrix
    """
    test_ds = datasets.ImageFolder(
        os.path.join(data_dir, 'val'), transform=get_transforms('val'))
    loader  = DataLoader(test_ds, batch_size=batch_size,
                         shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    overall_acc = (all_preds == all_labels).mean() * 100
    print(f"\n[RiceNet] Overall accuracy: {overall_acc:.2f}%")

    # Confusion matrix
    n = NUM_CLASSES
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    print("\n[RiceNet] Confusion Matrix (rows=actual, cols=predicted):")
    header = f"{'':12s}" + "  ".join(f"{c[:5]:>5s}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = f"{CLASS_NAMES[i]:<12s}" + "  ".join(f"{v:5d}" for v in row)
        print(row_str)

    print("\n[RiceNet] Per-class accuracy:")
    for i, cls in enumerate(CLASS_NAMES):
        total_cls = cm[i].sum()
        acc = cm[i, i] / total_cls * 100 if total_cls > 0 else 0
        print(f"  {cls:<12s}: {acc:.1f}%  ({cm[i,i]}/{total_cls})")

    return {
        'overall_acc': round(overall_acc, 2),
        'confusion_matrix': cm.tolist(),
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RiceNet-v2 Classifier')
    parser.add_argument('--mode',    choices=['train', 'infer', 'eval'], required=True)
    parser.add_argument('--data',    default='./dataset', help='Dataset root folder')
    parser.add_argument('--image',   help='Image path for inference mode')
    parser.add_argument('--weights', default='ricenet_v2.pth',
                        help='Model weights file (.pth)')
    parser.add_argument('--epochs',  type=int,   default=30)
    parser.add_argument('--batch',   type=int,   default=32)
    parser.add_argument('--lr',      type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == 'train':
        print("[RiceNet] Starting training...")
        train_model(
            data_dir  = args.data,
            epochs    = args.epochs,
            batch_size= args.batch,
            lr        = args.lr,
            save_path = args.weights,
        )

    elif args.mode == 'infer':
        if not args.image:
            parser.error('--image is required for infer mode')
        model, device = load_model(args.weights)
        predict_image(model, args.image, device)

    elif args.mode == 'eval':
        model, device = load_model(args.weights)
        evaluate_model(model, args.data, device)
