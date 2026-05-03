# Rice video pipeline (new_vids)

This repo contains two core capabilities:

1. **Counting rice grains in a video** (unique-grain counting via tracking)
2. **Classifying rice color/quality per grain crop** (brown / chalky / white / yellow)

The **current workflow** assumes the videos you want to use are in **`new_vids/`** and that each filename contains one of these labels:

- `brown`, `white`, `yellow`, `chalky`

Example: `test_24_yellow_bunch.mp4` → label `yellow`.

---

## 1) Methodology: Counting (tracking-based)

Implemented in:
- `Count/count_grains_updated.py`

### Detection
Each frame is converted to HSV and a **foreground mask** is built using a belt/background suppression approach (the conveyor/belt tends to be blue). Then contours are extracted and filtered by area.

Key steps:
- HSV thresholding to isolate the belt and belt shadows
- Invert to obtain foreground grains
- Morphological open/close to clean noise
- Contour filtering using `min_area` / `max_area`

### Splitting clumps
Large blobs (multiple grains touching) are split by estimating multiple centroids using a **distance-transform peak finding** heuristic.

### Unique counting
Instead of summing detections per frame, a `UniqueGrainTracker` assigns IDs to centroids across frames using a greedy nearest-neighbour match.

- New centroid → new track ID → increases unique count
- Tracks are dropped if missing for `max_missed` frames

This ensures grains visible across many frames are counted **once**.

---

## 2) Methodology: Grain extraction to 128×128 crops

Implemented in:
- `video_grain_dataset.py`
- `build_dataset_from_new_vids.py`

We reuse the *same detector* from `Count/count_grains_updated.py` (`detect_grains`) to ensure the classifier sees the same kind of crops the counter is based on.

For each processed frame:
- contours → bounding rectangles
- apply padding around each rectangle
- crop from the frame
- resize to **128×128**

### Train/val split
The split is deterministic and based on the **processed frame index** (not raw frame index), which avoids pathological settings (e.g. `frame_stride=5` and `val_every_n_frames=5`) putting everything into validation.

Output layout (ImageFolder-style):

```
extracted_grains/<name>/
  train/<class>/*.png
  val/<class>/*.png
  manifest.csv
```

---

## 3) Methodology: Training the model (torch ResNet18)

Implemented in:
- `train_grain_classifier_new_vids.py`
- `torch_rice_model.py`

### Model
- Backbone: **ResNet18**
- Input: 128×128 RGB crops
- Output classes: `brown`, `chalky`, `white`, `yellow`

### Why this works
On these crops, the most difficult separation is typically **white vs chalky** (the rest are easier, especially brown). The ResNet model learns color/brightness patterns and some texture cues even with aggressive blur and small crop size.

### Training details
- Data augmentation (small hue jitter, rotation, flips, mild color jitter)
- **Weighted sampling** to reduce class imbalance effects
- Optimizer: AdamW + cosine LR schedule
- Checkpoints saved at epochs **20 / 25 / 30** (matching the prior cadence you mentioned)

Outputs:
- `models/rice_resnet18_new_vids_best.pth` (best validation accuracy)

---

## 4) Evaluation runner (better `run_all_tests`)

Implemented in:
- `run_all_tests.py`

What it does:
1. (Optional) rebuild the crop dataset from `new_vids/`
2. evaluate:
   - torch checkpoint (`models/rice_resnet18_new_vids_best.pth`)
   - optionally the older baseline `rice_classifier_v4.pkl`

Metrics reported:
- confusion matrix
- per-class precision/recall/F1
- per-video majority vote metrics (useful since each video is a single class)

---

## Quickstart

### A) Build crops from `new_vids/`

```
/home/adithya/scratch/rice/venv/bin/python build_dataset_from_new_vids.py \
  --out-dir extracted_grains/new_vids_v1 \
  --frame-stride 10 \
  --val-every-n-frames 5 \
  --max-crops-per-video 25000 \
  --max-crops-per-class 25000
```

### B) Train

```
/home/adithya/scratch/rice/venv/bin/python train_grain_classifier_new_vids.py \
  --data-dir extracted_grains/new_vids_v1 \
  --epochs 30 \
  --batch-size 128 \
  --num-workers 4
```

### C) Evaluate

```
/home/adithya/scratch/rice/venv/bin/python run_all_tests.py \
  --extract-dir extracted_grains/new_vids_v1 \
  --eval-torch \
  --torch-ckpt models/rice_resnet18_new_vids_best.pth
```

---

## Notes / gotchas

- `new_vids/` labeling is filename-based; if you add new videos, include the class token in the name.
- Crop extraction can generate a lot of files. Use `--frame-stride` and caps (`--max-crops-per-*`) to control disk usage.
- If you want to re-train on larger data, it’s usually worth increasing crop count per class rather than lowering stride too much.
