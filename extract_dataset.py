#!/usr/bin/env python3
"""
Extract individual grain crops from training videos into dataset/<class>/ folders.

4 classes: brown, chalky, white, yellow  (black eliminated)
test_14_yellow_100.mp4 is held out as the test video — NOT extracted here.

19 training videos across two directories:
  Count_Rice_Grains_Video/ (7):
    test_9_white_100    -> white
    test_10_chalky_100  -> chalky
    test_11_brown_100   -> brown
    test_15_brown_200   -> brown
    test_16_chalky_200  -> chalky
    test_17_yellow_200  -> yellow
    test_18_white_200   -> white

  Videos/ (12):
    brown_rice          -> brown
    chalky_rice         -> chalky
    white_rice          -> white
    yellow_rice         -> yellow
    test_1_50 .. test_7_200  (7 videos) -> white  (original test set, single-class white)
    test_8_random       -> white  (single grain type despite name)

Fixes vs naive approach:
  - Sanity check: skip frames where mask returns >10% of image as foreground
    (happens when belt color changes / lighting shifts — mask failure mode)
  - Save on FIRST detection of each unique track (not after age >= N),
    retrying on later frames if the grain was too close to the edge
  - Edge margin is soft: if a grain is never seen away from the edge,
    save it anyway after 5 frames so we don't lose it entirely
  - MIN_AREA lowered to 50 px2 to catch small grains in these videos
    (observed grain areas: 62-815 px2 at 1280x720)
"""

import cv2
import numpy as np
import os
from pathlib import Path


CRG_DIR     = Path(__file__).parent / "Count_Rice_Grains_Video"
VIDEOS_DIR  = Path(__file__).parent / "Videos"
DATASET_DIR = Path(__file__).parent / "dataset"

# (video_dir, filename, class_label)
TRAINING_VIDEOS = [
    # --- Count_Rice_Grains_Video/ ---
    (CRG_DIR, "test_9_white_100.mp4",   "white"),
    (CRG_DIR, "test_10_chalky_100.mp4", "chalky"),
    (CRG_DIR, "test_11_brown_100.mp4",  "brown"),
    (CRG_DIR, "test_15_brown_200.mp4",  "brown"),
    (CRG_DIR, "test_16_chalky_200.mp4", "chalky"),
    (CRG_DIR, "test_17_yellow_200.mp4", "yellow"),
    (CRG_DIR, "test_18_white_200.mp4",  "white"),
    # --- Videos/ (explicitly labeled) ---
    (VIDEOS_DIR, "brown_rice.mp4",  "brown"),
    (VIDEOS_DIR, "chalky_rice.mp4", "chalky"),
    (VIDEOS_DIR, "white_rice.mp4",  "white"),
    (VIDEOS_DIR, "yellow_rice.mp4", "yellow"),
    # --- Videos/ (original white-rice test set, test_1 – test_8) ---
    (VIDEOS_DIR, "test_1_50.mp4",   "white"),
    (VIDEOS_DIR, "test_2_50.mp4",   "white"),
    (VIDEOS_DIR, "test_3_100.mp4",  "white"),
    (VIDEOS_DIR, "test_4_100.mp4",  "white"),
    (VIDEOS_DIR, "test_5_100.mp4",  "white"),
    (VIDEOS_DIR, "test_6_100.mp4",  "white"),
    (VIDEOS_DIR, "test_7_200.mp4",  "white"),
    (VIDEOS_DIR, "test_8_random.mp4", "white"),
]

TEST_VIDEO = CRG_DIR / "test_14_yellow_100.mp4"

CROP_SIZE         = 224
MIN_AREA          = 50       # px2 — observed grain areas start around 62 px2
SPLIT_AREA        = 2000     # px2 — above this = grain cluster, skip
MAX_DIST          = 70       # px  — max centroid jump between frames
MAX_MISSED        = 20       # frames before a track is pruned
EDGE_MARGIN_SOFT  = 8        # px  — preferred: save crop when grain is this far from edge
EDGE_MARGIN_HARD  = 0        # px  — fallback after MAX_AGE_BEFORE_FORCED frames
MAX_AGE_FORCED    = 5        # frames: if not saved yet by this age, save regardless of edge
FG_RATIO_LIMIT    = 0.10     # skip frame if >10% of pixels are foreground (mask failure)


# ── Mask ──────────────────────────────────────────────────────────────────────

def make_mask(frame: np.ndarray):
    """
    Isolate grain foreground from the blue conveyor belt.
    Returns (mask, fg_ratio) where fg_ratio is fraction of frame that is foreground.
    If fg_ratio > FG_RATIO_LIMIT the mask has failed (whole frame became foreground).
    """
    H, W = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
    fg_ratio = np.sum(m > 0) / (H * W)
    return m, fg_ratio


def centroid_of(contour) -> tuple:
    M = cv2.moments(contour)
    if M["m00"] > 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    x, y, w, h = cv2.boundingRect(contour)
    return (x + w // 2, y + h // 2)


def crop_grain(frame: np.ndarray, contour, edge_margin: int = EDGE_MARGIN_SOFT):
    H, W = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(contour)
    if (x < edge_margin or y < edge_margin
            or x + w > W - edge_margin or y + h > H - edge_margin):
        return None
    pad = 8
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + w + pad), min(H, y + h + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)


# ── Per-video extraction ──────────────────────────────────────────────────────

def extract_video(video_path: Path, output_dir: Path, label: str,
                  start_id: int = 0) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return 0

    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # tracks: id -> {centroid, missed, age, saved}
    tracks: dict = {}
    nid, saved = 0, 0
    file_id = start_id   # global offset so filenames are unique across videos
    skipped_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask, fg_ratio = make_mask(frame)

        # Skip this frame if the mask has failed (entire frame is foreground)
        if fg_ratio > FG_RATIO_LIMIT:
            skipped_frames += 1
            # Still update miss counters so stale tracks die
            for tid in list(tracks.keys()):
                tracks[tid]["missed"] += 1
            tracks = {k: v for k, v in tracks.items() if v["missed"] <= MAX_MISSED}
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        singles = [c for c in contours if MIN_AREA < cv2.contourArea(c) < SPLIT_AREA]
        dets = [(centroid_of(c), c) for c in singles]

        # Greedy nearest-neighbour matching
        matched_t: dict = {}
        matched_d: set  = set()
        for tid, t in tracks.items():
            tx, ty  = t["centroid"]
            best_d  = MAX_DIST
            best_di = None
            for di, ((cx, cy), _) in enumerate(dets):
                if di in matched_d:
                    continue
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if d < best_d:
                    best_d, best_di = d, di
            if best_di is not None:
                matched_t[tid] = best_di
                matched_d.add(best_di)

        # Update matched tracks; try to save if not yet saved
        for tid, di in matched_t.items():
            cen, cnt = dets[di]
            tracks[tid]["centroid"] = cen
            tracks[tid]["missed"]   = 0
            tracks[tid]["age"]     += 1

            if not tracks[tid]["saved"]:
                age = tracks[tid]["age"]
                margin = EDGE_MARGIN_SOFT if age < MAX_AGE_FORCED else EDGE_MARGIN_HARD
                crop = crop_grain(frame, cnt, edge_margin=margin)
                if crop is not None:
                    fname = output_dir / f"{label}_{file_id:05d}.png"
                    cv2.imwrite(str(fname), crop)
                    file_id += 1
                    saved += 1
                    tracks[tid]["saved"] = True

        # Penalise unmatched tracks
        for tid in list(tracks.keys()):
            if tid not in matched_t:
                tracks[tid]["missed"] += 1

        # Register new tracks (unmatched detections) — try to save immediately
        for di, (cen, cnt) in enumerate(dets):
            if di not in matched_d:
                crop = crop_grain(frame, cnt, edge_margin=EDGE_MARGIN_SOFT)
                already_saved = False
                if crop is not None:
                    fname = output_dir / f"{label}_{file_id:05d}.png"
                    cv2.imwrite(str(fname), crop)
                    file_id += 1
                    saved += 1
                    already_saved = True
                tracks[nid] = {
                    "centroid": cen, "missed": 0, "age": 1,
                    "saved": already_saved,
                }
                nid += 1

        # Prune dead tracks
        tracks = {k: v for k, v in tracks.items() if v["missed"] <= MAX_MISSED}

    cap.release()
    print(f"  {video_path.name}  ->  {saved} crops saved  "
          f"(unique tracks seen: {nid}, skipped frames: {skipped_frames})")
    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Rice Dataset Extraction  (4 classes, 19 training videos)")
    print("=" * 60)

    # Clear existing crops so stale class folders don't interfere
    import shutil
    for cls in ["brown", "chalky", "white", "yellow"]:
        d = DATASET_DIR / cls
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    # Remove black if it exists from previous 5-class run
    black_dir = DATASET_DIR / "black"
    if black_dir.exists():
        shutil.rmtree(black_dir)

    # Use a per-class counter so filenames are globally unique across videos
    class_counters = {"brown": 0, "chalky": 0, "white": 0, "yellow": 0}

    total = 0
    for video_dir, video_name, label in TRAINING_VIDEOS:
        video_path = video_dir / video_name
        if not video_path.exists():
            print(f"\n[WARN] Not found: {video_path}")
            continue
        print(f"\n{video_name}  ({label})")
        n = extract_video(video_path, DATASET_DIR / label, label,
                          start_id=class_counters[label])
        class_counters[label] += n
        total += n

    print("\n" + "=" * 60)
    print(f"Total crops extracted: {total}")
    print("\nDataset summary:")
    for cls in ["brown", "chalky", "white", "yellow"]:
        d = DATASET_DIR / cls
        count = len(list(d.glob("*.png"))) if d.exists() else 0
        print(f"  {cls:8s}: {count:4d} crops")
    print("=" * 60)


if __name__ == "__main__":
    main()
