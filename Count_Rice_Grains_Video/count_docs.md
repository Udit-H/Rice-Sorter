# Rice Grain Counter — Documentation

## Overview

`count_grains.py` counts rice grains in a conveyor belt video by detecting grain
contours in each frame, tracking them across frames with unique IDs, and returning
the total number of unique IDs ever assigned. A grain visible across N frames is
counted exactly once.

---

## Problem with the Original Approach

The original notebooks summed per-frame grain detections. A single grain visible
across 16 frames was counted 16 times, producing massive overcounts (e.g. 3289
detected vs 200 ground truth for `test_7_200_original`).

---

## Pipeline

### 1. Rice Type Detection — `_auto_mode(frame)`

Inspects the luminance (V channel) of the first video frame:

- If more than 15% of pixels are dark (V < 80) **and** dark pixels rival bright pixels
  (V > 150), the video is classified as `"dark"` (black rice).
- Otherwise defaults to `"white"`.

Can be overridden via the `--mode` CLI flag or the `OVERRIDES` dict in `run_tests.py`.

---

### 2. Colour Masking — `_make_mask(frame, mode)`

Converts the frame to HSV and builds a binary mask of grain pixels.

#### White / Chalky Rice (default)

```
Layer 1:  V > 100,  S < 90   (bright grains)
Layer 2:  V >  75,  S < 55   (dim / chalky grains)
Belt sub: H = 90-135, S > 70  (excluded)
```

The blue conveyor belt has H ≈ 115, S ≈ 120–130, V ≈ 90–110. Explicitly
subtracting it allows lowering the V threshold from 120 to 100, which is
necessary for dim videos like `test_7_200` (V_max ≈ 149) where grains are
barely brighter than the belt.

#### Dark (Black) Rice

```
V range: 10 – 75   (grains are darker than the belt at V ≈ 100)
```

Tight upper bound prevents belt noise from leaking into the mask.

#### Brown Rice

```
H: 5–30,  S: 20–210,  V: 60–210
```

#### Morphological Cleanup

After masking, two operations are applied to all modes:

| Step | Kernel | Iterations | Purpose |
|------|--------|-----------|---------|
| MORPH_OPEN  | 3×3 ellipse | 1 | Remove small noise blobs |
| MORPH_CLOSE | 5×5 ellipse | 2 | Fill small holes inside grains |

---

### 3. Contour Detection — `detect_grains(frame, mode)`

Finds external contours in the binary mask and filters by area:

- **min_area = 55 px²** — removes dust / single-pixel noise
- **max_area = 20 000 px²** — allows grain clusters (raised from original 7 000)

---

### 4. Blob Splitting — `_split_blob(contour)` / `centroids_of(contours)`

A single contour can represent multiple touching grains. `centroids_of` handles this:

- Contours **≤ 1 400 px²** → one centroid (standard moment calculation).
- Contours **> 1 400 px²** → treated as a cluster.

Cluster splitting uses a **distance transform + iterative peak suppression**:

1. Draw the contour into a local binary image.
2. Compute `cv2.distanceTransform` — pixels near the centre of each grain have
   the highest values.
3. Repeatedly find the global maximum, record it as a grain centroid, then
   suppress a circle of radius ≈ 0.6 × sqrt(typical_area / pi) around it.
4. Repeat for `n = round(area / typical_grain_area)` times (default 750 px²).

This correctly split clusters of 1 764–4 925 px² in `test_7_200` into individual
grain positions, lifting its accuracy from 35% to ~103%.

---

### 5. Centroid Tracking — `UniqueGrainTracker`

Greedy nearest-neighbour tracker across frames.

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `max_dist` | 70 px | Maximum distance to match a detection to an existing track |
| `max_missed` | `int(fps × 0.33)` | Frames a track survives without a match (~0.33 s real time) |

**Why fps-scaled `max_missed`?**  
At 30 fps, 10 frames = 0.33 s. At 60 fps, the same 10 frames = only 0.17 s — too
short, causing new grains entering a vacated position to be counted as separate
grains when they should reuse the dying track. Scaling by FPS keeps the real-time
window consistent across videos.

**Per-frame update logic:**

1. Match existing tracks to nearest detections within `max_dist`.
2. Increment `missed` counter for unmatched tracks.
3. Any detection not matched to an existing track → **new unique ID** (`_nid += 1`).
4. Prune tracks with `missed > max_missed`.

Final count = `tracker._nid` (total unique IDs ever assigned).

---

## Test Runner — `run_tests.py`

Runs `process_video` on all labelled videos and prints an accuracy table.

### Ground Truth

| Video | GT Count |
|-------|---------|
| test_1_50.mp4 | 50 |
| test_2_50.mp4 | 50 |
| test_3_100.mp4 | 100 |
| test_4_100.mp4 | 100 |
| test_5_100.mp4 | 100 |
| test_6_100.mp4 | 100 |
| test_7_200.mp4 | 200 |
| test_7_200_original.mp4 | 200 |
| black_rice.mp4 | 54 |
| test_set_1.mp4 | 100 |

### Mode Overrides

```python
OVERRIDES = {
    "black_rice.mp4":          "dark",
    "test_7_200_original.mp4": "white",
}
```

`test_7_200_original` was being auto-detected as `"dark"` (dim conveyor
belt triggered the dark_frac threshold), giving only 19.5% accuracy. Forcing
`"white"` is correct for this video.

---

## Accuracy Results (final)

| Video | GT | Detected | Accuracy |
|-------|----|---------|---------|
| test_7_200.mp4 | 200 | 207 | 103.5% |
| test_7_200_original.mp4 | 200 | 188 | 94.0% |
| test_6_100.mp4 | 100 | 107 | 107.0% |
| test_4_100.mp4 | 100 | 86 | 86.0% |
| test_3_100.mp4 | 100 | 84 | 84.0% |
| test_5_100.mp4 | 100 | 77 | 77.0% |
| test_set_1.mp4 | 100 | 132 | 132.0% |
| test_1_50.mp4 | 50 | 60 | 120.0% |
| test_2_50.mp4 | 50 | 79 | 158.0% |
| black_rice.mp4 | 54 | 144 | 266.7% |

---

## CLI Usage

```bash
# Count grains (auto-detect rice type)
python count_grains.py path/to/video.mp4

# Force white rice mode and save annotated output
python count_grains.py path/to/video.mp4 --mode white --out annotated.mp4

# Run full accuracy test suite
python run_tests.py
```

---

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Unique-ID tracking instead of frame summation | Frame summation counts each grain N times (once per visible frame) |
| Belt exclusion in white mask | Allows V threshold as low as 100 without belt false-positives |
| Distance-transform blob splitting | Clusters of touching grains were each counted as 1; splitting recovers individual grain centres |
| FPS-scaled max_missed | Prevents 60 fps videos from recycling stale track IDs too aggressively |
| max_area raised to 20 000 | Grain clusters larger than the old 7 000 px limit were silently dropped |
