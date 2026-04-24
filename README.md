# Rice Grain Sorter & Counter

A high-speed rice grain counting and classification system optimized for conveyor belt environments (60 FPS).

## 🚀 Rice Counting System
The primary counting logic is located in:
`Count_Rice_Grains_Video/count_grains_updated.py`

### How it Works
The system uses a multi-stage computer vision pipeline to ensure accuracy at high speeds:

1.  **Detection (HSV Masking):**
    *   The algorithm isolates grains by subtracting the blue conveyor belt background using HSV color ranges.
    *   It filters out "noise" using shape analysis (rejecting long thin streaks caused by motion blur).
2.  **Tracking (Hungarian Algorithm):**
    *   Instead of simple nearest-neighbor matching, we use the **Hungarian Algorithm** (global optimal assignment) to track grains across frames.
    *   **High-Speed Tuning:** Since grains are only visible for 2-3 frames at 60 FPS, the tracker is tuned with a very short "track life" (`max_missed=4`) to prevent merging new grains into old paths.
3.  **Spike Guard:**
    *   A rolling median filter detects sudden spikes in detection counts (caused by lighting reflections or belt edge artifacts) and suppresses those frames to prevent overcounting.
4.  **Classification (Whole vs. Broken):**
    *   Grains are classified in real-time based on their maximum observed area.
    *   **Threshold:** Grains with an area < 300px² are classified as **Broken**, while larger grains are classified as **Whole**.

## 📊 Performance & Validation
We use a batch testing framework to validate accuracy against ground truth (GT) videos.

### Running Tests
To run the full validation suite:
```bash
python Count_Rice_Grains_Video/batch_test_all.py
```

### Current Status (±10% Accuracy Goal)
The system currently achieves a **Mean Absolute Error of 7.6%**, with 8 out of 12 test/training videos falling within the ±10% accuracy target.

| Category | Typical Accuracy |
| :--- | :--- |
| White/Chalky Rice | 94% - 98% |
| Mixed/Broken Rice | 89% - 103% |
| Brown/Yellow Rice | 87% - 112% |

## 📁 Project Structure
*   `Count_Rice_Grains_Video/`: Core counting and tracking implementation.
*   `test_vids/` & `training_vids/`: Dataset used for validation.
*   `scratch/`: Temporary outputs and accuracy logs.
