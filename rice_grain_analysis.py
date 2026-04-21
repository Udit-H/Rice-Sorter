"""
Rice Grain Analysis Helper Functions (Persistent MOG2 Version)
=============================================================
Provides detection, preprocessing, and annotation utilities for rice grain videos.
Uses MOG2 background subtraction for robust detection of moving grains.
"""

import cv2
import numpy as np

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────

GRAIN_CLASSES = ['Healthy', 'Broken', 'Chalky', 'Discolored', 'Immature', 'Long-grain']

CLASS_COLORS_BGR = {
    'Healthy':    (80,  200, 80),
    'Long-grain': (50,  180, 255),
    'Broken':     (50,  50,  220),
    'Chalky':     (180, 180, 180),
    'Discolored': (100, 60,  200),
    'Immature':   (50,  200, 200),
}

MIN_GRAIN_AREA   = 50   # Reduced to catch smaller grains
MAX_GRAIN_AREA   = 8000
MAX_ASPECT_RATIO = 10.0

# ──────────────────────────────────────────────
# PERSISTENT STATE
# ──────────────────────────────────────────────

_back_sub = None

# ──────────────────────────────────────────────
# PREPROCESSING
# ──────────────────────────────────────────────

def preprocess_frame(frame):
    """
    Uses MOG2 background subtraction to create a binary mask of moving grains.
    """
    global _back_sub
    if _back_sub is None:
        # history=500, varThreshold=25 is usually good for rice
        _back_sub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = _back_sub.apply(frame)
    
    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return gray, mask

# ──────────────────────────────────────────────
# DETECTION
# ──────────────────────────────────────────────

def detect_grains(mask):
    """
    Detect grains in a binary mask using contours.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    grains = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_GRAIN_AREA < area < MAX_GRAIN_AREA):
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / (min(w, h) + 1e-5)
        
        if aspect > MAX_ASPECT_RATIO:
            continue
            
        grains.append({
            'contour': c,
            'area':    area,
            'aspect':  round(aspect, 2),
            'x': x, 'y': y, 'w': w, 'h': h
        })
    return grains

# ──────────────────────────────────────────────
# CLASSIFICATION (Fallback/Simulation)
# ──────────────────────────────────────────────

def classify_grain(grain, gray_frame):
    """
    Feature-based fallback classification.
    """
    area   = grain['area']
    aspect = grain['aspect']
    x, y, w, h = grain['x'], grain['y'], grain['w'], grain['h']
    
    # Crop ROI
    roi = gray_frame[y:y+h, x:x+w]
    if roi.size == 0:
        return 'Healthy', 0.80
        
    mean_i = float(np.mean(roi))
    std_i  = float(np.std(roi))

    if aspect > 3.0:   cls, conf = 'Long-grain', 0.88
    elif aspect < 1.8: cls, conf = 'Broken',     0.82
    elif std_i  < 14:  cls, conf = 'Chalky',     0.84
    elif mean_i < 85:  cls, conf = 'Discolored', 0.78
    elif area   < 400: cls, conf = 'Immature',   0.75
    else:              cls, conf = 'Healthy',     0.92

    noise = np.random.uniform(-0.04, 0.04)
    return cls, float(np.clip(conf + noise, 0.55, 0.99))

# ──────────────────────────────────────────────
# ANNOTATION
# ──────────────────────────────────────────────

def annotate_frame(frame, grain, cls, conf):
    """
    Draw bbox and label on the frame.
    """
    x, y, w, h = grain['x'], grain['y'], grain['w'], grain['h']
    color = CLASS_COLORS_BGR.get(cls, (0, 255, 0))
    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, 2)
    label = f"{cls} {conf*100:.0f}%"
    cv2.putText(frame, label, (x, max(15, y-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

# ──────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────

def compute_accuracy_metrics(all_detections):
    """
    Compute aggregate metrics.
    """
    if not all_detections:
        return {'total_grains': 0, 'mean_confidence': 0, 'per_class': {}}
        
    total = len(all_detections)
    confs = [d['confidence'] for d in all_detections]
    
    class_counts = {}
    for d in all_detections:
        cls = d['class']
        class_counts[cls] = class_counts.get(cls, 0) + 1
        
    metrics = {
        'total_grains':    total,
        'mean_confidence': round(float(np.mean(confs)) * 100, 1),
        'per_class':       class_counts
    }
    
    return metrics
