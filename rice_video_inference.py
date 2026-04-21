"""
Real-Time Rice Grain Video Inference
======================================
Combines OpenCV contour-based detection with the RiceNet-v2 CNN classifier
to run frame-by-frame inference on a rice machine video.

Outputs:
  - Annotated video with bounding boxes + labels
  - JSON report with per-frame and aggregate results

Requirements:
    pip install opencv-python torch torchvision numpy

Usage:
    # With trained model weights
    python rice_video_inference.py --video input.mp4 --weights ricenet_v2.pth

    # Without weights (uses feature-based simulation)
    python rice_video_inference.py --video input.mp4 --simulate
"""

import cv2
import numpy as np
import torch
import argparse
import os
import json
import time

from rice_model import RiceNetV2, get_transforms, CLASS_NAMES, load_model
from rice_grain_analysis import (
    preprocess_frame, detect_grains, annotate_frame,
    compute_accuracy_metrics, classify_grain
)
from PIL import Image

# ──────────────────────────────────────────────
# CNN-BASED CLASSIFIER (when model weights available)
# ──────────────────────────────────────────────

def classify_grain_cnn(grain, bgr_frame, model, transform, device):
    """
    Crop the grain ROI from the frame, run it through the trained CNN,
    and return (class_name, confidence).
    """
    x, y, w, h = grain['x'], grain['y'], grain['w'], grain['h']
    # Add padding around grain crop
    pad = 8
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(bgr_frame.shape[1], x + w + pad), min(bgr_frame.shape[0], y + h + pad)
    roi_bgr = bgr_frame[y1:y2, x1:x2]

    if roi_bgr.size == 0:
        return 'Healthy', 0.80

    # Convert BGR (OpenCV) → RGB (PIL) → tensor
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    tensor  = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx  = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx])


# ──────────────────────────────────────────────
# VIDEO WRITER SETUP
# ──────────────────────────────────────────────

def create_video_writer(input_path, output_path, fps, width, height):
    """Create an annotated output video writer."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return writer


# ──────────────────────────────────────────────
# OVERLAY: HUD drawn on each frame
# ──────────────────────────────────────────────

def draw_hud(frame, frame_idx, fps, grain_count, stats):
    """Draw a heads-up display with running stats on the frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Top bar background
    cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    t_sec = frame_idx / fps
    cv2.putText(frame, f"RiceNet-v2  |  t={t_sec:.2f}s  frame={frame_idx}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 1)
    cv2.putText(frame, f"Grains this frame: {grain_count}   "
                f"Total: {stats['total']}   "
                f"Defects: {stats['defects']}   "
                f"Conf: {stats['avg_conf']:.1f}%",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)


# ──────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ──────────────────────────────────────────────

def run_inference(video_path, weights_path=None, simulate=False,
                  output_dir='./output', frame_step=1, show_preview=False):
    """
    Run full inference pipeline on the video.

    Parameters
    ----------
    video_path   : path to input .mp4
    weights_path : path to ricenet_v2.pth (None → simulate)
    simulate     : use feature-based fallback instead of CNN
    output_dir   : where to save results
    frame_step   : process every Nth frame (1 = all frames)
    show_preview : show live OpenCV window (requires display)
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Load video ──
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration     = total_frames / fps

    print(f"[Inference] {os.path.basename(video_path)}  "
          f"{W}x{H} @ {fps:.0f}fps  {duration:.1f}s")

    # ── Load model (optional) ──
    model, device, transform = None, None, None
    if not simulate and weights_path and os.path.exists(weights_path):
        model, device = load_model(weights_path)
        transform = get_transforms('val')
        print(f"[Inference] Using trained CNN: {weights_path}")
    else:
        print("[Inference] No weights — using feature-based simulation")

    # ── Video writer ──
    out_video_path = os.path.join(output_dir, 'annotated_output.mp4')
    writer = create_video_writer(video_path, out_video_path, fps, W, H)

    # ── Run ──
    all_detections = []
    frame_results  = []
    running_stats  = {'total': 0, 'defects': 0, 'confs': []}
    t_start        = time.time()
    frame_idx      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            gray, mask = preprocess_frame(frame)
            grains     = detect_grains(mask)
            frame_grains = []

            for grain in grains:
                if model is not None:
                    cls, conf = classify_grain_cnn(grain, frame, model, transform, device)
                else:
                    cls, conf = classify_grain(grain, gray)

                det = {
                    'frame':      frame_idx,
                    'time_sec':   round(frame_idx / fps, 3),
                    'class':      cls,
                    'confidence': round(conf, 4),
                    'area_px2':   grain['area'],
                    'aspect':     round(grain['aspect'], 2),
                    'bbox':       [grain['x'], grain['y'], grain['w'], grain['h']],
                }
                all_detections.append(det)
                frame_grains.append(det)

                # Track running stats
                running_stats['total'] += 1
                running_stats['confs'].append(conf)
                if cls in ('Broken', 'Chalky', 'Discolored', 'Immature'):
                    running_stats['defects'] += 1

                # Draw bounding box on frame
                annotate_frame(frame, grain, cls, conf)

            avg_conf = (np.mean(running_stats['confs']) * 100
                        if running_stats['confs'] else 0.0)
            draw_hud(frame, frame_idx, fps, len(frame_grains),
                     {**running_stats, 'avg_conf': avg_conf})

            frame_results.append({
                'frame': frame_idx,
                'time':  round(frame_idx / fps, 2),
                'count': len(frame_grains),
            })

        writer.write(frame)

        if show_preview:
            cv2.imshow('Rice Grain Analysis', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[Inference] User stopped early.")
                break

        frame_idx += 1

        if frame_idx % 300 == 0:
            elapsed = time.time() - t_start
            print(f"  [frame {frame_idx}/{total_frames}]  "
                  f"total_grains={running_stats['total']}  "
                  f"elapsed={elapsed:.1f}s")

    cap.release()
    writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    # ── Aggregate ──
    from rice_grain_analysis import compute_accuracy_metrics
    metrics  = compute_accuracy_metrics(all_detections)
    elapsed  = time.time() - t_start

    print(f"\n[Inference] Done in {elapsed:.1f}s")
    print(f"  Total grains    : {metrics.get('total_grains', 0)}")
    print(f"  Mean confidence : {metrics.get('mean_confidence', 0):.1f}%")
    print(f"  Output video    : {out_video_path}")

    # ── Save report ──
    report = {
        'video':         os.path.basename(video_path),
        'resolution':    f"{W}x{H}",
        'fps':           fps,
        'duration_sec':  round(duration, 2),
        'frame_step':    frame_step,
        'model':         weights_path or 'simulation',
        'metrics':       metrics,
        'frame_results': frame_results,
        'detections':    all_detections,
    }
    report_path = os.path.join(output_dir, 'inference_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report     : {report_path}")

    return report


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-Time Rice Grain Video Inference')
    parser.add_argument('--video',    required=True, help='Input video path')
    parser.add_argument('--weights',  default=None,  help='Path to ricenet_v2.pth')
    parser.add_argument('--simulate', action='store_true',
                        help='Use feature-based simulation (no model weights needed)')
    parser.add_argument('--output',   default='./output', help='Output directory')
    parser.add_argument('--step',     type=int, default=1,
                        help='Process every Nth frame (1=all, 5=fast preview)')
    parser.add_argument('--preview',  action='store_true',
                        help='Show live OpenCV window (requires display)')
    args = parser.parse_args()

    run_inference(
        video_path   = args.video,
        weights_path = args.weights,
        simulate     = args.simulate,
        output_dir   = args.output,
        frame_step   = args.step,
        show_preview = args.preview,
    )
