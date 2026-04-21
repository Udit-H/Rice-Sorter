import subprocess
import os
import json

# Correct paths relative to the root rice directory
python_exe = os.path.join(os.getcwd(), '..', 'venv', 'bin', 'python')
inference_script = os.path.join(os.getcwd(), '..', 'rice_video_inference.py')

videos = [
    ('../rice_videos/test_1_50.mp4', 50),
    ('../rice_videos/test_2_50.mp4', 50),
    ('../rice_videos/test_3_100.mp4', 100),
    ('../rice_videos/test_4_100.mp4', 100),
    ('../rice_videos/test_5_100.mp4', 100),
    ('../rice_videos/test_6_100.mp4', 100),
    ('../rice_videos/test_7_200.mp4', 200),
    ('../rice_videos/test_7_200_original.mp4', 200),
    ('../New_Dataset_rice_sorter/black_rice.mp4', 54),
    ('../New_Dataset_rice_sorter/test_set_1.mp4', 100),
]

results = []

for video_path, ground_truth in videos:
    video_abs = os.path.abspath(video_path)
    if not os.path.exists(video_abs):
        print(f"Skipping {video_path} (not found)")
        continue
        
    print(f"Running on {os.path.basename(video_path)}...")
    output_dir = os.path.join(os.getcwd(), f"output_{os.path.basename(video_path).split('.')[0]}")
    cmd = [
        python_exe, inference_script,
        '--video', video_abs,
        '--simulate',
        '--output', output_dir
    ]
    # Run from the root directory so imports work
    subprocess.run(cmd, cwd=os.path.join(os.getcwd(), '..'), capture_output=True)
    
    report_path = os.path.join(output_dir, 'inference_report.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            data = json.load(f)
            detected = data['metrics']['total_grains']
            accuracy = (detected / ground_truth) * 100 if ground_truth > 0 else 0
            results.append({
                'video': os.path.basename(video_path),
                'ground_truth': ground_truth,
                'detected': detected,
                'accuracy': round(accuracy, 2)
            })
            print(f"  Done. Detected: {detected} / {ground_truth} ({accuracy:.2f}%)")
    else:
        print(f"  Failed to find report for {video_path}")

with open('all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nAll runs complete. Summary:")
print(f"{'Video':30s} | {'GT':3s} | {'Det':3s} | {'Acc':7s}")
print("-" * 50)
for r in results:
    print(f"{r['video']:30s} | {r['ground_truth']:3d} | {r['detected']:3d} | {r['accuracy']:6.2f}%")
