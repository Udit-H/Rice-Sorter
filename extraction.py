import cv2
import os


# Path to your videos directory inside Drive
videos_dir = "./rice_dataset_files"   # adjust to your folder
output_base = "./rice_dataset_files/original_frames"  # where frames will be saved
target_fps = 60

os.makedirs(output_base, exist_ok=True)

# Loop through all mp4 files in the directory
for video_file in os.listdir(videos_dir):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(videos_dir, video_file)
        output_folder = os.path.join(output_base, os.path.splitext(video_file)[0])
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_duration = 1 / video_fps
        target_duration = 1 / target_fps

        current_time = 0
        next_capture_time = 0
        saved_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if current_time >= next_capture_time:
                filename = f"{output_folder}/frame_{saved_id:05d}.jpg"
                cv2.imwrite(filename, frame)
                saved_id += 1
                next_capture_time += target_duration

            current_time += frame_duration

        cap.release()
        print(f"{video_file}: Saved {saved_id} frames into {output_folder}")
