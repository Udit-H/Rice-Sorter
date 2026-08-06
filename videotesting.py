import cv2
import os
def extract_frames(video_path, frame_natural_Size, target_fps=60):
    if not os.path.exists(frame_natural_Size):
        os.makedirs(frame_natural_Size)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(round(video_fps / target_fps)) if video_fps > 0 else 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(frame_natural_Size, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
        frame_count += 1
    

    #resize to 64x64
    '''
    if not os.path.exists("frame_64x64"):
        os.makedirs("frame_64x64")
    for frame_file in os.listdir(frame_natural_Size):
        frame_path = os.path.join(frame_natural_Size, frame_file)
        img = cv2.imread(frame_path)
        resized_img = cv2.resize(img, (64, 64))
        resized_frame_filename = os.path.join("frame_64x64", frame_file)
        cv2.imwrite(resized_frame_filename, resized_img)
    
    #resize to 128x128
    if not os.path.exists("frame_128x128"):
        os.makedirs("frame_128x128")
    for frame_file in os.listdir(frame_natural_Size):
        frame_path = os.path.join(frame_natural_Size, frame_file)
        img = cv2.imread(frame_path)
        resized_img = cv2.resize(img, (128, 128))
        resized_frame_filename = os.path.join("frame_128x128", frame_file)
        cv2.imwrite(resized_frame_filename, resized_img)
    cap.release()
'''
video_path = "long.mp4" #add it in your vscode
extract_frames(video_path, "frame_natural_Size", target_fps=60 )
