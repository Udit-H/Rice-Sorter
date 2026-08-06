import cv2
import os

def resize_images(input_dir, frame_64, frame_128):

    
    # Create output directories if they don't exist
    os.makedirs(frame_64, exist_ok=True)
    os.makedirs(frame_128, exist_ok=True)

    # Loop through all files in input_dir
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}. Skipping.")
            continue

        # Resize to 64x64
        resized_64 = cv2.resize(img, (64, 64))
        cv2.imwrite(os.path.join(frame_64, img_name), resized_64)

        # Resize to 128x128
        resized_128 = cv2.resize(img, (128, 128))
        cv2.imwrite(os.path.join(frame_128, img_name), resized_128)

    print("Resizing complete.")

# Example usage
resize_images(
    input_dir="frame_natural_Size",
    frame_64="dataset_moving/resized_64",
    frame_128="dataset_moving/resized_128"
)
