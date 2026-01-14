import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Constants
IMAGE_SIZE = (64, 64)
NUM_CLASSES = 6
LABEL_MAP = {0: "black", 1: "brown", 2: "yellow", 3: "chalky", 4: "perfect", 5: "husk"}

# Load model
MODEL_PATH = '/home/rvce/Desktop/compiled/o4mnew.keras' 
model = load_model(MODEL_PATH)

# Warmup model with dummy input
_ = model.predict(tf.random.uniform((1, 64, 64, 3)), verbose=0)


def detect_stones(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_stone_color = np.array([5, 50, 50])
    upper_stone_color = np.array([25, 255, 200])
    stone_mask = cv2.inRange(hsv_image, lower_stone_color, upper_stone_color)
    morphological_kernel = np.ones((3, 3), np.uint8)
    cleaned_stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, morphological_kernel, iterations=2)
    stone_contours, _ = cv2.findContours(cleaned_stone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    stone_count = 0
    total_stone_area = 0
    result_image = image.copy()

    for contour in stone_contours:
        area = cv2.contourArea(contour)
        if area > 20:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis, minor_axis = max(axes), min(axes)
                aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
                if 1.0 <= aspect_ratio <= 2.0:
                    stone_count += 1
                    total_stone_area += area
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
                    dilated_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
                    result_image[dilated_mask == 255] = (255, 255, 0)

    return stone_count


def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    img = img * 0.72
    img = tf.image.adjust_contrast(img, 1.04)
    img = tf.image.adjust_saturation(img, 0.54)
    img = tf.clip_by_value(img, 0.0, 1.0)
    return img


def predict_images_from_folder(folder_path):
    class_counts = {label: 0 for label in LABEL_MAP.values()}
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = load_and_preprocess_image(img_path)
        img_expanded = tf.expand_dims(img, 0)
        predictions = model.predict(img_expanded, verbose=0)
        predicted_class = np.argmax(predictions)
        confidence = predictions[0][predicted_class] * 100
        class_name = LABEL_MAP[predicted_class]
        class_counts[class_name] += 1
    return class_counts


def detect_and_count_rice_grains(original_image):
    if original_image is None:
        raise ValueError("Could not read image")

    visualization_copy = original_image.copy()

    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    grayscale_image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    morphological_kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morphological_kernel, iterations=2)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, morphological_kernel, iterations=1)

    background = cv2.dilate(cleaned_image, morphological_kernel, iterations=2)
    distance_transform = cv2.distanceTransform(cleaned_image, cv2.DIST_L2, 3)
    cv2.normalize(distance_transform, distance_transform, 0, 1.0, cv2.NORM_MINMAX)
    _, foreground = cv2.threshold(distance_transform, 0.2 * distance_transform.max(), 255, 0)
    foreground = np.uint8(foreground)
    unknown_region = cv2.subtract(background, foreground)

    _, markers = cv2.connectedComponents(foreground)
    markers += 1
    markers[unknown_region == 255] = 0
    markers = cv2.watershed(original_image, markers)
    unique_markers = np.unique(markers)

    broken_grain_count = 0
    broken_25_count = 0
    broken_50_count = 0
    broken_75_count = 0
    merge_count = 0

    region_size = 64
    half_size = region_size // 2
    img_h, img_w = original_image.shape[:2]

    for label in unique_markers:
        if label <= 1:
            continue
        grain_mask = np.zeros(grayscale_image.shape, dtype="uint8")
        grain_mask[markers == label] = 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            area = cv2.contourArea(contours[0])
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                contour_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, contours, -1, 1, thickness=cv2.FILLED)
                masked_pixels = original_image[contour_mask == 1]
                sorted_bgr = masked_pixels[np.lexsort((masked_pixels[:, 2], masked_pixels[:, 1], masked_pixels[:, 0]))]
                masked_pixels = sorted_bgr[5:-5]
                count_for_brown = np.sum(
                    (np.all(masked_pixels >= [107, 64, 81], axis=1) &
                     np.all(masked_pixels <= [182, 141, 147], axis=1)))

                if area < 160:
                    broken_grain_count += 1
                    if 60 < area < 100:
                        broken_50_count += 1
                    elif area <= 60:
                        broken_25_count += 1
                    else:
                        broken_75_count += 1
                    cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=cv2.FILLED)  # Red
                    continue
                elif area > 280 and count_for_brown <= 5:
                    merge_count += 1
                    cv2.drawContours(visualization_copy, contours, -1, (128, 128, 0), thickness=cv2.FILLED)  # Grey
                    continue
                else:
                    cv2.drawContours(visualization_copy, contours, -1, (0, 255, 0), thickness=1)  # Green border

    return visualization_copy


# ========== MAIN SECTION ==========
if __name__ == "__main__":
    test_image_path = "/home/rvce/Desktop/compiled/static/captured/captured_1748480031.jpg"  # Replace with a real image path
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)
        print("[INFO] Detecting stones...")
        stone_count = detect_stones(img)
        print("Stones detected:", stone_count)

        print("[INFO] Detecting and analyzing rice grains...")
        output = detect_and_count_rice_grains(img)
        print(output)
    else:
        print(f"[ERROR] Test image not found: {test_image_path}")
