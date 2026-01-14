import cv2
import numpy as np
import os
import sys
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==================== MODEL CONFIGURATION ====================
# EfficientNet configuration
IMAGE_SIZE = (224, 224)  # EfficientNet standard input size
NUM_CLASSES = 6
LABEL_MAP = {0: "black", 1: "brown", 2: "yellow", 3: "chalky", 4: "perfect", 5: "husk"}

# Model path - EfficientNet Rice Classifier
MODEL_PATH = '/home/rvce/Desktop/compiled/efficientnet_rice_final_inference.keras'

# Load model once at startup
print("Loading Keras model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ==================== PREPROCESSING FUNCTIONS ====================

def preprocess_grain_image(grain_crop):
    """
    Preprocess a grain crop for model prediction.
    Adjust this based on your transfer learning model's training preprocessing.
    
    Args:
        grain_crop (numpy array): BGR image of a single grain
        
    Returns:
        numpy array: Preprocessed image ready for model input
    """
    # Resize to model's expected input size
    img = cv2.resize(grain_crop, IMAGE_SIZE)
    
    # Convert BGR to RGB (TensorFlow models expect RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # EfficientNet preprocessing - scales to [-1, 1] range
    img = img.astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    return img


def classify_grain(grain_crop):
    """
    Classify a single grain crop using the loaded model.
    
    Args:
        grain_crop (numpy array): BGR image of a single grain
        
    Returns:
        tuple: (class_name, confidence)
    """
    img = preprocess_grain_image(grain_crop)
    img_batch = np.expand_dims(img, axis=0)  # Add batch dimension
    
    predictions = model.predict(img_batch, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    return LABEL_MAP[predicted_class], confidence


def load_and_preprocess_image_from_path(path):
    """
    Load and preprocess an image from disk for batch prediction.
    
    Args:
        path (str): Path to image file
        
    Returns:
        tensorflow tensor: Preprocessed image
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.cast(img, tf.float32)
    
    # Apply EfficientNet preprocessing
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    
    return img


def predict_images_from_folder(folder_path):
    """
    Predict classifications for all grain images in a folder.
    
    Args:
        folder_path (str): Path to folder containing grain images
        
    Returns:
        dict: Count of each class
    """
    class_counts = {label: 0 for label in LABEL_MAP.values()}
    
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    
    if not image_files:
        print(f"Warning: No PNG images found in {folder_path}")
        return class_counts
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = load_and_preprocess_image_from_path(img_path)
        img_expanded = tf.expand_dims(img, 0)
        
        predictions = model.predict(img_expanded, verbose=0)
        predicted_class = np.argmax(predictions)
        
        class_name = LABEL_MAP[predicted_class]
        class_counts[class_name] += 1
    
    return class_counts


# ==================== GRAIN DETECTION & CLASSIFICATION ====================

def detect_and_count_rice_grains_ml(original_image):
    """
    Detects rice grains using watershed segmentation and classifies them using ML model.
    Saves grain crops to 'grains' folder for batch prediction.
    
    Args:
        original_image (numpy array): Input image containing rice grains.
        
    Returns:
        tuple: (visualization_image, class_counts, percentage_list, broken_grain_count)
    """
    if original_image is None:
        raise ValueError("Could not read image")
    
    visualization_copy = original_image.copy()
    grains_dir = "grains"
    os.makedirs(grains_dir, exist_ok=True)
    
    img_counter = 1
    
    # Convert to HSV and grayscale
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    grayscale_image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
    
    # Thresholding to create a binary image
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean the image
    morphological_kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morphological_kernel, iterations=2)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, morphological_kernel, iterations=1)
    
    # Background extraction
    background = cv2.dilate(cleaned_image, morphological_kernel, iterations=2)
    
    # Distance transform for watershed preparation
    distance_transform = cv2.distanceTransform(cleaned_image, cv2.DIST_L2, 3)
    cv2.normalize(distance_transform, distance_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Foreground detection
    _, foreground = cv2.threshold(distance_transform, 0.3 * distance_transform.max(), 255, 0)
    foreground = np.uint8(foreground)
    
    # Unknown region identification
    unknown_region = cv2.subtract(background, foreground)
    
    # Connected components labeling
    _, markers = cv2.connectedComponents(foreground)
    markers += 1
    markers[unknown_region == 255] = 0
    
    # Watershed segmentation
    markers = cv2.watershed(original_image, markers)
    unique_markers = np.unique(markers)
    
    # Initialize counters
    broken_grain_count = 0
    broken_25_count = 0
    broken_50_count = 0
    broken_75_count = 0
    average_rice_area = 190
    
    # Region size for grain extraction
    region_size = 64  # Can be changed to match IMAGE_SIZE if needed
    half_size = region_size // 2
    img_h, img_w = original_image.shape[:2]
    
    # Process each grain
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
                
                area_ratio = area / average_rice_area
                
                # Classify broken rice by area
                if area_ratio <= 0.75:
                    broken_grain_count += 1
                    if area_ratio > 0.45:
                        broken_25_count += 1
                    elif area_ratio > 0.3:
                        broken_50_count += 1
                    else:
                        broken_75_count += 1
                    cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=2)
                    continue
                
                # Extract grain region for ML classification
                adj_cX = min(max(cX, half_size), img_w - half_size)
                adj_cY = min(max(cY, half_size), img_h - half_size)
                grain_region = cv2.getRectSubPix(
                    original_image, 
                    (region_size, region_size), 
                    (float(adj_cX), float(adj_cY))
                )
                
                # Create blue background and composite grain
                background_img = np.full((region_size, region_size, 3), (255, 0, 0), dtype=np.uint8)
                
                # Shift contour to local coordinates
                contour = contours[0].copy().astype(np.int32)
                contour[:, 0, 0] = contour[:, 0, 0] - (adj_cX - half_size)
                contour[:, 0, 1] = contour[:, 0, 1] - (adj_cY - half_size)
                
                mask = np.zeros((region_size, region_size), dtype=np.uint8)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                
                composed = background_img.copy()
                composed[mask == 255] = grain_region[mask == 255]
                
                # Save grain image for batch prediction
                cv2.imwrite(os.path.join(grains_dir, f"{img_counter}.png"), composed)
                img_counter += 1
    
    percentage_list = {
        "25%": broken_25_count,
        "50%": broken_50_count,
        "75%": broken_75_count
    }
    
    return visualization_copy, percentage_list, broken_grain_count


def detect_stones(image):
    """
    Detects stones in an image using HSV color space filtering.
    
    Args:
        image (numpy array): Input image containing potential stones.
        
    Returns:
        int: Stone count
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_stone_color = np.array([5, 50, 50])
    upper_stone_color = np.array([25, 255, 200])
    stone_mask = cv2.inRange(hsv_image, lower_stone_color, upper_stone_color)
    
    morphological_kernel = np.ones((3, 3), np.uint8)
    cleaned_stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, morphological_kernel, iterations=2)
    stone_contours, _ = cv2.findContours(cleaned_stone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    stone_count = 0
    
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
    
    return stone_count


def process_image(input_image):
    """
    Main processing function: detects grains, classifies using ML model, and returns results.
    
    Args:
        input_image (numpy array): Input image containing rice and potential impurities.
        
    Returns:
        tuple: (visualization_image, perfect_count, chalky_count, black_count, 
                yellow_count, brown_count, percentage_list, broken_grain_count, 
                stone_count, husk_count)
    """
    if input_image is None:
        raise ValueError("Could not read image")
    
    # Crop image (adjust as needed for your setup)
    input_image = input_image[:, 10:-10]
    cv2.imwrite('cropped_image.jpg', input_image)
    
    # Detect and segment grains
    visualization_copy, percentage_list, broken_grain_count = detect_and_count_rice_grains_ml(input_image)
    
    # Detect stones
    stone = detect_stones(input_image)
    
    # Classify grains using ML model
    grains_folder = "grains"
    if not os.path.exists(grains_folder):
        print(f"Warning: Folder not found: {grains_folder}")
        class_counts = {label: 0 for label in LABEL_MAP.values()}
    else:
        class_counts = predict_images_from_folder(grains_folder)
        shutil.rmtree(grains_folder)  # Clean up
    
    return (
        visualization_copy,
        class_counts["perfect"],
        class_counts["chalky"],
        class_counts["black"],
        class_counts["yellow"],
        class_counts["brown"],
        percentage_list,
        broken_grain_count,
        stone,
        class_counts["husk"]
    )


# ==================== TEST MAIN ====================

if __name__ == "__main__":
    # Test the function
    image_path = "/home/rvce/Desktop/compiled/static/captured/captured_1748464654.jpg"
    
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
    else:
        print(f"\nProcessing image...")
        print("-" * 50)
        
        visual, perfect_count, chalky_count, black_count, yellow_count, brown_count, \
        percentage_list, broken_grain_count, stone_count, husk_count = process_image(image)
        
        print("\nRice Analysis Results:")
        print("-" * 30)
        print(f"Perfect Rice: {perfect_count}")
        print(f"Chalky Rice: {chalky_count}")
        print(f"Black Rice: {black_count}")
        print(f"Yellow Rice: {yellow_count}")
        print(f"Brown Rice: {brown_count}")
        print(f"Husk: {husk_count}")
        print(f"Broken Rice: {broken_grain_count}")
        print(f"Stones: {stone_count}")
        
        print("\nBroken Rice Distribution:")
        print("-" * 30)
        for percentage, count in percentage_list.items():
            print(f"{percentage} broken: {count}")
        
        cv2.imwrite('processed_image.jpg', visual)
        print("\nProcessed image saved as 'processed_image.jpg'")
