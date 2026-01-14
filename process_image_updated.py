import cv2
import numpy as np
import os
import sys
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime

# ==================== MODEL CONFIGURATION ====================
# EfficientNet configuration
IMAGE_SIZE = (224, 224)  # EfficientNet standard input size
NUM_CLASSES = 6
LABEL_MAP = {0: "black", 1: "brown", 2: "yellow", 3: "chalky", 4: "perfect", 5: "husk"}

# Model path - EfficientNet Rice Classifier
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'efficientnet_rice_final_inference.keras')

# Load model once at startup
model = None
try:
    print("--------------------------------------------------")
    print("Initializing Rice Classification Model...")
    print(f"Model path: {MODEL_PATH}")
    
    if os.path.exists(MODEL_PATH):
        print(f"Path exists. Is directory: {os.path.isdir(MODEL_PATH)}")
        # Check files if it is a directory
        if os.path.isdir(MODEL_PATH):
            print(f"Directory contents: {os.listdir(MODEL_PATH)}")
    else:
        print("ERROR: Model path does not exist!")
        
    print("Calling load_model...")
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
    print("--------------------------------------------------")
except Exception as e:
    import traceback
    print("CRITICAL ERROR: Failed to load model!")
    print(f"Error details: {e}")
    traceback.print_exc()
    print("--------------------------------------------------")

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
    
    if model is None:
        print("Error: Model not loaded, returning 'unknown'")
        return "unknown", 0.0

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
        
        if model is None:
            print("Error: Model not loaded, cannot predict.")
            continue

        predictions = model.predict(img_expanded, verbose=0)
        predicted_class = np.argmax(predictions)
        
        class_name = LABEL_MAP[predicted_class]
        class_counts[class_name] += 1
    
    return class_counts


# ==================== GRAIN DETECTION & CLASSIFICATION ====================

def detect_and_count_rice_grains_ml(original_image):
    """
    Detects rice grains using HSV color masking (Blue Background) and classifies them using ML model.
    Saves grain crops to 'grains' folder for batch prediction.
    
    Args:
        original_image (numpy array): Input image containing rice grains.
        
    Returns:
        tuple: (visualization_image, percentage_list, broken_grain_count)
    """
    if original_image is None:
        raise ValueError("Could not read image")
    
    visualization_copy = original_image.copy()
    grains_dir = "grains"
    if os.path.exists(grains_dir):
        shutil.rmtree(grains_dir)
    os.makedirs(grains_dir, exist_ok=True)
    
    img_counter = 1
    
    # Constants for Rice Segmentation (Adapted from Dal Logic)
    MIN_GRAIN_AREA = 50   # Minimum area to be considered a grain
    
    # Predefined Blue Background Range (Adjust if needed)
    LOWER_BLUE = np.array([100, 100, 100])
    UPPER_BLUE = np.array([140, 255, 255])
    
    # Convert to HSV and apply blue mask
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    
    # Invert mask: We want what is NOT blue (the grains)
    grain_mask = cv2.bitwise_not(blue_mask)
    
    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"DEBUG: Found {len(contours)} contours.")
    
    # Initialize counters for broken classification (heuristic based)
    broken_grain_count = 0
    broken_25_count = 0
    broken_50_count = 0
    broken_75_count = 0
    
    average_rice_area = 190  # Reference for broken calculation
    
    region_size = 64
    half_size = region_size // 2
    img_h, img_w = original_image.shape[:2]
    
    # Process each grain
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter small noise
        if area < MIN_GRAIN_AREA:
            continue
            
        # Optional: Filter very large blobs that might be clusters (or handle them later)
        # if area > 2000: continue 

        # --- 1. Draw Contour on Visualization ---
        cv2.drawContours(visualization_copy, [contour], -1, (0, 255, 0), 2)
        
        # --- 2. Calculate Heuristics for Broken Rice (Optional parallel check) ---
        area_ratio = area / average_rice_area
        if area_ratio <= 0.75:
            broken_grain_count += 1
            if area_ratio > 0.45:
                broken_25_count += 1
            elif area_ratio > 0.3:
                broken_50_count += 1
            else:
                broken_75_count += 1
            # Mark broken in Red
            cv2.drawContours(visualization_copy, [contour], -1, (0, 0, 255), 2)
        
        # --- 3. Extract Grain for ML Classification ---
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Extract square region centered on the grain
            adj_cX = min(max(cX, half_size), img_w - half_size)
            adj_cY = min(max(cY, half_size), img_h - half_size)
            
            grain_region = cv2.getRectSubPix(
                original_image, 
                (region_size, region_size), 
                (float(adj_cX), float(adj_cY))
            )
            
            # Create cleaner synthetic background for the crop
            # We want the grain isolated on a blue/black background for consistent inference
            # 1. Create a mask just for this contour in the local 64x64 region
            mask_local = np.zeros((region_size, region_size), dtype=np.uint8)
            
            # We need to shift the contour to the local coordinates of the 64x64 crop
            contour_shifted = contour - [adj_cX - half_size, adj_cY - half_size]
            
            # Draw the filled contour on the local mask
            cv2.drawContours(mask_local, [contour_shifted], -1, 255, thickness=cv2.FILLED)
            
            # Create a blue background image
            background_img = np.full((region_size, region_size, 3), (255, 0, 0), dtype=np.uint8)
            
            # Combine: Place the masked grain onto the blue background
            composed = background_img.copy()
            # Only copy pixels where the mask is active
            composed[mask_local == 255] = grain_region[mask_local == 255]
            
            # Save for batch prediction
            cv2.imwrite(os.path.join(grains_dir, f"{img_counter}.png"), composed)
            img_counter += 1

    # Write debug log
    with open("debug_log.txt", "w") as f:
        f.write(f"Timestamp: {datetime.datetime.now()}\n")
        f.write(f"Contours found: {len(contours)}\n")
        f.write(f"Model Status: {'Loaded' if model else 'Not Loaded'}\n")
        if model:
            f.write("Prediction Mode: ML Model (EfficientNet)\n")
        else:
            f.write("Prediction Mode: FALLBACK (Model failed)\n")

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
