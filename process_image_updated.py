import cv2
import numpy as np
import os
import sys
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
import datetime

# ==================== MODEL CONFIGURATION ====================
# Model configuration - MUST match training!
# Model has 5 classes based on output shape (1, 5)
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 5
LABEL_MAP = {0: "black", 1: "brown", 2: "chalky", 3: "yellow", 4: "perfect"}

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
    MUST MATCH TRAINING PREPROCESSING EXACTLY!
    
    Args:
        path (str): Path to image file
        
    Returns:
        tensorflow tensor: Preprocessed image
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    # CRITICAL: Training used simple float32 cast, NOT EfficientNet preprocessing!
    # The training notebook did: grain_resized.astype(np.float32)
    # NO normalization to [-1, 1] range!
    img = tf.cast(img, tf.float32)
    
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
    
    # Debug: Print first prediction details
    first_prediction_logged = False
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        img = load_and_preprocess_image_from_path(img_path)
        img_expanded = tf.expand_dims(img, 0)
        
        if model is None:
            print("Error: Model not loaded, cannot predict.")
            continue

        predictions = model.predict(img_expanded, verbose=0)
        predicted_class = np.argmax(predictions)
        
        # Debug: Log first prediction details
        if not first_prediction_logged:
            print(f"DEBUG: Model output shape: {predictions.shape}")
            print(f"DEBUG: First prediction raw output: {predictions[0]}")
            print(f"DEBUG: Predicted class index: {predicted_class}")
            print(f"DEBUG: Available LABEL_MAP keys: {list(LABEL_MAP.keys())}")
            first_prediction_logged = True
        
        # Handle unknown class indices gracefully
        if predicted_class in LABEL_MAP:
            class_name = LABEL_MAP[predicted_class]
        else:
            print(f"WARNING: Unknown class index {predicted_class}, mapping to 'unknown'")
            class_name = "unknown"
            if "unknown" not in class_counts:
                class_counts["unknown"] = 0
        
        class_counts[class_name] += 1
    
    return class_counts

# ==================== GRAIN DETECTION & CLASSIFICATION ====================

# def is_chalky_by_color(grain_crop):
#     """
#     Heuristic check: Chalky rice should be predominantly white/very light.
#     Returns True if grain appears chalky based on color analysis.
#     """
#     # Convert to HSV for better color analysis
#     hsv = cv2.cvtColor(grain_crop, cv2.COLOR_BGR2HSV)
    
#     # Chalky rice characteristics:
#     # - High Value (brightness) > 200
#     # - Low Saturation < 30
#     v_channel = hsv[:, :, 2]
#     s_channel = hsv[:, :, 1]
    
#     avg_value = np.mean(v_channel)
#     avg_saturation = np.mean(s_channel)
    
#     # Chalky: bright and unsaturated
#     return avg_value > 200 and avg_saturation < 30

def detect_and_count_rice_grains_ml(original_image):
    """
    Detects rice grains using HSV color masking (excludes blue background).
    Saves grain crops to 'grains' folder with classification in filename.
    
    Args:
        original_image (numpy array): Input image containing rice grains.
        
    Returns:
        tuple: (visualization_image, percentage_list, broken_grain_count, class_counts)
    """
    if original_image is None:
        raise ValueError("Could not read image")
    
    visualization_copy = original_image.copy()
    grains_dir = "grains"
    
    if os.path.exists(grains_dir):
        for file in os.listdir(grains_dir):
            file_path = os.path.join(grains_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(grains_dir, exist_ok=True)
    
    img_counter = 1
    
    # Initialize class counts
    class_counts = {label: 0 for label in LABEL_MAP.values()}
    
    # Constants for Rice Segmentation
    MIN_GRAIN_AREA = 50  # Minimum area for a valid grain
    MAX_GRAIN_AREA = 750  # Maximum area to filter out large noise
    
    # ==================== HSV-BASED SEGMENTATION (EXCLUDE BLUE) ====================
    # Convert to HSV color space
    hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # DEBUG: Sample background HSV values
    h, w = hsv.shape[:2]
    corners = [(10, 10), (10, w-10), (h-10, 10), (h-10, w-10)]
    print("DEBUG: HSV values at corners (background):")
    for y, x in corners:
        print(f"  ({y},{x}): H={hsv[y,x,0]}, S={hsv[y,x,1]}, V={hsv[y,x,2]}")
    
    # STEP 1: Create mask to EXCLUDE blue background
    # Blue background: H=100-130 (blue hue), high saturation
    LOWER_BLUE = np.array([100, 100, 50])
    UPPER_BLUE = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    cv2.imwrite('debug_blue_mask.jpg', blue_mask)
    
    # Invert to get non-blue regions (potential grains)
    non_blue_mask = cv2.bitwise_not(blue_mask)
    cv2.imwrite('debug_non_blue_mask.jpg', non_blue_mask)
    
    # STEP 2: Also exclude very dark regions (shadows, noise)
    # V channel < 50 is too dark
    v_channel = hsv[:, :, 2]
    bright_mask = (v_channel > 50).astype(np.uint8) * 255
    cv2.imwrite('debug_bright_mask.jpg', bright_mask)
    
    # STEP 3: Combine masks - grain must be non-blue AND bright enough
    grain_mask = cv2.bitwise_and(non_blue_mask, bright_mask)
    cv2.imwrite('debug_combined_mask.jpg', grain_mask)
    
    # ==================== MORPHOLOGICAL CLEANUP ====================
    # Opening: remove small noise spots
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    
    # Closing: fill small holes inside grains
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    grain_mask = cv2.morphologyEx(grain_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # DEBUG: Save cleaned mask
    cv2.imwrite('debug_grain_mask_cleaned.jpg', grain_mask)
    
    # ==================== FIND CONTOURS ====================
    contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"DEBUG: Found {len(contours)} raw contours.")
    
    # ==================== CALCULATE DYNAMIC AVERAGE AREA ====================
    all_areas = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if MIN_GRAIN_AREA <= area <= MAX_GRAIN_AREA:
            all_areas.append(area)
    
    if len(all_areas) == 0:
        print("DEBUG: No valid grains found!")
        return visualization_copy, {"25%": 0, "50%": 0, "75%": 0}, 0, class_counts
    
    # Calculate average from top 25% largest grains (these are full grains)
    sorted_areas = sorted(all_areas, reverse=True)
    top_25_percent = sorted_areas[:max(1, len(sorted_areas) // 4)]
    average_rice_area = np.mean(top_25_percent)
    
    print(f"DEBUG: Valid grain count: {len(all_areas)}")
    print(f"DEBUG: Area range: {min(all_areas):.0f} - {max(all_areas):.0f}")
    print(f"DEBUG: Average rice area (top 25%): {average_rice_area:.0f}")
    
    # ==================== PROCESS EACH GRAIN ====================
    broken_grain_count = 0
    broken_25_count = 0
    broken_50_count = 0
    broken_75_count = 0
    
    region_size = 64
    half_size = region_size // 2
    img_h, img_w = original_image.shape[:2]
    
    valid_grain_count = 0
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        # Filter by area
        if area < MIN_GRAIN_AREA or area > MAX_GRAIN_AREA:
            continue
        
        valid_grain_count += 1
        
        # Calculate area ratio for broken detection
        area_ratio = area / average_rice_area
        
        # Broken rice: less than 75% of average
        if area_ratio <= 0.75:
            broken_grain_count += 1
            if area_ratio > 0.5:
                broken_25_count += 1
            elif area_ratio > 0.25:
                broken_50_count += 1
            else:
                broken_75_count += 1
            # Mark broken in Red
            cv2.drawContours(visualization_copy, [contour], -1, (0, 0, 255), 2)
            continue
        
        # Full grain - Draw in Green
        cv2.drawContours(visualization_copy, [contour], -1, (0, 255, 0), 2)
        
        # Extract Grain for ML Classification
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
            
            # Create mask for this specific grain
            mask_local = np.zeros((region_size, region_size), dtype=np.uint8)
            contour_shifted = contour - [adj_cX - half_size, adj_cY - half_size]
            cv2.drawContours(mask_local, [contour_shifted], -1, 255, thickness=cv2.FILLED)
            
            # Blue background to match training data
            background_img = np.full((region_size, region_size, 3), (255, 0, 0), dtype=np.uint8)
            composed = background_img.copy()
            composed[mask_local == 255] = grain_region[mask_local == 255]
            
            # Classify the grain using ML model ONLY (no heuristic override)
            if model is not None:
                class_name, confidence = classify_grain(composed)
                class_counts[class_name] += 1
            else:
                class_name = "unknown"
                if "unknown" not in class_counts:
                    class_counts["unknown"] = 0
                class_counts["unknown"] += 1
            
            # Save with classification in filename
            filename = f"{img_counter}_{class_name}.png"
            cv2.imwrite(os.path.join(grains_dir, filename), composed)
            img_counter += 1
    
    print(f"DEBUG: Valid grains after filtering: {valid_grain_count}")
    print(f"DEBUG: Full grains classified: {sum(class_counts.values())}")
    print(f"DEBUG: Broken grains: {broken_grain_count}")

    percentage_list = {
        "25%": broken_25_count,
        "50%": broken_50_count,
        "75%": broken_75_count
    }
    
    return visualization_copy, percentage_list, broken_grain_count, class_counts


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
    
    # Crop image
    input_image = input_image[:, 10:-10]
    cv2.imwrite('cropped_image.jpg', input_image)
    
    # Detect, segment, and classify grains
    visualization_copy, percentage_list, broken_grain_count, class_counts = detect_and_count_rice_grains_ml(input_image)
    
    # Detect stones
    stone = detect_stones(input_image)
    
    # Print grain images info
    grains_folder = "grains"
    if os.path.exists(grains_folder):
        print(f"\nGrain images saved in: {os.path.abspath(grains_folder)}")
        print(f"Total grain images: {len([f for f in os.listdir(grains_folder) if f.endswith('.png')])}")
    
    # Calculate total from ML classifications
    ml_total = sum(class_counts.values())
    
    return (
        visualization_copy,
        ml_total,
        class_counts.get("chalky", 0),
        class_counts.get("black", 0),
        class_counts.get("yellow", 0),
        class_counts.get("brown", 0),
        percentage_list,
        broken_grain_count,
        stone,
        0
    )

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


# def process_image(input_image):
#     """
#     Main processing function: detects grains, classifies using ML model, and returns results.
    
#     Args:
#         input_image (numpy array): Input image containing rice and potential impurities.
        
#     Returns:
#         tuple: (visualization_image, perfect_count, chalky_count, black_count, 
#                 yellow_count, brown_count, percentage_list, broken_grain_count, 
#                 stone_count, husk_count)
#     """
#     if input_image is None:
#         raise ValueError("Could not read image")
    
#     # Crop image (adjust as needed for your setup)
#     input_image = input_image[:, 10:-10]
#     cv2.imwrite('cropped_image.jpg', input_image)
    
#     # Detect and segment grains
#     visualization_copy, percentage_list, broken_grain_count = detect_and_count_rice_grains_ml(input_image)
    
#     # Detect stones
#     stone = detect_stones(input_image)
    
#     # Classify grains using ML model
#     grains_folder = "grains"
#     if not os.path.exists(grains_folder):
#         print(f"Warning: Folder not found: {grains_folder}")
#         class_counts = {label: 0 for label in LABEL_MAP.values()}
#     else:
#         class_counts = predict_images_from_folder(grains_folder)
#         print(f"\nGrain images saved in: {os.path.abspath(grains_folder)}")
#         print(f"Total grain images: {len([f for f in os.listdir(grains_folder) if f.endswith('.png')])}")
    
    
#     # Calculate total from ML classifications
#     ml_total = sum(class_counts.values())
    
#     return (
#         visualization_copy,
#         ml_total,  # full_grain_count = total classified by ML (not broken)
#         class_counts["chalky"],
#         class_counts["black"],
#         class_counts["yellow"],
#         class_counts["brown"],
#         percentage_list,
#         broken_grain_count,
#         stone,
#         0  # husk_count = 0 (model not trained on husk)
#     )


# ==================== TEST MAIN ====================

if __name__ == "__main__":
    # Test the function
    image_path = r"E:\elsem3\new_ricefromdevice\Rice-Sorter\static\captured\captured_1749462409.jpg"
    
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