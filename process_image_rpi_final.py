"""
RPi-Optimized Rice Image Processing Module
Uses EfficientNet for fast ML inference on Raspberry Pi
"""
import cv2
import numpy as np
import os
import sys
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==================== REDUCE TENSORFLOW LOGGING ====================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ==================== MODEL CONFIGURATION ====================
# EfficientNet Rice Classifier
IMAGE_SIZE = (224, 224)
NUM_CLASSES = 6
LABEL_MAP = {0: "black", 1: "brown", 2: "yellow", 3: "chalky", 4: "perfect", 5: "husk"}
MODEL_PATH = '/home/rvce/Desktop/compiled/efficientnet_rice_final_inference.keras'

# Global model variable (load once)
model = None

def load_model_once():
    """Load model once at startup to avoid memory issues on RPi"""
    global model
    if model is None:
        try:
            print("Loading EfficientNet model...")
            model = load_model(MODEL_PATH)
            print("✓ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    return True

# ==================== PREPROCESSING FUNCTIONS ====================

def preprocess_grain_image(grain_crop):
    """
    Preprocess grain crop for EfficientNet prediction.
    Optimized for RPi performance.
    
    Args:
        grain_crop (numpy array): BGR image of a single grain
        
    Returns:
        numpy array: Preprocessed image ready for inference
    """
    try:
        # Resize to EfficientNet input size
        img = cv2.resize(grain_crop, IMAGE_SIZE)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float and apply EfficientNet preprocessing
        img = img.astype(np.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        
        return img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None


def classify_grain(grain_crop):
    """
    Classify a single grain using the loaded model.
    
    Args:
        grain_crop (numpy array): BGR image of grain
        
    Returns:
        tuple: (class_name, confidence) or (None, 0) on error
    """
    if grain_crop is None or grain_crop.size == 0:
        return None, 0
    
    try:
        img = preprocess_grain_image(grain_crop)
        if img is None:
            return None, 0
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = model.predict(img_batch, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        class_name = LABEL_MAP.get(predicted_class, "unknown")
        return class_name, confidence
    except Exception as e:
        print(f"Error classifying grain: {str(e)}")
        return None, 0


def load_and_preprocess_image_from_path(path):
    """
    Load and preprocess image from disk for batch prediction.
    
    Args:
        path (str): Path to PNG image file
        
    Returns:
        numpy array: Preprocessed image or None on error
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        
        # Resize and convert
        img = cv2.resize(img, IMAGE_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        img = img.astype(np.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        
        return img
    except Exception as e:
        print(f"Error loading/preprocessing image from {path}: {str(e)}")
        return None


def predict_images_from_folder(folder_path):
    """
    Batch predict all grain images in a folder.
    
    Args:
        folder_path (str): Path to folder with PNG grain images
        
    Returns:
        dict: Count of each class
    """
    class_counts = {label: 0 for label in LABEL_MAP.values()}
    
    try:
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
        
        if not image_files:
            print(f"No PNG images found in {folder_path}")
            return class_counts
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            img = load_and_preprocess_image_from_path(img_path)
            
            if img is None:
                continue
            
            img_batch = np.expand_dims(img, axis=0)
            predictions = model.predict(img_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            
            class_name = LABEL_MAP.get(predicted_class, "unknown")
            if class_name in class_counts:
                class_counts[class_name] += 1
        
        return class_counts
    except Exception as e:
        print(f"Error predicting images from folder: {str(e)}")
        return class_counts


# ==================== GRAIN DETECTION ====================

def detect_and_count_rice_grains_ml(original_image):
    """
    Detects rice grains using watershed segmentation.
    Extracts grain crops for ML classification.
    
    Args:
        original_image (numpy array): Input image
        
    Returns:
        tuple: (visualization_image, percentage_list, broken_grain_count)
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
    
    # Thresholding
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations
    morphological_kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morphological_kernel, iterations=2)
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, morphological_kernel, iterations=1)
    
    # Background extraction
    background = cv2.dilate(cleaned_image, morphological_kernel, iterations=2)
    
    # Distance transform
    distance_transform = cv2.distanceTransform(cleaned_image, cv2.DIST_L2, 3)
    cv2.normalize(distance_transform, distance_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # Foreground detection
    _, foreground = cv2.threshold(distance_transform, 0.3 * distance_transform.max(), 255, 0)
    foreground = np.uint8(foreground)
    
    # Unknown region
    unknown_region = cv2.subtract(background, foreground)
    
    # Connected components
    _, markers = cv2.connectedComponents(foreground)
    markers += 1
    markers[unknown_region == 255] = 0
    
    # Watershed
    markers = cv2.watershed(original_image, markers)
    unique_markers = np.unique(markers)
    
    # Initialize counters
    broken_grain_count = 0
    broken_25_count = 0
    broken_50_count = 0
    broken_75_count = 0
    average_rice_area = 190
    
    region_size = 64
    half_size = region_size // 2
    img_h, img_w = original_image.shape[:2]
    
    # Process each grain
    for label in unique_markers:
        if label <= 1:
            continue
        
        grain_mask = np.zeros(grayscale_image.shape, dtype="uint8")
        grain_mask[markers == label] = 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        area = cv2.contourArea(contours[0])
        M = cv2.moments(contours[0])
        
        if M["m00"] == 0:
            continue
        
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
        
        # Create composite with blue background
        background_img = np.full((region_size, region_size, 3), (255, 0, 0), dtype=np.uint8)
        
        contour = contours[0].copy().astype(np.int32)
        contour[:, 0, 0] = contour[:, 0, 0] - (adj_cX - half_size)
        contour[:, 0, 1] = contour[:, 0, 1] - (adj_cY - half_size)
        
        mask = np.zeros((region_size, region_size), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        
        composed = background_img.copy()
        composed[mask == 255] = grain_region[mask == 255]
        
        # Save for batch prediction
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
    Detect stones using HSV color filtering.
    
    Args:
        image (numpy array): Input image
        
    Returns:
        int: Stone count
    """
    try:
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
            if area > 20 and len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, axes, angle) = ellipse
                major_axis, minor_axis = max(axes), min(axes)
                aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
                if 1.0 <= aspect_ratio <= 2.0:
                    stone_count += 1
        
        return stone_count
    except Exception as e:
        print(f"Error detecting stones: {str(e)}")
        return 0


def process_image(input_image):
    """
    Main processing function: detects, segments, and classifies rice grains.
    
    Args:
        input_image (numpy array): Input image
        
    Returns:
        tuple: (visualization, perfect_count, chalky_count, black_count, yellow_count, 
                brown_count, percentage_list, broken_grain_count, stone_count, husk_count)
    """
    if input_image is None:
        raise ValueError("Could not read image")
    
    # Crop image
    input_image = input_image[:, 10:-10]
    cv2.imwrite('cropped_image.jpg', input_image)
    
    # Detect and segment grains
    visualization_copy, percentage_list, broken_grain_count = detect_and_count_rice_grains_ml(input_image)
    
    # Detect stones
    stone = detect_stones(input_image)
    
    # Classify grains using ML model
    grains_folder = "grains"
    if not os.path.exists(grains_folder):
        print(f"Warning: Grains folder not found")
        class_counts = {label: 0 for label in LABEL_MAP.values()}
    else:
        class_counts = predict_images_from_folder(grains_folder)
        shutil.rmtree(grains_folder)  # Cleanup
    
    return (
        visualization_copy,
        class_counts.get("perfect", 0),
        class_counts.get("chalky", 0),
        class_counts.get("black", 0),
        class_counts.get("yellow", 0),
        class_counts.get("brown", 0),
        percentage_list,
        broken_grain_count,
        stone,
        class_counts.get("husk", 0)
    )


# ==================== TEST CODE ====================

if __name__ == "__main__":
    # Load model once
    if not load_model_once():
        sys.exit(1)
    
    # Test image path
    image_path = "/home/rvce/Desktop/compiled/static/captured/test_image.jpg"
    
    if os.path.exists(image_path):
        print(f"\nProcessing image: {image_path}")
        print("-" * 50)
        
        image = cv2.imread(image_path)
        if image is not None:
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
            print("\n✓ Processed image saved as 'processed_image.jpg'")
        else:
            print(f"Error: Could not read image from {image_path}")
    else:
        print(f"Error: Image not found at {image_path}")
