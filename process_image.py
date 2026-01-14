# import cv2
# import numpy as np
# import os
# import tensorflow as tf
# import numpy as np
# import os
# import sys
# import shutil
# from tensorflow.keras.models import load_model

# # Constants
# IMAGE_SIZE = (64, 64)
# NUM_CLASSES = 6
# LABEL_MAP = {0: "black", 1: "brown", 2: "yellow", 3: "chalky", 4: "perfect", 5: "husk"}

# # Load model
# MODEL_PATH = '/home/rvce/Desktop/compiled/o4mnew.keras' 
# model = load_model(MODEL_PATH)

# def detect_stones(image):
#     """
#     Detects stones in an image using HSV color space filtering and applies a dark blue mask.

#     Args:
#         image (numpy array): Input image containing potential stones.

#     Returns:
#         tuple: Image with stones highlighted in dark blue, stone count, and total stone area.
#     """
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_stone_color = np.array([5, 50, 50])
#     upper_stone_color = np.array([25, 255, 200])
#     stone_mask = cv2.inRange(hsv_image, lower_stone_color, upper_stone_color)
#     morphological_kernel = np.ones((3, 3), np.uint8)
#     cleaned_stone_mask = cv2.morphologyEx(stone_mask, cv2.MORPH_CLOSE, morphological_kernel, iterations=2)
#     stone_contours, _ = cv2.findContours(cleaned_stone_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     stone_count = 0
#     total_stone_area = 0
#     result_image = image.copy()

#     for contour in stone_contours:
#         area = cv2.contourArea(contour)
#         if area > 20:
#             if len(contour) >= 5:
#                 ellipse = cv2.fitEllipse(contour)
#                 (center, axes, angle) = ellipse
#                 major_axis, minor_axis = max(axes), min(axes)
#                 aspect_ratio = major_axis / minor_axis if minor_axis != 0 else 0
#                 if 1.0 <= aspect_ratio <= 2.0:
#                     stone_count += 1
#                     total_stone_area += area
#                     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#                     cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
#                     # Use a larger kernel and more iterations for stronger dilation
#                     dilated_mask = cv2.dilate(mask, np.ones((7, 7), np.uint8), iterations=2)
#                     result_image[dilated_mask == 255] = (255,255,0)



#     return stone_count


# def load_and_preprocess_image(path):
#     img = tf.io.read_file(path)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.resize(img, IMAGE_SIZE)
#     img = tf.cast(img, tf.float32) / 255.0

#     # brightness, contrast, saturation adjustments
#     img = img * 0.72                              # reduce brightness by 28%
#     img = tf.image.adjust_contrast(img, 1.04)     # increase contrast by 4%
#     img = tf.image.adjust_saturation(img, 0.54)   # decrease saturation by 46%
#     img = tf.clip_by_value(img, 0.0, 1.0)
#     return img

# # Predict for all images in the 'grains' folder and count predictions
# def predict_images_from_folder(folder_path):
#     # Initialize counters for each class
#     class_counts = {label: 0 for label in LABEL_MAP.values()}

#     # Get all PNG images in the folder, sorted by name
#     image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])

#     for img_file in image_files:
#         img_path = os.path.join(folder_path, img_file)
#         img = load_and_preprocess_image(img_path)
#         img_expanded = tf.expand_dims(img, 0)  # Add batch dimension
#         predictions = model.predict(img_expanded, verbose=0)
#         predicted_class = np.argmax(predictions)
#         confidence = predictions[0][predicted_class] * 100

#         class_name = LABEL_MAP[predicted_class]
#         class_counts[class_name] += 1
#     return class_counts

# def detect_and_count_rice_grains(original_image):
#     """
#     Detects rice grains in an image, saves 64x64 crops of grains with area < 300 to the 'grains' folder,
#     and draws gray contours for grains with area >= 300. Returns the visualization image.
    
#     Args:
#         original_image (numpy array): Input image containing rice grains.
        
#     Returns:
#         visualization image (numpy array)
#     """
#     if original_image is None:
#         raise ValueError("Could not read image")
    
#     visualization_copy = original_image.copy()
#     grains_dir = "grains"
#     os.makedirs(grains_dir, exist_ok=True)
    
#     img_counter = 1

#     # Convert to HSV and grayscale
#     hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
#     grayscale_image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

#     # Thresholding to create a binary image
#     _, binary_image = cv2.threshold(
#         grayscale_image, 0, 255, 
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
#     # Morphological operations to clean the image
#     morphological_kernel = np.ones((3, 3), np.uint8)
#     cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morphological_kernel, iterations=2)
#     cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, morphological_kernel, iterations=1)

#     # Background extraction
#     background = cv2.dilate(cleaned_image, morphological_kernel, iterations=2)
    
#     # Distance transform for watershed preparation
#     distance_transform = cv2.distanceTransform(cleaned_image, cv2.DIST_L2, 3)
#     cv2.normalize(distance_transform, distance_transform, 0, 1.0, cv2.NORM_MINMAX)

#     # Foreground detection
#     _, foreground = cv2.threshold(distance_transform, 0.2 * distance_transform.max(), 255, 0)
#     foreground = np.uint8(foreground)

#     # Unknown region identification
#     unknown_region = cv2.subtract(background, foreground)

#     # Connected components labeling
#     _, markers = cv2.connectedComponents(foreground)
#     markers += 1
#     markers[unknown_region == 255] = 0

#     # Watershed segmentation
#     markers = cv2.watershed(original_image, markers)
#     unique_markers = np.unique(markers)
#     average_area = 200

#     broken_grain_count=0
#     broken_25_count = 0
#     broken_50_count = 0
#     broken_75_count = 0
#     percentage_list={}
#     merge_count = 0

#     region_size = 64
#     half_size = region_size // 2
#     img_h, img_w = original_image.shape[:2]

#     for label in unique_markers:
#         if label <= 1:
#             continue
#         grain_mask = np.zeros(grayscale_image.shape, dtype="uint8")
#         grain_mask[markers == label] = 255
#         contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         if contours:
#             area = cv2.contourArea(contours[0])
#             M = cv2.moments(contours[0])
#             if M["m00"] != 0:
#                 cX = int(M["m10"] / M["m00"])
#                 cY = int(M["m01"] / M["m00"])

#                 contour_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
#                 cv2.drawContours(contour_mask, contours, -1, 1, thickness=cv2.FILLED)  # Fill the contour
                
#                 # Extract the pixel values from the original image using the mask
#                 masked_pixels = original_image[contour_mask == 1]
#                 sorted_bgr = masked_pixels[np.lexsort((masked_pixels[:, 2], masked_pixels[:, 1], masked_pixels[:, 0]))]
#                 masked_pixels = sorted_bgr[5:-5]
#                 count_for_brown = np.sum(
#                     (np.all(masked_pixels >= [107, 64, 81], axis=1) &
#                     np.all(masked_pixels <= [182, 141, 147], axis=1)))


#                 if 20 < area < 160:
#                     broken_grain_count += 1
#                     if 60 < area < 100:
#                         broken_50_count+=1
#                     elif area<=60:
#                         broken_25_count+=1
#                     else:
#                         broken_75_count+=1
#                     cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=cv2.FILLED)  # Red
#                     continue
#                 elif area > 280 and count_for_brown <= 5:
#                     merge_count += 1
#                     cv2.drawContours(visualization_copy, contours, -1, (128, 128, 0), thickness=cv2.FILLED)  # Grey
#                     continue
#                 else:
#                     # Extract 64x64 region centered at (cX, cY)
#                     adj_cX = min(max(cX, half_size), img_w - half_size)
#                     adj_cY = min(max(cY, half_size), img_h - half_size)
#                     grain_region = cv2.getRectSubPix(original_image, (region_size, region_size), (float(adj_cX), float(adj_cY)))
#                     # Blue background
#                     background = np.full((region_size, region_size, 3), (255, 0, 0), dtype=np.uint8)
#                     # Shift contour to local coordinates
#                     contour = contours[0].copy().astype(np.int32)
#                     contour[:, 0, 0] = contour[:, 0, 0] - (adj_cX - half_size)
#                     contour[:, 0, 1] = contour[:, 0, 1] - (adj_cY - half_size)
#                     mask = np.zeros((region_size, region_size), dtype=np.uint8)
#                     cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
#                     composed = background.copy()
#                     composed[mask == 255] = grain_region[mask == 255]
#                     # Save the image with incremental name in the image-specific folder
#                     cv2.imwrite(os.path.join(grains_dir, f"{img_counter}.png"), composed)
#                     img_counter += 1
#     percentage_list = {
#         "25%": broken_25_count,
#         "50%": broken_50_count,
#         "75%": broken_75_count
#     } 

#     return visualization_copy , merge_count,percentage_list,broken_grain_count

# def process_image(input_image):
#     """
#     Processes an image to identify and visualize different components (rice, stones, husk).
    
#     Args:
#         input_image (numpy array): Input image containing rice and potential impurities.
        
#     Returns:
#         tuple: Processed image with masks applied, and counts of various components.
#     """
#     if input_image is None:
#         raise ValueError("Could not read image")

#     # Crop to the inner blue region with a tight bounding box and mask
#     # hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
#     # lower_blue = np.array([100, 150, 50])
#     # upper_blue = np.array([140, 255, 255])
#     # blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     # contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # if contours:
#     #     largest_contour = max(contours, key=cv2.contourArea)
#     #     mask = np.zeros_like(blue_mask)
#     #     cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#     #     x, y, w, h = cv2.boundingRect(largest_contour)
#     #     cropped = input_image[y:y+h, x:x+w]
#     #     mask_cropped = mask[y:y+h, x:x+w]
#     #     input_image = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)
#     #     input_image = input_image[50:-60, 35:-40]
#     # Detect rice grains
#     input_image = input_image[:, 10:-10]
#     cv2.imwrite('cropped_image.jpg', input_image)  # Save cropped image for debugging
#     visualization_copy , merge_count,percentage_list,broken_grain_count = detect_and_count_rice_grains(input_image)
#     # stone = detect_stones(input_image)
#     stone = 0
#     grains_folder = "grains"
#     if not os.path.exists(grains_folder):
#         print(f"Folder not found: {grains_folder}")
#         sys.exit(1)
#     class_counts = predict_images_from_folder(grains_folder)
#     shutil.rmtree(grains_folder)  # Clean up the folder after processing
#     return visualization_copy, class_counts["perfect"],class_counts["chalky"],class_counts["black"],class_counts["yellow"],class_counts["brown"],percentage_list,broken_grain_count,stone,class_counts["husk"]

# # Test the function
# if __name__ == "__main__":
#     # Specify the input image path
#     image_path = "/home/rvce/Desktop/compiled/static/captured/captured_1748464654.jpg"  # Change this to your image path
    
#     # Read image
#     image = cv2.imread(image_path)
    
#     if image is None:
#         print(f"Failed to load image: {image_path}")
#     else:
#         # Get image name without extension
        
#         print(f"\nProcessing ")
#         print("-" * 50)
        
#         # Process the image
#         visual, perfect_count, chalky_count, black_count, yellow_count, brown_count, percentage_list, broken_grain_count, stone_count, husk_count = process_image(image)
        
#         # Print results
#         print("\nRice Analysis Results:")
#         print("-" * 30)
#         print(f"Perfect Rice: {perfect_count}")
#         print(f"Chalky Rice: {chalky_count}")
#         print(f"Black Rice: {black_count}")
#         print(f"Yellow Rice: {yellow_count}")
#         print(f"Brown Rice: {brown_count}")
#         print(f"Husk: {husk_count}")
#         print(f"Broken Rice: {broken_grain_count}")
#         print(f"Stones: {stone_count}")
        
#         print("\nBroken Rice Distribution:")
#         print("-" * 30)
#         for percentage, count in percentage_list.items():
#             print(f"{percentage} broken: {count}")
        
#         # Display the result
#         cv2.imwrite('processed_image.jpg', visual)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
import cv2
import numpy as np


# def adjust_brightness_contrast_saturation(bgr_img):
#     img = bgr_img.astype(np.float32)
#     img = img * 0.72  # Brightness
#     img = np.clip(img, 0, 255)
#     mean = np.mean(img, axis=(0,1), keepdims=True)
#     img = (img - mean) * 1.04 + mean  # Contrast
#     img = np.clip(img, 0, 255)
#     hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
#     hsv[...,1] = hsv[...,1] * 0.54  # Saturation
#     hsv[...,1] = np.clip(hsv[...,1], 0, 255)
#     img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
#     cv2.imshow("image : ", img)
#     cv2.imwrite("adjusted_image.jpg", img)  # Save the adjusted image
#     return img



def detect_and_count_rice_grains(original_image):
    """
    Detects and counts rice grains in an image using watershed segmentation.
    
    Args:
        original_image (numpy array): Input image containing rice grains.
        
    Returns:
        tuple: Processed image, full grain count, broken grain count, and average rice area.
    """
    if original_image is None:
        raise ValueError("Could not read image")
    
    # Create a copy of the original image for visualization
    visualization_copy = original_image.copy()
    
    # original_image = adjust_brightness_contrast_saturation(original_image)
    # Convert to HSV for processing 
    hsv = cv2.cvtColor(original_image,cv2.COLOR_BGR2HSV)

    # Convert to grayscale for processing
    grayscale_image = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    # Thresholding to create a binary image
    _, binary_image = cv2.threshold(
        grayscale_image, 0, 255, 
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphological operations to clean the image
    morphological_kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, morphological_kernel, iterations=2)
    # Add closing operation to fill small holes
    cleaned_image = cv2.morphologyEx(cleaned_image, cv2.MORPH_CLOSE, morphological_kernel, iterations=1)

    # Background extraction
    background = cv2.dilate(cleaned_image, morphological_kernel, iterations=2)
    
    # Distance transform for watershed preparation
    distance_transform = cv2.distanceTransform(cleaned_image, cv2.DIST_L2, 3)

    cv2.normalize(distance_transform, distance_transform, 0, 1.0, cv2.NORM_MINMAX)

    
    # Foreground detection - Lower threshold to detect more grains
    _, foreground = cv2.threshold(distance_transform, 0.3 * distance_transform.max(), 255, 0)
    foreground = np.uint8(foreground)

    # Unknown region identification
    unknown_region = cv2.subtract(background, foreground)

    # Connected components labeling
    _, markers = cv2.connectedComponents(foreground)
    markers += 1  # Ensure background is not 0
    markers[unknown_region == 255] = 0
    
    # Watershed segmentation
    markers = cv2.watershed(original_image, markers)
    
    # Count unique regions (excluding background and boundaries)
    unique_markers = np.unique(markers)
    total_grain_count = len(unique_markers) - 2  # Subtract background and boundary
    
    # Initialize counters and storage for full and broken grains
    full_grain_count = 0
    broken_25_count = 0
    broken_50_count = 0
    broken_75_count = 0
    broken_grain_count = 0
    percentage_list = {}
    chalky_count =0
    black_count = 0
    yellow_count = 0
    # Calculate average area of rice grains - Adjust this value based on your rice size
    average_rice_area = 190  # Reduced from 160 to detect smaller grains
    
    # Dictionary to store contour numbers and their RGB values
    contour_data = {}
    contour_number = 1
    # Classify grains as full or broken based on shape and size
    for label in unique_markers:
        if label <= 1:  # Skip background and boundary
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

                # Extract the pixel values in a circle with a radius of 3 pixels
                circle_radius = 3
                circle_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
                cv2.circle(circle_mask, (cX, cY), circle_radius, 1, -1)  # Create a filled circle mask

                # Extract the pixel values from the original image using the mask
                masked_pixels = original_image[circle_mask == 1]
            # Extract the pixel values from the original image using the mask
            contour_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
            cv2.drawContours(contour_mask, contours, -1, 1, thickness=2)  # Fill the contour

            # Extract the pixel values from the original image using the mask
            masked_pixels = original_image[contour_mask == 1]
            sorted_bgr = masked_pixels[np.lexsort((masked_pixels[:, 2], masked_pixels[:, 1], masked_pixels[:, 0]))]
            masked_pixels = sorted_bgr[5:-5]

            # Calculate mean RGB values
            mean_rgb = np.mean(masked_pixels, axis=0)

            # Rest of the existing classification code...
            count_for_chalky = np.sum(np.all(masked_pixels >= [220, 200, 190], axis=1))
            count_for_yellow = np.sum(
                (np.all(masked_pixels >= [155, 145, 145], axis=1) &
                np.all(masked_pixels <= [200, 180, 180], axis=1)))

            try:
                # Calculate eccentricity for shape analysis
                (center, (major_axis, minor_axis), angle) = cv2.fitEllipse(contours[0])
                major = max(major_axis, minor_axis)
                minor = min(major_axis, minor_axis)
                eccentricity = np.sqrt(1 - (minor)**2 / (major)**2)
            except:
                eccentricity = 0

            # Draw contour number on the image
            cv2.putText(visualization_copy, str(contour_number), (cX, cY),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            contour_number += 1

            # Handle overlapping or clustered grains
            grain_multiplier = 0
            if area > 2 * average_rice_area:
                grain_multiplier = area // average_rice_area - 1
                total_grain_count += grain_multiplier

            area_ratio = area / average_rice_area

            # 1. Broken rice
            if area_ratio <= 0.75:
                broken_grain_count += 1 + grain_multiplier
                if area_ratio > 0.45:
                    broken_25_count += 1
                elif area_ratio > 0.3:
                    broken_50_count += 1
                else:
                    broken_75_count += 1
                cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=2)

            # 4. Yellow rice
            if count_for_yellow >= 8:
                yellow_count += 1 + grain_multiplier
                cv2.drawContours(visualization_copy, contours, -1, (0, 255, 255), thickness=2)

            # 5. Chalky rice
            elif count_for_chalky >= 6:
                chalky_count += 1 + grain_multiplier
                cv2.drawContours(visualization_copy, contours, -1, (255, 255, 255), thickness=2)

            # 6. Full grain rice
            elif eccentricity >= 0.84 and area > 0.75 * average_rice_area:
                if(chalky_count >= 1):
                    chalky_count += 1 + grain_multiplier
                    cv2.drawContours(visualization_copy, contours, -1, (255, 255, 255), thickness=2)
                    continue
                full_grain_count += 1 + grain_multiplier
                cv2.drawContours(visualization_copy, contours, -1, (0, 255, 0), thickness=2)

            percentage_list = {
                '25%': broken_25_count,
                '50%': broken_50_count,
                '75%': broken_75_count
            }
    
    return (
        visualization_copy,
        full_grain_count,
        broken_grain_count,
        chalky_count,
        black_count,
        yellow_count,
        percentage_list,
    )

def main():

    # Read the image
    image = "res.jpg"
    image = cv2.imread(image)
    # Process the image
    results = detect_and_count_rice_grains(image)
    
    # Print results
    print("\nDetection Results:")
    print(f"Full grains: {results[1]}")
    print(f"Broken grains: {results[2]}")
    print(f"Chalky grains: {results[3]}")
    print(f"Black grains: {results[4]}")
    print(f"Yellow grains: {results[5]}")
    print("\nBroken grain percentages:")
    for percentage, count in results[6].items():
        print(f"{percentage} broken: {count}")
    
    # Display the processed image
    cv2.imshow("Detected Rice Grains", results[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
