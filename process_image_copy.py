import cv2
import numpy as np
import os
import glob
from datetime import datetime

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
    brown_count = 0
    merged_rice = 0
    full_contour = []
    broken_contour =[]
    chalky_contour =[]
    black_contour =[]
    yellow_contour =[]
    brown_contour =[]
    merged_rice_contour = []

    # Calculate average area of rice grains - Adjust this value based on your rice size
    average_rice_area = 120  # Reduced from 160 to detect smaller grains
    
    # Dictionary to store contour numbers and their RGB values
    contour_data = {}
    contour_number = 1

    # (Removed CSV directory and file creation for contour details)

    # Prepare for saving individual grain images
    # Try to get the input image path via cv2 global variable (if called from main), else use fallback
    import inspect
    frame = inspect.currentframe()
    input_image_path = None
    for i in range(2, 8):
        try:
            caller = inspect.getouterframes(frame)[i]
            local_vars = caller.frame.f_locals
            if 'image_path' in local_vars:
                input_image_path = local_vars['image_path']
                break
        except Exception:
            break
    if input_image_path is None:
        input_image_path = "unknown_image"
    # Remove extension for folder name
    base_folder_name = os.path.splitext(os.path.basename(input_image_path))[0]
    grains_dir = os.path.join('grains', base_folder_name)
    if not os.path.exists(grains_dir):
        os.makedirs(grains_dir)

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
            # cv2.drawContours(contour_mask, contours, -1, 1, thickness=cv2.FILLED)  # Fill the contour

            # Extract the pixel values from the original image using the mask
            masked_pixels = original_image[contour_mask == 1]
            sorted_bgr = masked_pixels[np.lexsort((masked_pixels[:, 2], masked_pixels[:, 1], masked_pixels[:, 0]))]
            masked_pixels = sorted_bgr[5:-5]

            # Calculate mean RGB values
            mean_rgb = np.mean(masked_pixels, axis=0)

            # Rest of the existing classification code...
            count_for_chalky = np.sum(np.all(masked_pixels >= [220, 200, 190], axis=1) & (np.abs(masked_pixels[:, 0] - masked_pixels[:, 2]) < 40))
            count_for_black = np.sum(np.all(masked_pixels <= [100, 100, 100], axis=1))
            count_for_yellow = np.sum(
                (np.all(masked_pixels >= [155, 145, 145], axis=1) &
                np.all(masked_pixels <= [200, 180, 180], axis=1)) &
                (np.abs(masked_pixels[:, 0] - masked_pixels[:, 2]) < 30))

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
                # grain_multiplier = area // average_rice_area - 1
                total_grain_count += grain_multiplier

            # Classify as full or broken grain
            # Determine area ratio relative to average rice
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
                # cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
                broken_contour.append(contours)

            # 2. Merged rice
            elif area > 2.75 * average_rice_area:
                merged_rice += 1 + grain_multiplier
                # cv2.drawContours(visualization_copy, contours, -1, (100, 100, 100), thickness=cv2.FILLED)
                merged_rice_contour.append(contours)

            # 3. Black rice
            elif count_for_black >= 8:
                black_count += 1 + grain_multiplier
                # cv2.drawContours(visualization_copy, contours, -1, (10, 10, 10), thickness=cv2.FILLED)
                black_contour.append(contours)

            # 4. Yellow rice
            elif count_for_yellow >= 8:
                yellow_count += 1 + grain_multiplier
                # cv2.drawContours(visualization_copy, contours, -1, (0, 255, 255), thickness=cv2.FILLED)
                yellow_contour.append(contours)

            # 5. Chalky rice
            elif count_for_chalky >= 4:
                chalky_count += 1 + grain_multiplier
                # cv2.drawContours(visualization_copy, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
                chalky_contour.append(contours)

            # 6. Full grain rice
            elif eccentricity >= 0.84 and area > 0.75 * average_rice_area:
                full_grain_count += 1 + grain_multiplier
                # cv2.drawContours(visualization_copy, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
                full_contour.append(contours)

            # Optional: Handle unknown/ambiguous cases (if needed)
            #else:
            #    unknown_count += 1
            #    cv2.drawContours(visualization_copy, contours, -1, (255, 0, 0), thickness=cv2.FILLED)
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
        merged_rice,
        percentage_list,
        full_contour,
        broken_contour,
        yellow_contour,
        black_contour,
        chalky_contour,
        merged_rice_contour
    )

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



#     return result_image, stone_count, total_stone_area

# def detect_husk(image):
#     """
#     Detects husk in an image using HSV color space filtering and applies a dark blue mask.

#     Args:
#         image (numpy array): Input image containing potential husk.

#     Returns:
#         tuple: Image with husk highlighted in dark blue, binary mask of husks,
#                number of detected husks, and total husk area.
#     """
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_husk_color = np.array([0, 10, 30])
#     upper_husk_color = np.array([40, 255, 255])
#     husk_mask = cv2.inRange(hsv_image, lower_husk_color, upper_husk_color)

#     # Morphological operation to clean up the mask
#     morphological_kernel = np.ones((3, 3), np.uint8)
#     cleaned_husk_mask = cv2.morphologyEx(husk_mask, cv2.MORPH_CLOSE, morphological_kernel, iterations=2)

#     # Find contours
#     husk_contours, _ = cv2.findContours(cleaned_husk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     husk_count = 0
#     total_husk_area = 0
#     result_image = image.copy()
#     mask_output = np.zeros(image.shape[:2], dtype=np.uint8)

#     husk_contours_return =[]

#     for contour in husk_contours:
#         area = cv2.contourArea(contour)
#         if area > 120:
#             x, y, width, height = cv2.boundingRect(contour)
#             aspect_ratio = float(width) / height
#             if 0.2 < aspect_ratio < 5.0:
#                 husk_contours_return.append(contour)
#                 husk_count += 1
#                 total_husk_area += area
#                 cv2.drawContours(mask_output, [contour], -1, 255, thickness=cv2.FILLED)
#                 # Highlight husks in result image
#                 dilated_mask = cv2.dilate(mask_output, np.ones((7, 7), np.uint8), iterations=2)
#                 result_image[dilated_mask == 255] = (220,7,13)

#     return result_image, husk_count, husk_contours_return

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
#     hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([100, 150, 50])
#     upper_blue = np.array([140, 255, 255])
#     blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         mask = np.zeros_like(blue_mask)
#         cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         cropped = input_image[y:y+h, x:x+w]
#         mask_cropped = mask[y:y+h, x:x+w]
#         input_image = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)
#         input_image = input_image[50:-60, 35:-40]

#     cv2.imshow("cropped : ",input_image)
#     # stone_image, stone_count, stone_area = detect_stones(input_image)
#     # husk_image, husk_count, husk_contours_return = detect_husk(input_image)

#     # Detect rice grains
#     # visual, full_grain_count, broken_grain_count, chalky_count, black_count, yellow_count, brown_count, merge_rice ,percent_list, full_contour, yellow_contour, black_contour, chalky_contour,merged_rice_contour = detect_and_count_rice_grains(input_image)

#     # Draw husk contours on the visual image in pink
#     # cv2.drawContours(visual, husk_contours_return, -1, (255, 192, 203), thickness=cv2.FILLED)  # Pink color

#     return (
#         visual
#     #     int(full_grain_count),
#     #     int(chalky_count),
#     #     int(black_count),
#     #     int(yellow_count),
#     #     int(brown_count),
#     #     percent_list,
#     #     int(broken_grain_count),
#     #     # int(stone_count),
#     #     # int(husk_count),
#     )

# Test the function

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



#     return result_image, stone_count, total_stone_area

# def detect_husk(image):
#     """
#     Detects husk in an image using HSV color space filtering and applies a dark blue mask.

#     Args:
#         image (numpy array): Input image containing potential husk.

#     Returns:
#         tuple: Image with husk highlighted in dark blue, binary mask of husks,
#                number of detected husks, and total husk area.
#     """
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     lower_husk_color = np.array([0, 10, 30])
#     upper_husk_color = np.array([40, 255, 255])
#     husk_mask = cv2.inRange(hsv_image, lower_husk_color, upper_husk_color)

#     # Morphological operation to clean up the mask
#     morphological_kernel = np.ones((3, 3), np.uint8)
#     cleaned_husk_mask = cv2.morphologyEx(husk_mask, cv2.MORPH_CLOSE, morphological_kernel, iterations=2)

#     # Find contours
#     husk_contours, _ = cv2.findContours(cleaned_husk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     husk_count = 0
#     total_husk_area = 0
#     result_image = image.copy()
#     mask_output = np.zeros(image.shape[:2], dtype=np.uint8)

#     husk_contours_return =[]

#     for contour in husk_contours:
#         area = cv2.contourArea(contour)
#         if area > 120:
#             x, y, width, height = cv2.boundingRect(contour)
#             aspect_ratio = float(width) / height
#             if 0.2 < aspect_ratio < 5.0:
#                 husk_contours_return.append(contour)
#                 husk_count += 1
#                 total_husk_area += area
#                 cv2.drawContours(mask_output, [contour], -1, 255, thickness=cv2.FILLED)
#                 # Highlight husks in result image
#                 dilated_mask = cv2.dilate(mask_output, np.ones((7, 7), np.uint8), iterations=2)
#                 result_image[dilated_mask == 255] = (220,7,13)

#     return result_image, husk_count, husk_contours_return

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
#     hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
#     lower_blue = np.array([100, 150, 50])
#     upper_blue = np.array([140, 255, 255])
#     blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
#     contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         mask = np.zeros_like(blue_mask)
#         cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
#         x, y, w, h = cv2.boundingRect(largest_contour)
#         cropped = input_image[y:y+h, x:x+w]
#         mask_cropped = mask[y:y+h, x:x+w]
#         input_image = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)
#         input_image = input_image[50:-60, 35:-40]

#     cv2.imshow("cropped : ",input_image)
#     # stone_image, stone_count, stone_area = detect_stones(input_image)
#     # husk_image, husk_count, husk_contours_return = detect_husk(input_image)

#     # Detect rice grains
#     # visual, full_grain_count, broken_grain_count, chalky_count, black_count, yellow_count, brown_count, merge_rice ,percent_list, full_contour, yellow_contour, black_contour, chalky_contour,merged_rice_contour = detect_and_count_rice_grains(input_image)

#     # Draw husk contours on the visual image in pink
#     # cv2.drawContours(visual, husk_contours_return, -1, (255, 192, 203), thickness=cv2.FILLED)  # Pink color

#     return (
#         visual
#     #     int(full_grain_count),
#     #     int(chalky_count),
#     #     int(black_count),
#     #     int(yellow_count),
#     #     int(brown_count),
#     #     percent_list,
#     #     int(broken_grain_count),
#     #     # int(stone_count),
#     #     # int(husk_count),
#     )

# Test the function
if __name__ == "__main__":
    image_paths = ["everything1.jpg", "everything2.jpg", "allchalky1.jpg", "allchalky2.jpg"]  # Replace with your image path
    st_time = datetime.now()
    for image_path in image_paths:
        image = cv2.imread(image_path)
        results = detect_and_count_rice_grains(image)
        print("Detection results:")
        cv2.imshow("Detected Rice Grains", results[0])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Total grains detected: {results[1] + results[2] + results[3] + results[4] + results[5] + results[6]}")
        print(f"Full grains: {results[1]}")
        print(f"Broken grains: {results[2]}")
        print(f"Chalky grains: {results[3]}")
        print(f"Black grains: {results[4]}")
        print(f"Yellow grains: {results[5]}")
        print(f"Merged rice: {results[6]}")
        print(f"Broken grain percentages: {results[7]}")
    print(f"Time Taken: {datetime.now()-st_time}")