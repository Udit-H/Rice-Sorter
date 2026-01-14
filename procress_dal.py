import cv2
import numpy as np
import math

def process_dal(img):
    MIN_DAL_AREA = 200 # 250
    GOOD_DAL_MIN_AREA = 720
    GOOD_DAL_MAX_AREA = 1500

    BROKEN_SOLIDITY_THRESHOLD = 0.92
    BROKEN_CIRCULARITY_THRESHOLD = 0.77
    BROKEN_ASPECT_RATIO_THRESHOLD = 1.77

    LOWER_BLUE = np.array([100, 100, 100])
    UPPER_BLUE = np.array([140, 255, 255])
    if img is None:
        print(f"Error loading image at {img}")
        exit()


    # Convert to HSV and apply blue mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    dal_mask = cv2.bitwise_not(blue_mask)


    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dal_mask = cv2.morphologyEx(dal_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_img = img.copy()

    # Counters
    good_dal_count = 0
    broken_dal_count = 0
    gray_dal_count = 0  # For large gray contours ignored
    black_spots_count = 0
    broken_50=0
    broken_25=0
    broken_75=0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_DAL_AREA:
            continue

        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        box_w, box_h = rect[1]
        aspect_ratio = box_w / box_h if box_h != 0 else 0
        if aspect_ratio < 1:
            aspect_ratio = 1 / aspect_ratio

        circularity = (4 * math.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0

        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
        masked_pixels = img[mask > 0]
        
        count_for_black = np.sum((np.all(masked_pixels <= [50, 50, 60], axis=1) ))

        # Classification based on updated thresholds
        if area > GOOD_DAL_MAX_AREA:
            # Large contour, draw gray and ignore counts
            color = (128, 128, 128)  # Gray
            gray_dal_count += 1
            label = "Large (Ignored)"
        elif count_for_black>10:
            black_spots_count+=1
            color=(30,30,30)
            label ="black"
        elif (area < GOOD_DAL_MIN_AREA or
            solidity < BROKEN_SOLIDITY_THRESHOLD or
            circularity < BROKEN_CIRCULARITY_THRESHOLD or
            aspect_ratio > BROKEN_ASPECT_RATIO_THRESHOLD):
            if 300 < area < 450:
                broken_50 +=1
            elif 450 >= area:
                broken_75 +=1
            else :
                broken_25 +=1
            color = (0, 0, 255)  # Red for Broken
            broken_dal_count += 1
            label = "Broken"
        
        else:
            color = (0, 255, 0)  # Green for Good
            good_dal_count += 1
            label = "Good"

        # --- Shrink contour inward by 3 pixels ---
        contour_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=-1)
        eroded_mask = cv2.erode(contour_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        inner_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for inner_contour in inner_contours:
            cv2.drawContours(output_img, [inner_contour], -1, color, cv2.FILLED)

    # Print summary
    total_dal_count = good_dal_count + broken_dal_count + gray_dal_count
    # Show final result
    display_img = cv2.resize(output_img, (800, int(output_img.shape[0] * (800 / output_img.shape[1]))))
    broken_percent = {
        "25%": broken_25,
        "50%": broken_50,
        "75%": broken_75
    }
    return good_dal_count, broken_dal_count, broken_percent, display_img , black_spots_count

if __name__ == "__main__":
    image = cv2.imread("/home/rvce/Desktop/compiled/static/captured/captured_1748522574.jpg")
    if image is None:
        print("Error: Could not load image")
    else:
        full_count, broken_count, broken_percent, visualization, black_spots = process_dal(image)
        cv2.imwrite("gay_2.png",visualization)
        print(f"Full grains: {full_count}, Broken grains: {broken_count}, Black spots: {black_spots}")
        print(f"Broken percent breakdown: {broken_percent}")
        