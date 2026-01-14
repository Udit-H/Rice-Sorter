import cv2
import numpy as np
import math

def detect_and_count_rice_grains(original_image):
    """
    Detects and counts rice grains in an image using watershed segmentation with cluster detection.
    
    Args:
        original_image (numpy array): Input image containing rice grains.
        
    Returns:
        tuple: Processed image, full grain count, broken grain count, cluster info, and average rice area.
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
    full_contour = []
    broken_contour = []
    cluster_contour = []
    cluster_info = []

    # Calculate average area of rice grains
    # First pass to compute average rice area from small non-cluster areas
    area_list = []
    for label in unique_markers:
        if label <= 1:
            continue
        grain_mask = np.zeros(grayscale_image.shape, dtype="uint8")
        grain_mask[markers == label] = 255
        contours, _ = cv2.findContours(grain_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            area = cv2.contourArea(contours[0])
            if 50 < area < 300:  # Filter out noise and clusters
                area_list.append(area)

    # Compute median area as robust estimate
    average_rice_area = np.median(area_list) if area_list else 120

    # Set fixed average area based on empirical measurement
    average_rice_area = 450 # Fixed average area in pixels

    # Dictionary to store contour numbers
    contour_data = {}
    contour_number = 1

    # Update cluster detection in the classification loop
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

                # Calculate eccentricity and shape features
                try:
                    # Eccentricity calculation
                    (center, (major_axis, minor_axis), angle) = cv2.fitEllipse(contours[0])
                    major = max(major_axis, minor_axis)
                    minor = min(major_axis, minor_axis)
                    if major != 0:
                        eccentricity = np.sqrt(1 - (minor/major)**2)
                    else:
                        eccentricity = 0
                        
                    # Shape features
                    x, y, w, h = cv2.boundingRect(contours[0])
                    aspect_ratio = float(w) / h if h != 0 else 0
                    hull = cv2.convexHull(contours[0])
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area != 0 else 0
                except:
                    eccentricity = 0
                    aspect_ratio = 0
                    solidity = 0

                # Determine area ratio relative to average rice
                area_ratio = area / average_rice_area

                # Detect if this is a cluster
                is_cluster = area > 1.8 * average_rice_area
                estimated_grain_count = math.ceil(area/average_rice_area) if is_cluster else 1

                # Now calculate brokenness score with all features defined
                brokenness_score = (
                    (1 - area_ratio) * 0.5 +
                    (1 - eccentricity) * 0.3 +
                    abs(1 - aspect_ratio) * 0.1 +
                    (1 - solidity) * 0.1
                )

                # More lenient full grain criteria
                is_full = (eccentricity >= 0.7 and  # Relaxed from 0.84
                          area > 0.65 * average_rice_area and  # Relaxed from 0.75
                          brokenness_score < 0.5)  # Added condition
                
                if is_full:
                    # Full grain rice
                    full_grain_count += 1
                    if is_cluster:
                        cv2.drawContours(visualization_copy, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
                        cv2.drawContours(visualization_copy, contours, -1, (255, 255, 0), thickness=3)
                        cv2.putText(visualization_copy, f"FC{contour_number}({estimated_grain_count})", 
                                  (cX-15, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        cluster_info.append({
                            'type': 'full_cluster',
                            'count': estimated_grain_count,
                            'area': area,
                            'area_ratio': area_ratio
                        })
                        cluster_contour.append(contours)
                    else:
                        cv2.drawContours(visualization_copy, contours, -1, (0, 255, 0), thickness=cv2.FILLED)
                        cv2.putText(visualization_copy, str(contour_number), (cX, cY),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    full_contour.append(contours)
                else:
                    # If not full, classify as broken with categories
                    if not is_full:
                        broken_grain_count += 1
                        # More distinct thresholds for broken categories
                        if area_ratio <= 0.35:  # Changed from 0.4
                            broken_75_count += 1
                            broken_label = "75%"
                        elif area_ratio <= 0.55:  # Changed from 0.5
                            broken_50_count += 1
                            broken_label = "50%"
                        else:
                            broken_25_count += 1
                            broken_label = "25%"

                        # Adjust classification based on combined brokenness score
                        if brokenness_score >= 0.9:  # Very broken
                            broken_label = "75%"
                            broken_75_count += 1
                            broken_50_count -= 1 if broken_label == "50%" else 0
                            broken_25_count -= 1 if broken_label == "25%" else 0
                        elif brokenness_score >= 0.8:  # Moderately broken
                            if broken_label == "25%":
                                broken_label = "50%"
                                broken_50_count += 1
                                broken_25_count -= 1

                        # Visualization code remains the same
                        if is_cluster:
                            cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
                            cv2.drawContours(visualization_copy, contours, -1, (255, 255, 0), thickness=3)
                            cv2.putText(visualization_copy, f"B{broken_label[0]}{contour_number}({estimated_grain_count})", 
                                      (cX-15, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        else:
                            cv2.drawContours(visualization_copy, contours, -1, (0, 0, 255), thickness=cv2.FILLED)
                            cv2.putText(visualization_copy, str(contour_number), (cX, cY),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        broken_contour.append(contours)

                contour_number += 1

    percentage_list = {
        '25%': broken_25_count,
        '50%': broken_50_count,
        '75%': broken_75_count
    }
    
    # Add legend to the image
    legend_y = 30
    cv2.putText(visualization_copy, "Legend: Green=Full, Red=Broken, Gray=Other, Yellow outline=Cluster", 
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(visualization_copy, "Legend: Green=Full, Red=Broken, Gray=Other, Yellow outline=Cluster", 
                (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    return (
        visualization_copy,
        full_grain_count,
        broken_grain_count,
        percentage_list,
        full_contour,
        broken_contour,
        cluster_contour,
        cluster_info
    )

def main():
    """
    Main function to run the rice grain analysis with cluster detection
    """
    try:
        # Read image
        image_path = "/home/rvce/Desktop/compiled/static/captured/captured_1748859326.jpg"  # Update with your image path
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Process image and get results
        (processed_image,
         full_count,
         broken_count,
         percentages,
         full_contours,
         broken_contours,
         cluster_contours,
         cluster_info) = detect_and_count_rice_grains(image)
        
        # Print results
        print("\n=== Rice Analysis Results ===")
        print(f"Full grains: {full_count}")
        print(f"Broken grains: {broken_count}")
        
        print("\nBreakdown of broken grains:")
        print(f"25% broken: {percentages['25%']}")
        print(f"50% broken: {percentages['50%']}")
        print(f"75% broken: {percentages['75%']}")
        
        # Print cluster information
        if cluster_info:
            print(f"\n=== Cluster Analysis ===")
            print(f"Total clusters detected: {len(cluster_info)}")
            
            full_clusters = [c for c in cluster_info if c['type'] == 'full_cluster']
            broken_clusters = [c for c in cluster_info if c['type'] == 'broken_cluster']
            unclassified_clusters = [c for c in cluster_info if c['type'] == 'unclassified_cluster']
            
            if full_clusters:
                print(f"Full grain clusters: {len(full_clusters)}")
                total_full_in_clusters = sum(c['count'] for c in full_clusters)
                print(f"  Estimated full grains in clusters: {total_full_in_clusters}")
                
            if broken_clusters:
                print(f"Broken grain clusters: {len(broken_clusters)}")
                total_broken_in_clusters = sum(c['count'] for c in broken_clusters)
                print(f"  Estimated broken grains in clusters: {total_broken_in_clusters}")
                
            if unclassified_clusters:
                print(f"Unclassified clusters: {len(unclassified_clusters)}")
                total_unclassified_in_clusters = sum(c['count'] for c in unclassified_clusters)
                print(f"  Estimated grains in unclassified clusters: {total_unclassified_in_clusters}")
            
            # Print detailed cluster breakdown
            print(f"\nDetailed cluster breakdown:")
            for i, cluster in enumerate(cluster_info):
                print(f"  Cluster {i+1}: {cluster['type']}, "
                      f"estimated {cluster['count']} grains, "
                      f"area={cluster['area']:.1f}, "
                      f"area_ratio={cluster['area_ratio']:.2f}")
        else:
            print("\nNo clusters detected.")
        
        # Save and display results
        output_path = image_path.replace('.', '_analyzed.')
        cv2.imwrite(output_path, processed_image)
        print(f"\nProcessed image saved as: {output_path}")
        
        # cv2.imshow("Processed Rice with Clusters", processed_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()