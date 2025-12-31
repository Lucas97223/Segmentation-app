import cv2
import numpy as np

def get_scale_from_ribbon(img, ribbon_contour):
    try:
        rect = cv2.minAreaRect(ribbon_contour)
        center, size, angle = rect
        center, size = tuple(map(int, center)), tuple(map(int, size))
        
        if size[0] < size[1]:
            angle += 90
            size = (size[1], size[0])
            
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        ribbon_crop = cv2.getRectSubPix(rotated, size, center)
        
        if ribbon_crop is None or ribbon_crop.size == 0:
            return None

        gray = cv2.cvtColor(ribbon_crop, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        col_sum = np.sum(thresh, axis=0)
        col_sum = col_sum / (np.max(col_sum) + 1e-6)
        
        peaks = []
        threshold_val = 0.5
        min_dist = 20 
        
        for i in range(1, len(col_sum)-1):
            if col_sum[i] > threshold_val and col_sum[i] > col_sum[i-1] and col_sum[i] > col_sum[i+1]:
                if not peaks or (i - peaks[-1]) > min_dist:
                    peaks.append(i)
        
        if len(peaks) > 1:
            diffs = np.diff(peaks)
            median_px_per_mm = np.median(diffs)
            return median_px_per_mm
        else:
            return None
            
    except Exception as e:
        print(f"Error in scale detection: {e}")
        return None

def segment_image(img, threshold_offset=0):
    if img is None:
        return [], None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return contours, thresh

def segment_image_manual(img, threshold_val):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours, thresh

def segment_image_color(img, lower_hsv, upper_hsv):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours, mask

def get_flood_fill_mask(img, seed_point, tolerance, boundaries=None, thickness=5):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Draw boundaries on the mask as hard barriers
    if boundaries:
        for poly in boundaries:
            pts = np.array(poly, np.int32)
            # Shift points by +1, +1 because mask is padded
            pts = pts + 1 
            pts = pts.reshape((-1, 1, 2))
            # Draw with value 255 (or any non-zero) to act as barrier
            cv2.polylines(mask, [pts], False, 255, thickness) # Thickness for robust barrier
    
    # Fill with value 128 to distinguish from boundaries (255)
    flags = 4 | (128 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    
    cv2.floodFill(img, mask, seed_point, (255, 255, 255), lo_diff, up_diff, flags)
    
    mask = mask[1:-1, 1:-1]
    
    # Extract only the filled area (value 128), ignoring boundaries (value 255)
    final_mask = np.zeros_like(mask)
    final_mask[mask == 128] = 255
    return final_mask

def calculate_metrics(mask, scale_px_per_unit=None):
    """
    Calculates metrics for a given binary mask.
    scale_px_per_unit: Pixels per 1 unit (e.g., px/cm). If None, returns pixels.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {}
        
    # Assume largest contour is the object
    c = max(contours, key=cv2.contourArea)
    
    area_px = cv2.contourArea(c)
    perimeter_px = cv2.arcLength(c, True)
    x, y, w, h = cv2.boundingRect(c)
    
    if perimeter_px == 0:
        circularity = 0
    else:
        circularity = 4 * np.pi * (area_px / (perimeter_px * perimeter_px))
        
    metrics = {
        "Area (px)": int(area_px),
        "Perimeter (px)": int(perimeter_px),
        "Width (px)": w,
        "Height (px)": h,
        "Circularity": round(circularity, 3),
        "Aspect Ratio": round(float(w)/h, 3) if h != 0 else 0
    }
    
    if scale_px_per_unit and scale_px_per_unit > 0:
        scale_sq = scale_px_per_unit ** 2
        metrics["Area (unit²)"] = round(area_px / scale_sq, 4)
        metrics["Perimeter (unit)"] = round(perimeter_px / scale_px_per_unit, 2)
        metrics["Width (unit)"] = round(w / scale_px_per_unit, 2)
        metrics["Height (unit)"] = round(h / scale_px_per_unit, 2)
        
    return metrics

def resolve_overlap(mask1, mask2):
    """
    Resolves overlap between two masks by assigning intersection pixels 
    to the nearest non-overlapping 'core' of each mask.
    Returns (new_mask1, new_mask2) which are non-overlapping.
    """
    # Identify intersection
    intersection = cv2.bitwise_and(mask1, mask2)
    
    # If no intersection, return as is
    if cv2.countNonZero(intersection) == 0:
        return mask1, mask2
        
    # Identify cores (non-overlapping parts)
    core1 = cv2.bitwise_and(mask1, cv2.bitwise_not(intersection))
    core2 = cv2.bitwise_and(mask2, cv2.bitwise_not(intersection))
    
    # If one mask is entirely inside the other, the core might be empty.
    # In that case, we can't do distance transform on empty core.
    # Fallback: if core is empty, keep the original (it consumes the other) or split?
    # Let's assume we want to split. If core is empty, we treat the intersection as the core?
    # Better: If core is empty, it means one object is fully inside another. 
    # The user asked to separate, so maybe we shouldn't allow fully nested objects to split 
    # unless we erode? 
    # For now, let's handle the standard case where there are distinct cores.
    
    if cv2.countNonZero(core1) == 0 or cv2.countNonZero(core2) == 0:
        # Fallback: Just subtract intersection from the new one (mask2)
        # or split evenly? Let's subtract from the NEW one (mask2) usually, 
        # but here we don't know which is new.
        # Let's just return core1 and core2 (intersection removed from both)
        return core1, core2
        
    # Compute distance transform
    # We need distance FROM the core. 
    # dist_transform calculates distance to nearest ZERO pixel.
    # So we invert the core: 0 inside core, 1 outside.
    # But we want distance *from* core boundary outwards.
    # Actually, we want distance from the core pixels.
    # So we invert: White (255) becomes Black (0).
    # We want distance to the nearest White pixel of the Core.
    # cv2.distanceTransform calculates distance to nearest zero pixel.
    # So we want Core to be Zero (Black) and Background to be White (255)?
    # No, usually we want distance from object.
    # Let's invert: inv_core = bitwise_not(core). 
    # dist = distanceTransform(inv_core). 
    # Pixels inside core will be 0. Pixels outside will increase.
    
    inv_core1 = cv2.bitwise_not(core1)
    inv_core2 = cv2.bitwise_not(core2)
    
    dist1 = cv2.distanceTransform(inv_core1, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(inv_core2, cv2.DIST_L2, 5)
    
    # Mask for intersection region
    # We only care about distances within the intersection
    
    # Create new masks starting from cores
    new_mask1 = core1.copy()
    new_mask2 = core2.copy()
    
    # Iterate over intersection pixels (or use array ops)
    # Array ops are faster
    
    # Where dist1 < dist2, assign to 1. Else 2.
    # We only update pixels in the intersection mask
    
    # Create a mask where dist1 < dist2
    # Note: dist arrays are float32
    mask_closer_to_1 = (dist1 < dist2).astype(np.uint8) * 255
    mask_closer_to_2 = (dist2 <= dist1).astype(np.uint8) * 255
    
    # Intersect with the intersection region
    part_for_1 = cv2.bitwise_and(intersection, mask_closer_to_1)
    part_for_2 = cv2.bitwise_and(intersection, mask_closer_to_2)
    
    # Add to cores
    new_mask1 = cv2.bitwise_or(new_mask1, part_for_1)
    new_mask2 = cv2.bitwise_or(new_mask2, part_for_2)
    
    return new_mask1, new_mask2

def keep_largest_component(mask):
    """
    Keeps only the largest connected component of a binary mask.
    """
    if cv2.countNonZero(mask) == 0:
        return mask
        
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    if num_labels <= 1: # 0 is background
        return mask
        
    # stats[i, cv2.CC_STAT_AREA] is the area
    # Index 0 is background, so we look at 1 onwards
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    new_mask = np.zeros_like(mask)
    new_mask[labels == largest_label] = 255
    return new_mask

def smart_cut(mask, boundaries, click_point, thickness=5):
    """
    Removes the connected component of the mask containing the click_point,
    considering boundaries as cuts.
    boundaries: List of lists of points (polylines).
    """
    # 1. Create a temporary 'cut' mask
    cut_mask = mask.copy()
    
    # 2. Draw all boundaries in Black to 'cut' the mask
    for poly in boundaries:
        pts = np.array(poly, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(cut_mask, [pts], False, 0, thickness)
        
    # 3. Find connected components on the cut mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cut_mask, connectivity=8)
    
    # 4. Identify which component contains the click_point
    x, y = click_point
    # Ensure click is within bounds
    if y < 0 or y >= labels.shape[0] or x < 0 or x >= labels.shape[1]:
        return mask # Click outside
        
    target_label = labels[y, x]
    
    if target_label == 0:
        # Clicked on background or exactly on the cut line
        # Try searching neighborhood? Or just return original (no action)
        return mask
        
    # 5. Create a mask of the component to remove
    component_to_remove = np.zeros_like(mask)
    component_to_remove[labels == target_label] = 255
    
    # 6. Subtract this component from the CUT mask
    # We use cut_mask (where boundaries are already 0) as the base.
    # This ensures that pixels under the boundary lines are REMOVED from the result.
    # Result = (Original - Boundaries) - Component
    
    new_mask = cv2.bitwise_and(cut_mask, cv2.bitwise_not(component_to_remove))
    
    return new_mask
