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
    """
    Segments the image to find contours.
    threshold_offset: Value to add/subtract from the automatic threshold (not used with Otsu directly, 
                      but we can switch to adaptive or manual if needed).
    For now, we stick to Otsu but we could allow manual thresholding.
    """
    if img is None:
        return [], None
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Standard Otsu
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # If we wanted manual offset, we'd need to do manual thresholding using ret + offset
    # But Otsu finds the optimal 'ret'. 
    # Let's support manual override if offset is provided (and non-zero implies manual mode?)
    # Or just simple manual threshold if provided.
    
    if threshold_offset != 0:
        # Interpret offset as absolute threshold value if > 0? Or relative?
        # Let's say if offset is provided, we use it as the threshold value (0-255)
        # If it's -1 (default), use Otsu.
        pass 
        
    # Actually, for the app, a slider for threshold is good.
    # So let's change the signature to accept an explicit threshold value, or None for Otsu.
    
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
    # Create mask for color range
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Clean up mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours, mask

def get_flood_fill_mask(img, seed_point, tolerance):
    """
    Returns a binary mask of the region connected to seed_point with similar color.
    tolerance: Max difference in pixel value (0-255).
    """
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Flood fill flags
    # 4 connectivity, fixed range, mask only (do not change image)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY
    
    # Define lower and upper diff
    lo_diff = (tolerance, tolerance, tolerance)
    up_diff = (tolerance, tolerance, tolerance)
    
    cv2.floodFill(img, mask, seed_point, (255, 255, 255), lo_diff, up_diff, flags)
    
    # Crop mask to image size (remove border)
    mask = mask[1:-1, 1:-1]
    return mask
