import cv2
import numpy as np
import time
from typing import Dict, Any

def _get_base_response() -> Dict[str, Any]:
    return {
        "center": None,
        "radius": None,
        "success": False,
        "debug": {},
        "diagnostics": {"message": "", "elapsed_ms": 0.0}
    }

def detect_hough(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    """
    Detects a circle using the standard Hough Circle Transform.
    
    Args:
        gray_image: 2D uint8 numpy array representing the grayscale input image.
        params: Dictionary containing Hough parameters:
            - dp (float)
            - minDist (int)
            - param1 (int)
            - param2 (int)
            - minRadius (int)
            - maxRadius (int)
            
    Returns:
        Dict adhering to the unified API response format.
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale numpy array")
            
        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param1 = params.get('param1', 50)
        param2 = params.get('param2', 30)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        # Apply a slight blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        response['debug']['edge_map'] = cv2.Canny(blurred, param1 // 2, param1)
        
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist,
            param1=param1, 
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))
            # If multiple circles, pick the one with the strongest vote (which is the first one returned by HoughCircles)
            best_circle = circles[0, 0]
            response['center'] = [int(best_circle[0]), int(best_circle[1])]
            response['radius'] = int(best_circle[2])
            response['success'] = True
            response['diagnostics']['message'] = f"Found {circles.shape[1]} circle(s). Returning the most confident."
            response['debug']['votes'] = circles[0] # Store all found circles for debug
        else:
            response['diagnostics']['message'] = "No circles found matching criteria."

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"Exception during detection: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response

def detect_min_enclosing(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    """
    Detects a circle by binarizing the image, finding contours, and fitting a minimum enclosing circle.
    
    Args:
        gray_image: 2D uint8 numpy array representing the grayscale input image.
        params: Dictionary containing thresholding parameters:
            - binary_thresh (int, optional): Threshold value. If None, Otsu's optimal threshold is used.
            - min_area (int): Minimum contour area. Defaults to 100.
            
    Returns:
        Dict adhering to the unified API response format.
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale numpy array")

        binary_thresh = params.get('binary_thresh', None)
        min_area = params.get('min_area', 100)

        # Binarize
        if binary_thresh is None:
            # Otsu's thresholding
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, thresh = cv2.threshold(gray_image, binary_thresh, 255, cv2.THRESH_BINARY)

        response['debug']['edge_map'] = thresh

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            response['success'] = False
            response['diagnostics']['message'] = "No contours found in binary image."
            return response

        # Filter by area and find the largest
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        if not valid_contours:
            response['success'] = False
            response['diagnostics']['message'] = "No contours found exceeding minimum area."
            return response

        largest_contour = max(valid_contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        response['center'] = [int(x), int(y)]
        response['radius'] = int(radius)
        response['success'] = True
        response['diagnostics']['message'] = f"Found {len(valid_contours)} valid contours. Bounding largest."

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"Exception during detection: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response

def detect_canny_hough(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    """
    Detects a circle by explicitly running Canny edge detection followed by Hough gradient.
    
    Args:
        gray_image: 2D uint8 numpy array representing the grayscale input image.
        params: Dictionary containing parameters:
            - canny_low (int)
            - canny_high (int)
            - morphological_close (bool)
            - dp, minDist, param1, param2, minRadius, maxRadius (for Hough)
            
    Returns:
        Dict adhering to the unified API response format.
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale numpy array")

        canny_low = params.get('canny_low', 50)
        canny_high = params.get('canny_high', 150)
        use_morph = params.get('morphological_close', True)

        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param1 = params.get('param1', 50)
        param2 = params.get('param2', 30)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        response['debug']['edge_map'] = edges

        # Important: param1 in HoughCirlces controls the *internal* Canny. 
        # When passing an edge map, we should ideally use a different variant (like HOUGH_GRADIENT_ALT or standard HOUGH_GRADIENT but it still expects grayscale).
        # Standard HoughCircles DOES NOT take binary edge maps gracefully for HOUGH_GRADIENT (it runs Canny internally again).
        # We will pass the binarized *edge* map as the grayscale image, which will threshold internally, but effectively act on our edges.
        
        circles = cv2.HoughCircles(
            edges, 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist,
            param1=1, # Very low to accept our pre-computed edges
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))
            best_circle = circles[0, 0]
            response['center'] = [int(best_circle[0]), int(best_circle[1])]
            response['radius'] = int(best_circle[2])
            response['success'] = True
            response['diagnostics']['message'] = "Found circle via Canny + Hough."
            response['debug']['votes'] = circles[0]
        else:
            response['diagnostics']['message'] = "No circles found in Canny edge map."

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"Exception during detection: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response

def detect_multi_scale_fusion(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    """
    Detects a circle by fusing multiple Canny edge maps across different threshold pairs, then applying Hough.
    
    Args:
        gray_image: 2D uint8 numpy array representing the grayscale input image.
        params: Dictionary containing parameters:
            - canny_pairs (list of tuples): e.g., [(20, 80), (50, 150), (100, 200)]
            - fusion_thresh (float): [0.0 - 1.0] threshold for the normalized fusion map.
            - morphological_close (bool)
            - dp, minDist, param2, minRadius, maxRadius (for Hough)
            
    Returns:
        Dict adhering to the unified API response format.
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("Input must be a 2D grayscale numpy array")

        # Default pairs if none provided
        canny_pairs = params.get('canny_pairs', [(30, 90), (50, 150), (80, 200)])
        fusion_thresh = params.get('fusion_thresh', 0.3)
        use_morph = params.get('morphological_close', True)

        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param2 = params.get('param2', 25)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Accumulate edge maps
        fusion_map = np.zeros(gray_image.shape, dtype=np.float32)
        
        for low, high in canny_pairs:
            edges = cv2.Canny(blurred, low, high)
            fusion_map += (edges.astype(np.float32) / 255.0)

        # Normalize to 0-1
        num_pairs = len(canny_pairs)
        if num_pairs > 0:
            fusion_map = fusion_map / float(num_pairs)
            
        response['debug']['fusion_map'] = fusion_map.copy()

        # Threshold the fusion map to create a strong binary edge map
        _, fused_binary = cv2.threshold((fusion_map * 255).astype(np.uint8), int(fusion_thresh * 255), 255, cv2.THRESH_BINARY)
        
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)

        response['debug']['edge_map'] = fused_binary

        # Hough on the fused map
        circles = cv2.HoughCircles(
            fused_binary, 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist,
            param1=1, # Very low since we already have edges
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))
            best_circle = circles[0, 0]
            response['center'] = [int(best_circle[0]), int(best_circle[1])]
            response['radius'] = int(best_circle[2])
            response['success'] = True
            response['diagnostics']['message'] = f"Found circle via multi-scale fusion ({num_pairs} scales)."
            response['debug']['votes'] = circles[0]
        else:
            response['diagnostics']['message'] = "No circles found in fused edge map."

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"Exception during multi-scale fusion: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response
