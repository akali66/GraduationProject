import cv2
import numpy as np

def evaluate_circle(gray_img: np.ndarray, x: int, y: int, r: int):
    """
    Evaluates the quality of a detected circle based on edge coverage.
    
    Args:
        gray_img: Grayscale original image (2D numpy array).
        x, y: Center coordinates of the detected circle.
        r: Radius of the detected circle.
        
    Returns:
        coverage (float): Ratio of edge pixels overlapping the calculated circle boundary (0.0 to 1.0).
        confidence (float): Abstract confidence percentage score (0.0 to 100.0).
    """
    if gray_img is None or r <= 0:
        return 0.0, 0.0

    # Extract true edges from the image
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Create the idealized predicted circle boundary
    mask = np.zeros_like(gray_img)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, 1)
    
    # Dilate mask slightly to tolerate a 1~2 pixel alignment deviation
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel)
    
    total_circle_pixels = np.sum(mask > 0)
    if total_circle_pixels == 0:
        return 0.0, 0.0
        
    overlap = np.logical_and(edges > 0, mask > 0)
    coverage = np.sum(overlap) / total_circle_pixels
    
    # Simple heuristic for a 0-100 score. E.g. 80%+ coverage equals nearly 100% confidence
    confidence = min(coverage * 120.0, 100.0)
    
    return float(coverage), float(confidence)
