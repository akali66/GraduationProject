import pytest
import numpy as np
import cv2
from detectors import (
    detect_hough, 
    detect_min_enclosing, 
    detect_canny_hough, 
    detect_multi_scale_fusion
)

# --- Helper to create synthetic test images ---
def create_synthetic_circle(size=200, center=(100, 100), radius=50, noise_level=10):
    """Creates a black image with a white circle and optional Gaussian noise."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, center, radius, 255, -1) # Filled circle
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, img.shape).astype(np.float32)
        noisy_img = cv2.add(img.astype(np.float32), noise)
        img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        
    return img

@pytest.fixture
def clean_circle_image():
    # True values: center(100,100), radius=40
    return create_synthetic_circle(size=200, center=(100, 100), radius=40, noise_level=0)

@pytest.fixture
def noisy_circle_image():
    # True values: center(150,150), radius=60
    return create_synthetic_circle(size=300, center=(150, 150), radius=60, noise_level=20)

@pytest.fixture
def bad_input():
    return np.zeros((10, 10, 3), dtype=np.uint8) # 3D array instead of 2D

# --- Reusable assertion logic ---
def assert_detection(result, expected_center, expected_radius, tolerance_pct=0.05):
    assert isinstance(result, dict)
    assert 'center' in result
    assert 'radius' in result
    assert 'success' in result
    assert 'debug' in result
    assert 'diagnostics' in result
    
    assert result['success'] is True, f"Detection failed: {result['diagnostics']['message']}"
    
    # Check center (allow 1-2 pixel deviation)
    cx, cy = result['center']
    ex, ey = expected_center
    assert abs(cx - ex) <= 2, f"Center X {cx} too far from expected {ex}"
    assert abs(cy - ey) <= 2, f"Center Y {cy} too far from expected {ey}"
    
    # Check radius (+/- tolerance)
    r = result['radius']
    er = expected_radius
    tolerance = er * tolerance_pct
    assert abs(r - er) <= tolerance, f"Radius {r} outside {tolerance_pct*100}% tolerance of {er}"


# --- Tests for detect_hough ---
def test_hough_clean(clean_circle_image):
    params = {'minRadius': 20, 'maxRadius': 60}
    res = detect_hough(clean_circle_image, params)
    assert_detection(res, (100, 100), 40)

def test_hough_noisy(noisy_circle_image):
    params = {'minRadius': 40, 'maxRadius': 80, 'param2': 15} # Lower param2 for noise
    res = detect_hough(noisy_circle_image, params)
    assert_detection(res, (150, 150), 60, tolerance_pct=0.10) # Noisy might be slightly off

def test_hough_bad_input(bad_input):
    res = detect_hough(bad_input, {})
    assert res['success'] is False
    assert "2D grayscale" in res['diagnostics']['message']

def test_hough_no_circle():
    empty = np.zeros((100, 100), dtype=np.uint8)
    res = detect_hough(empty, {'minRadius': 10, 'maxRadius': 20})
    assert res['success'] is False
    assert "No circles found" in res['diagnostics']['message']


# --- Tests for detect_min_enclosing ---
def test_min_enclosing_clean(clean_circle_image):
    # Otsu thresholding should be perfect here
    res = detect_min_enclosing(clean_circle_image, {})
    assert_detection(res, (100, 100), 40)

def test_min_enclosing_missing_contours():
    empty = np.zeros((100, 100), dtype=np.uint8)
    res = detect_min_enclosing(empty, {})
    assert res['success'] is False
    assert "No contours" in res['diagnostics']['message']


# --- Tests for detect_canny_hough ---
def test_canny_hough_clean(clean_circle_image):
    params = {'minRadius': 20, 'maxRadius': 60}
    res = detect_canny_hough(clean_circle_image, params)
    assert_detection(res, (100, 100), 40)

def test_canny_hough_bad_input(bad_input):
    res = detect_canny_hough(bad_input, {})
    assert res['success'] is False


# --- Tests for detect_multi_scale_fusion ---
def test_multi_scale_fusion_clean(clean_circle_image):
    params = {'minRadius': 20, 'maxRadius': 60}
    res = detect_multi_scale_fusion(clean_circle_image, params)
    assert_detection(res, (100, 100), 40)
    # Verify fusion map is in debug
    assert 'fusion_map' in res['debug']
    assert isinstance(res['debug']['fusion_map'], np.ndarray)

def test_multi_scale_fusion_bad_input(bad_input):
    res = detect_multi_scale_fusion(bad_input, {})
    assert res['success'] is False
