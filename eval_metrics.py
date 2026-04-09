import numpy as np
import cv2

def compute_edge_coverage(circle_center, radius, edge_map, samples=360, tol=2):
    if edge_map is None or radius <= 0 or circle_center is None:
        return 0.0

    h, w = edge_map.shape

    # 1. 在同样大小的空白画布上画出1像素宽的“理想数学圆”
    ideal_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ideal_circle, tuple(int(x) for x in circle_center), int(radius), 1, thickness=1)

    # 理想圆的总像素数（最精确完美的周长分母）
    total_ideal_pixels = np.sum(ideal_circle == 1)
    if total_ideal_pixels == 0:
        return 0.0

    # 2. 为了模拟找边缘时的容差(tol=2)，对真实的边缘图进行矩形核膨胀（变粗）
    # kernel size = 2*tol+1，若 tol=2 则是 5x5 的全1核。等于边缘向外四面八方扩散2像素。
    kernel_size = 2 * tol + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_edge_map = cv2.dilate(edge_map, kernel, iterations=1)

    # 3. 统计命中：在理想数学圆经过的所有独立像素里，看看膨胀扩展出的宽带真实边缘图有没有盖到它
    hit_pixels = dilated_edge_map[ideal_circle == 1]
    matched = np.sum(hit_pixels > 0)

    if total_ideal_pixels == 0:
        return 0.0
    return float(matched) / float(total_ideal_pixels)

def compute_hough_confidence(debug, circle_center, radius):
    map_to_use = debug.get('fusion_map')
    if map_to_use is None:
        map_to_use = debug.get('edge_map')

    if map_to_use is None or radius <= 0 or circle_center is None:
        return 0.0

    blank = np.zeros_like(map_to_use, dtype=np.uint8)
    cv2.circle(blank, tuple(int(x) for x in circle_center), int(radius), 1, thickness=1)

    max_possible_votes = np.sum(blank == 1)
    if max_possible_votes == 0:
        return 0.0

    hit_values = map_to_use[blank == 1]
    
    if map_to_use.dtype == np.uint8:
        votes = np.sum(hit_values > 0)
        return float(votes) / float(max_possible_votes)
    else:
        votes = np.sum(hit_values)
        return float(votes) / float(max_possible_votes)

def compute_mask_iou(circle_center, radius, mask_img):
    """
    计算 面具交并比 (Mask IoU)，用于评估“基于面”提取算法的拟合度。
    只有二维分割算法（算法二、算法四）适用。
    """
    if mask_img is None or radius <= 0 or circle_center is None:
        return 0.0
    h, w = mask_img.shape
    
    # 按照算出的圆心和半径，生成一个标准的白色实心圆图层
    ideal_circle = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(ideal_circle, tuple(int(x) for x in circle_center), int(radius), 255, thickness=-1)
    
    # 确保掩码图是纯正的二值化图（只有0和255）
    _, mask_binary = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    
    # 交集：两个图都是白色的区域
    intersection = cv2.bitwise_and(ideal_circle, mask_binary)
    # 并集：任意一张图是白色的区域
    union = cv2.bitwise_or(ideal_circle, mask_binary)
    
    inter_area = np.sum(intersection > 0)
    union_area = np.sum(union > 0)
    
    if union_area == 0:
        return 0.0
    return float(inter_area) / float(union_area)
