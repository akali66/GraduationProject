import numpy as np
import cv2

def compute_edge_coverage(circle_center, radius, edge_map, samples=360, tol=2):
    """
    评估圆周边界的覆盖率。
    通过在推断的圆周上均匀取样，并在其径向法线的极小范围内寻找边缘图上的边缘点。
    
    参数:
        circle_center: (x, y) 坐标元组
        radius: 圆心半径
        edge_map: 二值化的边缘图 (二维 np.ndarray，包含边缘时元素 > 0)
        samples: 沿圆周的采样点数 (默认360)
        tol: 径向上允许的法向像素容差 (默认±2)
        
    返回:
        coverage (浮点数 0-1): 找到匹配边界的样本点占比
    """
    if edge_map is None or radius <= 0:
        return 0.0

    cx, cy = circle_center
    h, w = edge_map.shape
    matched = 0

    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    for i in range(samples):
        found = False
        # 在径向方向探测 ±tol 个像素范围
        for dt in range(-tol, tol + 1):
            r_curr = radius + dt
            px = int(round(cx + r_curr * cos_t[i]))
            py = int(round(cy + r_curr * sin_t[i]))
            
            if 0 <= px < w and 0 <= py < h:
                if edge_map[py, px] > 0:
                    found = True
                    break
        if found:
            matched += 1

    return float(matched) / samples

def compute_hough_confidence(debug, circle_center, radius, samples=360):
    """
    计算侦测出的圆的置信度。
    如果分析数据内附带了概率融合图 (fusion_map)，优先依赖融合图上各采样点均值。
    否则降级使用二值边缘图的均值作为置信度替代(surrogate)计算。
    """
    map_to_use = debug.get('fusion_map')
    if map_to_use is None:
        map_to_use = debug.get('edge_map') 
        
    if map_to_use is None or radius <= 0 or circle_center is None:
        return 0.0

    cx, cy = circle_center
    h, w = map_to_use.shape
    
    # 极坐标向直角坐标转化采样点并过滤出合法的点
    theta = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    pts_x = np.round(cx + radius * np.cos(theta)).astype(int)
    pts_y = np.round(cy + radius * np.sin(theta)).astype(int)
    
    valid = (pts_x >= 0) & (pts_x < w) & (pts_y >= 0) & (pts_y < h)
    
    if not np.any(valid):
        return 0.0
        
    response = map_to_use[pts_y[valid], pts_x[valid]]
    
    # 若本身图是带有色深的（0-255），需统一转换为0-1
    if response.dtype == np.uint8:
        return float(np.mean(response) / 255.0)
    
    return float(np.mean(response))
