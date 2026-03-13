import numpy as np
import os
import cv2
from eval_metrics import compute_edge_coverage, compute_hough_confidence

def test_edge_coverage_perfect():
    img_size = 200
    cx, cy, r = 100, 100, 50
    edge_map = np.zeros((img_size, img_size), dtype=np.uint8)
    
    # 借助 CV 画笔强制手动画一个极度完美的纯白圈出来当作测试地基。为防止插值像素断层，线宽给到3像素覆盖
    cv2.circle(edge_map, (cx, cy), r, 255, 3)
    
    coverage = compute_edge_coverage((cx, cy), r, edge_map, samples=360, tol=2)
    # 因为离散化画素点有极其轻微的偏差抖动，所以容限范围测出来要大于0.95就算满分符合
    assert coverage > 0.95

def test_hough_confidence():
    img_size = 200
    cx, cy, r = 100, 100, 50
    fusion_map = np.zeros((img_size, img_size), dtype=np.float32)
    
    # 同样给予较粗的线宽保证抽样点位能踩在实线上
    cv2.circle(fusion_map, (cx, cy), r, 1.0, 3)
    
    debug = {'fusion_map': fusion_map}
    confidence = compute_hough_confidence(debug, (cx, cy), r)
    assert confidence > 0.95


    os.remove(path) # 测试结束后擦屁股删掉
