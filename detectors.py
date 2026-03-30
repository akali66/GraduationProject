import cv2
import numpy as np
import time
from typing import Dict, Any

try:
    from eval_metrics import compute_edge_coverage
except ImportError:
    def compute_edge_coverage(*args, **kwargs): return 1.0

def _get_base_response() -> Dict[str, Any]:
    """生成一个统一的默认返回字典结构，保证调用方能正确解析基本字段"""
    return {
        "center": None,   # 预测圆心的 (x, y) 坐标
        "radius": None,   # 预测圆的半径
        "success": False, # 检测成功与否的标志
        "debug": {},      # 提供中间变量调试使用的字典（如边缘图、投票矩阵等）
        "diagnostics": {"message": "", "elapsed_ms": 0.0} # 诊断信息：提示信息、以及耗费的时间
    }

def detect_hough(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    # 算法一：基于霍夫变换的圆检测 定义函数，返回一个字典

    response = _get_base_response() # 创建答题卡
    start_time = time.time()    # 开始计时
    
    try:
        # 安全检查：是否是灰度图
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("输入的数据必须是 2D 的灰度 numpy 矩阵")
            
        # 读取参数，没有则默认
        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param1 = params.get('param1', 50)
        param2 = params.get('param2', 30)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        # 步骤A: 高斯滤波 模糊处理
        # 应用 5x5 的高斯核做轻微的模糊(平滑处理)，这用来过滤掉图像里微小的噪点，防止产生误导边缘引发多余的圆
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # 将测试Canny图提供给 debug （即将边缘存下来，放在答题卡里）
        response['debug']['edge_map'] = cv2.Canny(blurred, param1 // 2, param1)
        
        # 步骤B: 调用 HoughCircles 算法进行提取
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT,  # 使用梯度法进行搜寻
            dp=dp, 
            minDist=minDist,
            param1=param1, 
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # 步骤C: 对提取结果分析
        if circles is not None and len(circles) > 0:
            # 将小数坐标转整数（像素坐标只能是整数）
            circles = np.uint16(np.around(circles))
            
            # 取第一个圆（得票数最高的）
            best_circle = circles[0, 0]
            
            response['center'] = [int(best_circle[0]), int(best_circle[1])] # 圆心坐标 [0] 为 x 坐标， [1] 为 y 坐标 
            response['radius'] = int(best_circle[2]) # 圆的半径 [2] 为半径
            response['success'] = True # 标记检测成功
            response['diagnostics']['message'] = f"共寻找到 {circles.shape[1]} 个候选圆，已抛出最优解。" # 记录诊断信息
            response['debug']['votes'] = circles[0] # 把所有候选圆也存起来
        else:
            response['diagnostics']['message'] = "未在该图和参数下找到能构成完整的圆。" # 没找到圆的情况

    except Exception as e:
        response['success'] = False # 标记检测失败
        response['diagnostics']['message'] = f"运行过程发生异常: {str(e)}" # 捕获并记录异常信息
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0 # 记录耗时
        return response

def detect_min_enclosing(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:

    # 初始化
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("输入的数据必须是 2D 的灰度 numpy 矩阵") # 安全检查：是否是灰度图

        binary_thresh = params.get('binary_thresh', None) # 黑白分割的阈值
        min_area = params.get('min_area', 100) # 最小面积，小于100的忽略
        max_area_ratio = params.get('max_area_ratio', 0.95) # 最大面积比例（排除占画面95%以上的轮廓）
        min_circularity = params.get('min_circularity', 0.2) # 最小圆度（排除圆度低于0.2的轮廓）

        # 二值化：把灰度图变成只有纯黑(0)和纯白(255)的图
        if binary_thresh is None or binary_thresh == 0:
            # 若传入空或0，需先防止噪点引发 切分翻车
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            # 用大津法自动找最佳分割点
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 直接按这个阈值分割
            _, thresh = cv2.threshold(gray_image, binary_thresh, 255, cv2.THRESH_BINARY)

        # 保存二值图
        response['debug']['edge_map'] = thresh

        # 找轮廓，采用 RETR_LIST 提取所有层级轮廓，避免被大面积背景层包裹的内部孔洞被忽略（能同时找到外轮廓和内轮廓，不会遗漏）
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            response['success'] = False
            response['diagnostics']['message'] = "图像二值化后并未发现可构成团块的封闭线条。"
            return response

        # 计算图片总面积 用于后续的面积上限检查
        img_area = gray_image.shape[0] * gray_image.shape[1]
        # 计算图片中心坐标，为"距离中心最近"策略做准备。
        img_center_x = gray_image.shape[1] / 2.0
        img_center_y = gray_image.shape[0] / 2.0
        
        selection_mode = params.get('selection_mode', 1)  # 1: 面积最大, 2: 最接近正圆, 3: 距离中心最近
        valid_contours_info = []

        # 智能轮廓筛选
        for c in contours:
            area = cv2.contourArea(c)
            # 条件1：面积范围过滤，（去掉背景）
            if min_area <= area <= img_area * max_area_ratio:
                perimeter = cv2.arcLength(c, True)
                # 条件2：计算圆度，初步淘汰那些过于细长或严重不规则的干涉杂块
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > min_circularity:
                        # 计算轮廓质心以备 "中心距离" 策略使用
                        M = cv2.moments(c) # 计算轮廓矩
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])   # 质心x坐标
                            cy = int(M["m01"] / M["m00"])   # 质心y坐标
                        else:
                            cx, cy = 0, 0
                        
                        # 欧几里得距离公式，计算质心到图片中心的距离
                        dist_to_center = ((cx - img_center_x) ** 2 + (cy - img_center_y) ** 2) ** 0.5
                        
                        valid_contours_info.append({
                            "contour": c,
                            "area": area,
                            "circularity": circularity,
                            "dist_to_center": dist_to_center
                        })
        
        if not valid_contours_info:
            response['success'] = False
            response['diagnostics']['message'] = "未找到满足面积限定和基本形态（圆度过低或近似全屏框）的对象。"
            return response

        # 找满足上述基本常理过滤后，从候选列表中挑选最终目标的策略
        if selection_mode == 1:
            best_target = max(valid_contours_info, key=lambda x: x["area"])
            strategy_name = "最大面积"
        elif selection_mode == 2:
            best_target = max(valid_contours_info, key=lambda x: x["circularity"])
            strategy_name = "最佳圆度"
        else:
            best_target = min(valid_contours_info, key=lambda x: x["dist_to_center"])
            strategy_name = "最靠近视图中心"

        largest_contour = best_target["contour"]
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        response['center'] = [int(x), int(y)]
        response['radius'] = int(radius)
        response['success'] = True
        response['diagnostics']['message'] = f"在 {len(valid_contours_info)} 个有效轮廓中，使用[{strategy_name}]策略完成目标提取并拟合。"

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"运行过程发生异常: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response

def detect_canny_hough(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    # 初始化
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("输入的数据必须是 2D 的灰度 numpy 矩阵") # 安全检查

        canny_low = params.get('canny_low', 50)     # 判断模糊边缘点，低阈值：弱边缘判断
        canny_high = params.get('canny_high', 150)  # 判断首发强边缘点，高阈值：强边缘判断
        use_morph = params.get('morphological_close', True) # 是否使用形态学修复,是否修复断裂的边缘

        dp = params.get('dp', 1.2)                  # 检测精度
        minDist = params.get('minDist', 30)         # 圆之间的最小距离
        param1 = params.get('param1', 50)           # 霍夫内部使用的 Canny 边缘检测的高阈值（我们在外部已经做了 Canny，所以这里设置得非常低，甚至1，来让它完全信任我们提供的边缘图，不再自己做二次边缘判断）
        param2 = params.get('param2', 30)           # 投票阈值
        minRadius = params.get('minRadius', 10)     # 最小半径
        maxRadius = params.get('maxRadius', 100)    # 最大半径

        # 步骤 1：自行执行并计算 Canny 以便对原图的边际精准抽取
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)   # 先高斯模糊去噪
        edges = cv2.Canny(blurred, canny_low, canny_high)   # 使用自定义阈值的Canny边缘检测
        
        # 步骤 2：对不连贯的点画进行拼接加粗（闭运算） ：形态学闭运算（修复边缘断裂）
        if use_morph:
            # 创建5x5的椭圆形结构元素，椭圆形，适合修复圆形边缘，操作范围，越大连接能力越强
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # 先膨胀后腐蚀，连接断裂处
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        response['debug']['edge_map'] = edges

        # 步骤 3：这步非常关键，调用普通的Hough，但把 param1 取值到极其微小(1)。
        # 因为入参 edges 此时已经是人为剔除背景后的极简纯白二值边缘线条了，
        # 如果 param1 设置高则导致其系统内置逻辑认为这些不是边致使识别出错崩盘，我们只需要让它专注执行“基于当前明确的二维线图寻找对应的圆形参数”。
        circles = cv2.HoughCircles(
            edges, 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist,
            param1=1, # 让其信任我们的手工 edges 输入层图，不作卡截
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        if circles is not None and len(circles) > 0:
            # 坐标四舍五入转为整数
            circles = np.uint16(np.around(circles))
            # 取第一个圆（置信度最高）
            best_circle = circles[0, 0]
            response['center'] = [int(best_circle[0]), int(best_circle[1])]
            response['radius'] = int(best_circle[2])
            response['success'] = True
            response['diagnostics']['message'] = "通过串联并修复独立定制的 Canny 图纸，拟合目标成功。"
            response['debug']['votes'] = circles[0]
        else:
            response['diagnostics']['message'] = "在提取出的边缘图层中无法寻找出几何上成立的圆。"

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"运行过程发生异常: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response

    """
    【算法四：多尺度边缘特征图融合检测 - 优化版】
    
    基于更新后的架构，避免多尺度偏移、过度融合以及形状畸变引发的伪检：
    1. 引入不同尺度的高斯平滑核与对应阈值的多重组合，并且将大核的权重调低以防偏移；
    2. 基于不同尺度的特征给予加权融合，增强有用结构，削弱偶发高光伪影。
    3. 形态学闭运算采用较小核(默认3x3)，避免误把直线刮痕粘连成圆。
    4. 霍夫检测后再增加边缘重合度校验机制（Edge Coverage Validation），剔除满足投票分数但是实际边缘实体的缺失的"虚假圆"。
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("输入的数据必须是 2D 的灰度 numpy 矩阵")

        # 读取多尺度配置参数：(高斯核大小, Canny低阈值, Canny高阈值, 融合权重)
        # 默认参数设计原则：尺度跨度不宜过大(如3, 5, 7)以防止边缘飘移；对小尺度细节保留一定权重。
        default_scales = [
            (3, 40, 120, 0.4), # 小尺度平滑：捕获细节和真实浅弱边缘
            (5, 30, 90,  0.4), # 中等尺度平滑：补充主体结构、抗部分噪点
            (7, 20, 60,  0.2)  # 较大尺度平滑：抑制强噪声，但为了防特征畸变权重给小一点
        ]
        
        # 兼容老版接口或接收GUI的覆盖
        scale_configs = params.get('scale_configs', default_scales)
        
        fusion_thresh = params.get('fusion_thresh', 0.3)
        use_morph = params.get('morphological_close', True)
        
        # 避免使用(5,5)这样的大核，改默认使用(3,3)避免特征过度粘连
        morph_kernel_size = params.get('morph_kernel_size', 3)
        
        # 新增验证门槛：计算返回的圆周上边缘点的命中率。低于此门槛将判为环境杂纹误导的“心证”伪影。
        min_coverage_thresh = params.get('min_coverage_thresh', 0.4) 

        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param2 = params.get('param2', 25)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        # 建立一张用来收集和接纳不同程度套图的大黑板 (尺寸一致的空白图)，接收浮点结构保留透明度
        fusion_map = np.zeros(gray_image.shape, dtype=np.float32)
        
        # 步骤 1：遍历多尺度检测并且加权到实数底板
        for kernel_size, low, high, weight in scale_configs:
            blurred = cv2.GaussianBlur(gray_image, (kernel_size, kernel_size), 0)
            edges = cv2.Canny(blurred, low, high)
            fusion_map += (edges.astype(np.float32) / 255.0) * weight

        # 存给debug方便检视这套精美的概率叠图
        response['debug']['fusion_map'] = fusion_map.copy()

        # 步骤 2：对融合图进行截断，将高于阈值线全数转变回二值图块，不合格的归0
        _, fused_binary = cv2.threshold((fusion_map * 255).astype(np.uint8), int(fusion_thresh * 255), 255, cv2.THRESH_BINARY)
        
        # 步骤 3：轻量级修复微小断点
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
            fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)

        response['debug']['edge_map'] = fused_binary

        # 步骤 4：基础提取定位
        circles = cv2.HoughCircles(
            fused_binary, 
            cv2.HOUGH_GRADIENT, 
            dp=dp, 
            minDist=minDist,
            param1=1, 
            param2=param2, 
            minRadius=minRadius, 
            maxRadius=maxRadius
        )

        # 步骤 5：引入几何反向校验（Edge Coverage Validation）
        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))[0] # 转为可遍历独立坐标套
            
            best_circle_found = False
            for c in circles:
                cx, cy, cr = int(c[0]), int(c[1]), int(c[2])
                
                # 调用本检测器外挂体系下的评估函数进行验证（在合并边界图上取点看是否落在轮廓线上）
                cov = compute_edge_coverage((cx, cy), cr, fused_binary)
                
                if cov >= min_coverage_thresh:
                    response['center'] = [cx, cy]
                    response['radius'] = cr
                    response['success'] = True
                    response['diagnostics']['message'] = f"多尺度边缘融合定标完成！重合度校验：{cov*100:.1f}%"
                    response['debug']['votes'] = c
                    best_circle_found = True
                    break
            
            # 若列表找完都没有真实结构存在的，那很可能是结构性噪声诱发的幻影
            if not best_circle_found:
                response['diagnostics']['message'] = f"捕捉到 {len(circles)} 个潜在结果，但重合度均低于门槛({min_coverage_thresh*100:.0f}%)，被系统作为光斑伪影遗弃。"

        else:
            response['diagnostics']['message'] = "融合网段内因边缘结构弱化碎片化，仍未能寻汇出核心圆心。"

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"多尺度组合时产生错误阻塞: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response



try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

_yolo_model = None

def get_yolo_model(model_path='yolov8n-seg.pt'):
    global _yolo_model
    if _yolo_model is None:
        if YOLO is not None:
            _yolo_model = YOLO(model_path)
    return _yolo_model

def detect_yolo_segmentation(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    # 【算法四：YOLOv8 实例分割大模型检测】
    response = _get_base_response()
    start_time = time.time()
    try:
        model = get_yolo_model(params.get('yolo_model_path', 'yolov8n-seg.pt'))
        if model is None:
            raise RuntimeError('Ultralytics 库未安装或模型加载失败。')
            
        if len(gray_image.shape) == 2:
            rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = gray_image

        results = model.predict(rgb_image, conf=params.get('conf_thresh', 0.25), verbose=False)
        result = results[0]

        if result.masks is None or len(result.masks) == 0:
            response['diagnostics']['message'] = 'YOLO 未能在当前置信度下找到目标'
            response['success'] = False
            return response

        masks = result.masks.data.cpu().numpy() # [N, H, W]
        # 取面积最大的 Mask
        largest_mask_idx = np.argmax([mask.sum() for mask in masks])
        mask = masks[largest_mask_idx]

        # 【优化1】：使用双线性插值，消除原始 Nearest 查值造成的锯齿边缘
        mask_resized = cv2.resize(mask, (rgb_image.shape[1], rgb_image.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask_uint8 = (mask_resized * 255).astype(np.uint8)
        # 二值化并做基础平滑，使边缘轮廓更贴合圆的几何特性
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
        mask_uint8 = cv2.GaussianBlur(mask_uint8, (5, 5), 0)
        _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)

        response['debug']['edge_map'] = mask_uint8

        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # 【优化2】：用图像矩(Moments)计算质心，用等效面积极距计算半径
            # 相比于最小外接圆(minEnclosingCircle)，质心法完全免疫掩码边缘孤立噪点造成的圆心严重偏移
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(largest_contour)
                # 等效半径 (假设是完美的圆，Area = π * R^2)
                equivalent_radius = np.sqrt(area / np.pi)
                
                response['center'] = [cX, cY]
                response['radius'] = int(equivalent_radius)
                response['success'] = True
                response['diagnostics']['message'] = 'YOLO Segmentation (质心等效圆法重计算成功)'
            else:
                response['diagnostics']['message'] = 'Mask 寻找核心几何矩量失败'
        else:
            response['diagnostics']['message'] = 'Mask 寻找特征轮廓失败'
    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f'YOLO报错: {str(e)}'
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response
