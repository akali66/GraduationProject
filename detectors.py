import cv2
import numpy as np
import time
from typing import Dict, Any

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

        # 二值化：把灰度图变成只有纯黑(0)和纯白(255)的图
        if binary_thresh is None:
            # 若传入空或0，需先防止噪点引发 切分翻车
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            # 用大津法自动找最佳分割点
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 直接按这个阈值分割
            _, thresh = cv2.threshold(gray_image, binary_thresh, 255, cv2.THRESH_BINARY)

        # 保存二值图
        response['debug']['edge_map'] = thresh

        # 找轮廓，调用 findContours 寻找外围轮廓(RETR_EXTERNAL：仅提取最外层)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 检查是否找到轮廓
        if not contours:
            response['success'] = False
            response['diagnostics']['message'] = "图像二值化后并未发现构成团块的封闭线条。"
            return response

        # 过滤小轮廓，计算每个轮廓的面积，去掉太小的（可能是噪点）。cv2.contourArea(c)：计算轮廓包围的面积
        valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
        
        # 检查过滤后结果
        if not valid_contours:
            response['success'] = False
            response['diagnostics']['message'] = "找到的团块太微小，没有满足面积限定的对象。"
            return response

        # 找最大的轮廓
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        # 计算最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)

        response['center'] = [int(x), int(y)]
        response['radius'] = int(radius)
        response['success'] = True
        response['diagnostics']['message'] = f"从 {len(valid_contours)} 处有效轮廓中寻找到了主目标并实行成功拟合。"

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"运行过程发生异常: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response

def detect_canny_hough(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    """
    【算法三：基于 Canny 边缘的单尺度霍夫检测】
    
    原理说明：
    算法一是直接把图片塞给霍夫寻找，这对于某些对比度低的图片可能会漏掉很多本该发现的边缘。
    此进阶算法则将流程拆解为独立的精准控制步骤：
    1. Canny 边缘检测：我们先行介入，以自定义的精细参数将弱边缘(如锈迹、背景纹理)筛选掉，并把所有真切的强物理轮廓抠出来形成“二进制线条线稿图”。
    2. 形态学闭运算(Morphological Close)：通过形态学操作(膨胀然后腐蚀)能将 Canny 检测中细碎、由于缺口断裂的、并不连贯的零碎线段重新焊接连在一起。提高其组成圆的潜力。
    3. 后接霍夫变换：将处理好的这幅“完美线稿”，投喂给参数极低要求标准（既然线稿被筛选过已保证真实了，我们便可以让霍夫的审核标准放低不卡控）的霍夫圆算法进行圆心还原。
    优势：这是算法 1 与形态学修复相搭配的优质组合工作流。
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("输入的数据必须是 2D 的灰度 numpy 矩阵")

        canny_low = params.get('canny_low', 50)     # 判断模糊边缘点
        canny_high = params.get('canny_high', 150)  # 判断首发强边缘点
        use_morph = params.get('morphological_close', True)

        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param1 = params.get('param1', 50)
        param2 = params.get('param2', 30)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        # 步骤 1：自行执行并计算 Canny 以便对原图的边际精准抽取
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        
        # 步骤 2：对不连贯的点画进行拼接加粗（闭运算）
        if use_morph:
            # 采用 5x5 圆形核心对断裂的缝隙进行弥补
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
            circles = np.uint16(np.around(circles))
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

def detect_multi_scale_fusion(gray_image: np.ndarray, params: dict) -> Dict[str, Any]:
    """
    【算法四：多尺度边缘特征图融合检测】
    
    原理说明：
    由于工件往往因为光照不同或是由于工业打光阴影的不均匀，在整幅图片里，如果我们单独使用一组“低/高”单一阈值 (如同算法三那么干)，
    总是会导致阴影里的“暗边”因数值太低未能测出，同时强光下划痕里的“杂边”又被错误划进被选序列。
    那么本算法思路为：
    1. 平行测试多组环境条件(多尺度遍历)：我们定义多个高低不同的 Canny 边缘阀值组合，对同一张原图跑多遍，这就意味着我们获得了类似“多重曝光(亮度)”后拍出的边缘组合套图。
    2. 加权融合(Fusion)：将这几幅处于各阈值频段所提取出的图的概率全部叠加，只有在这多频段测验中，常年固在原地不变的“真正的钻孔坚硬黑边”，由于每一次能提取出，它将被加权为一个高概率的实影。背景那偶发的高光斑只在其特俗阈值图出现过，叠图后就只剩下稀薄的残影(数值低于 0-1 的一个小数)；
    3. 结果合并：我们定下一个截断（例如超过30%(即0.3)）把“常客”真线抠死做基图发给霍夫寻找。 
    综上：这是个耗时但稳定性与鲁棒性非常强大的抗光遮检测。能有效地把错误光块给平均掉。
    """
    response = _get_base_response()
    start_time = time.time()
    
    try:
        if gray_image is None or len(gray_image.shape) != 2:
            raise ValueError("输入的数据必须是 2D 的灰度 numpy 矩阵")

        # 默认下：第一套抓取暗边，第二套抓取均衡边，第三套抓取严苛强光。
        canny_pairs = params.get('canny_pairs', [(30, 90), (50, 150), (80, 200)])
        fusion_thresh = params.get('fusion_thresh', 0.3)
        use_morph = params.get('morphological_close', True)

        dp = params.get('dp', 1.2)
        minDist = params.get('minDist', 30)
        param2 = params.get('param2', 25)
        minRadius = params.get('minRadius', 10)
        maxRadius = params.get('maxRadius', 100)

        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # 建立一张用来收集和接纳不同程度套图的大黑板 (尺寸一致的空白图)，需用能接收小数类型的 float
        fusion_map = np.zeros(gray_image.shape, dtype=np.float32)
        
        # 步骤 1：遍历多套阈值图并且压制/除以 255 落点于 0~1 的实数，以实现小数级的透明度层加权累积 
        for low, high in canny_pairs:
            edges = cv2.Canny(blurred, low, high)
            fusion_map += (edges.astype(np.float32) / 255.0)

        # 步骤 2：用获取的图纸套数取平局（总共除以张数），让最清晰的点稳定成为值为 1.0 的百分百图点，而闪隐的点处于极低占比小数点
        num_pairs = len(canny_pairs)
        if num_pairs > 0:
            fusion_map = fusion_map / float(num_pairs)
            
        # 存给debug方便检视这套精美的概率叠图
        response['debug']['fusion_map'] = fusion_map.copy()

        # 步骤 3：切断它，将高于阈值线（例如大于0.3，出现过一次以上的保留边界）全数转变成结实的二值图块，不合格的归0淘汰 
        _, fused_binary = cv2.threshold((fusion_map * 255).astype(np.uint8), int(fusion_thresh * 255), 255, cv2.THRESH_BINARY)
        
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            fused_binary = cv2.morphologyEx(fused_binary, cv2.MORPH_CLOSE, kernel)

        response['debug']['edge_map'] = fused_binary

        # 步骤 4：基于抗击光斑污染完成后的究极边缘底版，通过超低要求的Hough完成终点定标寻路
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

        if circles is not None and len(circles) > 0:
            circles = np.uint16(np.around(circles))
            best_circle = circles[0, 0]
            response['center'] = [int(best_circle[0]), int(best_circle[1])]
            response['radius'] = int(best_circle[2])
            response['success'] = True
            response['diagnostics']['message'] = f"跨越 {num_pairs} 道尺度网段的特征图叠洗已结束，高维图谱建查定标完成。"
            response['debug']['votes'] = circles[0]
        else:
            response['diagnostics']['message'] = "融合网段内因边流弱化仍未能汇聚出能够有效判查通过的核心圆模型。"

    except Exception as e:
        response['success'] = False
        response['diagnostics']['message'] = f"多尺度组合时产生错误阻塞: {str(e)}"
        
    finally:
        response['diagnostics']['elapsed_ms'] = (time.time() - start_time) * 1000.0
        return response
