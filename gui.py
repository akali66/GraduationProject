import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import os

from detectors import (
    detect_hough,
    detect_min_enclosing,
    detect_canny_hough,
    detect_multi_scale_fusion
)

try:
    from eval_metrics import compute_edge_coverage, compute_hough_confidence
except ImportError as e:
    import traceback
    traceback.print_exc()
    # 兼容没有导入包的早期环境阶段
    def compute_edge_coverage(*args, **kwargs): return 0.0
    def compute_hough_confidence(*args, **kwargs): return 0.0

METHODS = {
    "霍夫圆变换": {
        "func": detect_hough,
        "params": ["dp", "minDist", "param1", "param2", "minRadius", "maxRadius"]
    },
    "轮廓最小外接圆": {
        "func": detect_min_enclosing,
        "params": ["binary_thresh", "min_area", "max_area_ratio", "min_circularity", "selection_mode"]
    },
    "基于Canny边缘的单尺度霍夫检测": {
        "func": detect_canny_hough,
        "params": ["canny_low", "canny_high", "dp", "minDist", "param2", "minRadius", "maxRadius"]
    },
    "多尺度边缘融合检测": {
        "func": detect_multi_scale_fusion,
        "params": ["fusion_thresh", "dp", "minDist", "param2", "minRadius", "maxRadius"]
    }
}

PARAM_DEF = {
    "dp": {"label": "分辨率比例(dp)", "type": "float", "range": (1.0, 3.0), "default": 1.2, "step": 0.1},
    "minDist": {"label": "最小圆心距", "type": "int", "range": (1, 500), "default": 30},
    "param1": {"label": "梯度阈值(P1)", "type": "int", "range": (1, 300), "default": 50},
    "param2": {"label": "累加器阈值(P2)", "type": "int", "range": (1, 200), "default": 30},
    "minRadius": {"label": "最小半径", "type": "int", "range": (1, 200), "default": 10},
    "maxRadius": {"label": "最大半径", "type": "int", "range": (10, 500), "default": 300},
    "binary_thresh": {"label": "二值化阈值(0=大津法)", "type": "int", "range": (0, 255), "default": 0},
    "min_area": {"label": "最小轮廓面积", "type": "int", "range": (10, 10000), "default": 100},
    "max_area_ratio": {"label": "最大面积比例极限", "type": "float", "range": (0.1, 1.0), "default": 0.95, "step": 0.01},
    "min_circularity": {"label": "最小圆度", "type": "float", "range": (0.01, 1.0), "default": 0.2, "step": 0.01},
    "selection_mode": {"label": "选取策略(面积|圆度|中心)", "type": "int", "range": (1, 3), "default": 1},
    "canny_low": {"label": "Canny 边缘低阈值", "type": "int", "range": (0, 255), "default": 50},
    "canny_high": {"label": "Canny 边缘高阈值", "type": "int", "range": (0, 255), "default": 150},
    "fusion_thresh": {"label": "融合阈值", "type": "float", "range": (0.1, 0.9), "default": 0.3, "step": 0.05}
}

class CircleApp:
    def __init__(self, root):
        """
        全应用的主控制器
        root: Tkinter的主窗口传递对象
        """
        self.root = root
        self.root.title("基于多尺度边缘融合的钻孔图像圆心鲁棒检测方法")
        self.root.geometry("1400x800")
        
        # 定义核心图像载体变量, 全局共享供所有算法和绘图使用
        self.original_image_pil = None # 从硬盘加载上来的 PIL 彩色原图对象
        self.grey_np = None            # 供OpenCV计算使用的 numpy 矩阵模型（灰阶单色道版本）
        self.color_np = None           # 供OpenCV结果绘图用的 numpy 矩阵模型（BGR三色道全彩版）
        
        # TKinter使用的GUI级别缓存
        self.left_img_tk = None
        self.result_images_tk = [] 
        self.export_image_pil = None
        
        # 通过 PARAM_DEF 配置字典，自动化构建存放每个调谐滑块数据的 TK内部变量 (IntVar / DoubleVar)
        self.param_vars = {}
        for k, v in PARAM_DEF.items():
            if v["type"] == "int":
                self.param_vars[k] = tk.IntVar(value=v["default"])
            else:
                self.param_vars[k] = tk.DoubleVar(value=v["default"])
                
        # 初始化构造屏幕可见的窗口元素
        self.setup_ui()
        
    def setup_ui(self):
        """
        界面构架方法。
        分为左中右三大竖向面板：
        1. Input 原始图像导入区
        2. Config 运算流程选项控制器
        3. Output 带有覆盖遮罩的绘图显示区域
        """
        main_pane = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # --- LEFT PANE (左侧显示被读入的初版图) ---
        self.left_frame = ttk.LabelFrame(main_pane, text="输入图像")
        main_pane.add(self.left_frame, minsize=400)
        
        self.btn_load = ttk.Button(self.left_frame, text="加载图像", command=self.load_image)
        self.btn_load.pack(pady=5)
        
        self.left_label = ttk.Label(self.left_frame, text="未加载图像", anchor=tk.CENTER)
        self.left_label.pack(fill=tk.BOTH, expand=True)

        # --- MIDDLE PANE (中间交互按键和滑块) ---
        self.mid_frame = ttk.LabelFrame(main_pane, text="控制面板")
        main_pane.add(self.mid_frame, minsize=350)
        
        # 下拉选单：选择将以哪种计算机视觉手段侦测
        ttk.Label(self.mid_frame, text="检测方法选择：").pack(pady=(10,0))
        self.method_var = tk.StringVar(value=list(METHODS.keys())[0])
        self.dropdown = ttk.Combobox(self.mid_frame, textvariable=self.method_var, values=list(METHODS.keys()), state="readonly")
        self.dropdown.pack(fill=tk.X, padx=10, pady=5)
        self.dropdown.bind("<<ComboboxSelected>>", self.on_method_changed)
        
        self.params_frame = ttk.Frame(self.mid_frame)
        self.params_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 装载完界面自动生成默认第一项算法下的专属细调条目
        self.on_method_changed()
        
        ttk.Separator(self.mid_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 激活四宫格比较对比视角
        self.compare_var = tk.BooleanVar(value=False)
        self.chk_compare = ttk.Checkbutton(self.mid_frame, text="启用对比模式", variable=self.compare_var)
        self.chk_compare.pack(pady=5)
        
        # 通知执行侦测方法
        self.btn_run = ttk.Button(self.mid_frame, text="运行检测", command=self.run_detection)
        self.btn_run.pack(pady=10, fill=tk.X, padx=10)
        
        # 将分析好的连带识别圆截出存至磁盘
        self.btn_export = ttk.Button(self.mid_frame, text="保存当前结果图", command=self.export_results, state=tk.DISABLED)
        self.btn_export.pack(pady=(0,10), fill=tk.X, padx=10)
        
        ttk.Separator(self.mid_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        btn_frame = ttk.Frame(self.mid_frame)
        btn_frame.pack(fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,2))
        ttk.Button(btn_frame, text="加载配置", command=self.load_config).pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=(2,0))

        # --- RIGHT PANE ---
        self.right_frame = ttk.LabelFrame(main_pane, text="结果视图")
        main_pane.add(self.right_frame, minsize=500)
        
        self.right_single_frame = ttk.Frame(self.right_frame)
        self.right_single_frame.pack(fill=tk.BOTH, expand=True)
        self.right_single_label = ttk.Label(self.right_single_frame, text="结果将显示在此处", anchor=tk.CENTER)
        self.right_single_label.pack(fill=tk.BOTH, expand=True)
        self.right_single_text = ttk.Label(self.right_single_frame, text="", anchor=tk.CENTER, font=("Helvetica", 11, "bold"))
        self.right_single_text.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        self.right_grid_frame = ttk.Frame(self.right_frame)
        self.grid_blocks = []
        for i in range(4):
            block = ttk.Frame(self.right_grid_frame, relief=tk.SUNKEN)
            block.grid(row=i//2, column=i%2, sticky="nsew", padx=2, pady=2)
            lbl_img = ttk.Label(block, text=f"图块 {i+1} 图像", anchor=tk.CENTER)
            lbl_img.pack(fill=tk.BOTH, expand=True)
            lbl_txt = ttk.Label(block, text=f"图块 {i+1} 信息", anchor=tk.CENTER, justify=tk.CENTER, font=("Helvetica", 9))
            lbl_txt.pack(side=tk.BOTTOM, fill=tk.X)
            self.grid_blocks.append({"img": lbl_img, "txt": lbl_txt})
            
        self.right_grid_frame.columnconfigure(0, weight=1, uniform="col")
        self.right_grid_frame.columnconfigure(1, weight=1, uniform="col")
        self.right_grid_frame.rowconfigure(0, weight=1, uniform="row")
        self.right_grid_frame.rowconfigure(1, weight=1, uniform="row")
        
        # --- BOTTOM PANE ---
        self.status_var = tk.StringVar(value="准备就绪")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_method_changed(self, event=None):
        """
        根据下拉菜单中切换的不同算法名字，动态向中间的“参数面板”填充该算法独有的调节滑块。
        先销毁全部旧控件，然后根据配置字典(PARAM_DEF)新建滑块。
        """
        for widget in self.params_frame.winfo_children():
            widget.destroy()
            
        method = self.method_var.get()
        param_keys = METHODS[method]["params"]
        
        for k in param_keys:
            pdef = PARAM_DEF[k]
            frame = ttk.Frame(self.params_frame)
            frame.pack(fill=tk.X, pady=2)
            
            lbl = ttk.Label(frame, text=pdef["label"], width=20)
            lbl.pack(side=tk.LEFT)
            
            val_lbl = ttk.Label(frame, width=5)
            val_lbl.pack(side=tk.RIGHT)
            
            # 使用 trace_add 对滑块绑定的数据对象建立监听，当拖动时动态刷新右侧的数字标签
            def update_lbl(*args, var=self.param_vars[k], l=val_lbl, is_int=(pdef["type"]=="int")):
                try:
                    val = var.get()
                    if is_int:
                        l.config(text=str(int(val)))
                    else:
                        l.config(text=f"{val:.2f}")
                except:
                    pass
                    
            self.param_vars[k].trace_add("write", update_lbl)
            update_lbl()
            
            slider = tk.Scale(frame, from_=pdef["range"][0], to=pdef["range"][1],
                              variable=self.param_vars[k], orient=tk.HORIZONTAL, 
                              resolution=pdef.get("step", 1), showvalue=0)
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

    def load_image(self):
        """打开Windows系统的原生文件选择窗口载入图片供系统分析"""
        path = filedialog.askopenfilename(filetypes=[("图像文件", "*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        
        # 利用 OpenCV 把图像从硬盘读进内存，这里读入的色彩通道排列默认是 BGR
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            messagebox.showerror("错误", "无法加载该图像。")
            return
            
        # 预先拆分好图像供各自工具使用 (CV2算法常要灰阶图找边, 界面及着色则需要RGB全彩)
        self.color_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # PIL 库预期拿到 RGB 色系，不转色全变蓝底 
        self.grey_np = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        self.original_image_pil = Image.fromarray(self.color_np)
        self.display_image(self.original_image_pil, self.left_label)
        self.status_var.set(f"已加载: {os.path.basename(path)} | 尺寸: {self.color_np.shape[1]}x{self.color_np.shape[0]}")
        self.btn_export.config(state=tk.DISABLED)
        
    def display_image(self, pil_img, label_widget, target_size=None):
        """
        动态计算显示区域的大小，并将传入的高清原图通过高级重采样按比例缩放塞进显示区，防止程序卡死变型
        """
        if target_size is not None:
            w, h = target_size
        else:
            # update_idletasks 让TK强制结算并绘制前置积压事件，从而能拿到它接下来准确占用的坐标空间大小 winfo_width
            self.root.update_idletasks()
            w, h = label_widget.winfo_width(), label_widget.winfo_height()
            
        # 万一窗口还在微小化初始态，给个默认保底画幅以防缩放除错
        if w < 50 or h < 50:
            w, h = 600, 600
        
        img_copy = pil_img.copy()
        # thumbnail 是原地依据指定边界限制按相机的原定宽高比 (AspectRatio) 实行缩水操作
        img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
        
        # 将 Pillow 下的图形对象封装进 Tkinter 的渲染接口支持模式
        tk_img = ImageTk.PhotoImage(img_copy)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img 
        
    def _draw_result(self, cv_rgb, result, method_name=""):
        """
        在计算机视觉返回成功的条件下，利用坐标向 numpy 矩阵图像层上叠画覆盖物 (红色的正圆+中心的蓝点)。
        顺带组合构建将要在界面上展现出来的解释字段文本。
        """
        res_img = cv_rgb.copy()
        text = ""
        result['method'] = method_name # 记录当前用的算法名进字典，方便后续制表导出
        
        if result['success']:
            x, y = result['center']
            r = result['radius']
            # Draw circle boundary (画笔涂在像素上, (255,0,0)是RGB模式下的大红)
            cv2.circle(res_img, (x, y), r, (255, 0, 0), max(2, int(r * 0.05)))
            # Draw center cross (Green)
            cv2.drawMarker(res_img, (x, y), (0, 255, 0), cv2.MARKER_CROSS, max(10, int(r * 0.2)), 2)
            
            # 使用新构建的边缘重合度和置信度进行评估
            edge_map = result.get('debug', {}).get('edge_map')
            if edge_map is not None:
                cov = compute_edge_coverage((x, y), r, edge_map)
                conf = compute_hough_confidence(result.get('debug', {}), (x, y), r)
            else:
                cov, conf = 0.0, 0.0
                
            result['coverage'] = cov
            result['confidence'] = conf
            
            text = f"{method_name}\n中心:({x},{y}) 半径:{r} | 覆盖率:{cov*100:.1f}% | 置信度:{conf*100:.1f}%"
        else:
            text = f"{method_name}\n检测失败"
            
        # 不再在此处绘制文字，而是仅仅返回渲染好标记的图像和提取出的文本供外部 Label 使用
        return res_img, text

    def _create_export_image_with_text(self, pil_img, text):
        """将文字拼接到PIL图下方，生成带有文字的导出专用图像。"""
        w, h = pil_img.size
        # 为了兼容不同的PIL版本和系统环境，手动测算文字行数预留高度
        lines = text.split('\n')
        line_height = max(30, int(w * 0.03)) 
        text_h = len(lines) * line_height
        text_margin = max(15, int(w * 0.02))
        
        # 建立一张白色背景的新图 (高 = 原图高 + 边距 + 文字区高度 + 边距)
        full_h = h + text_margin * 2 + text_h
        new_img = Image.new('RGB', (w, full_h), color=(255, 255, 255))
        new_img.paste(pil_img, (0, 0))
        
        draw = ImageDraw.Draw(new_img)
        # 尝试加载中文字体，失败则使用系统级默认字体
        try:
            # Win下的微软雅黑
            font = ImageFont.truetype("msyh.ttc", line_height - 5)
        except Exception:
            try:
                # 备用黑体
                font = ImageFont.truetype("simhei.ttf", line_height - 5)
            except Exception:
                font = ImageFont.load_default()
                
        # 逐行写字，保持居中
        y_cursor = h + text_margin
        for line in lines:
            try:
                # 新版 Pillow
                bbox = draw.textbbox((0, 0), line, font=font)
                line_w = bbox[2] - bbox[0]
            except Exception:
                # 旧版
                line_w = font.getsize(line)[0] if hasattr(font, 'getsize') else len(line)*15
                
            x_cursor = max(0, (w - line_w) // 2)
            draw.text((x_cursor, y_cursor), line, fill=(0, 0, 0), font=font)
            y_cursor += line_height
            
        return new_img

    def run_detection(self):
        """
        开始运行检测。核心引擎控制器。
        根据用户选择的 “单图模式(可精细调参)” 或 “四宫格模式(同时运行全量算法比较)” 实行运算路由。
        """
        if self.grey_np is None:
            messagebox.showwarning("警告", "请先加载图像！")
            return
            
        # 将鼠标指针变为等待模式，防止多次点击导致多次挂起计算
        self.root.config(cursor="watch")
        self.root.update()
        
        try:
            # ======= 【情形A：不勾选对比模式，即单独算法针对调参测试】 =======
            if not self.compare_var.get():
                # 切回独立的单面板显重视图布局
                self.right_grid_frame.pack_forget()
                self.right_single_frame.pack(fill=tk.BOTH, expand=True)
                
                # 读取并寻找目标挂载点，开始对用户输入提取表单当前数值
                method_name = self.method_var.get()
                func = METHODS[method_name]["func"]
                
                # 打包构建参数发送字典
                params = {k: self.param_vars[k].get() for k in METHODS[method_name]["params"]}
                if "binary_thresh" in params and params["binary_thresh"] == 0:
                    params["binary_thresh"] = None # 当值为0时，表示使用 cv2.THRESH_OTSU 全自动解析替代它
                    
                if method_name == "多尺度边缘融合检测":
                    # 因为UI的滑块很难表示一个列表组套圈，这部分的多配置硬编码传入底层
                    params["canny_pairs"] = [(30, 90), (50, 150), (80, 200)] 
                
                # 注入灰阶源图片，正式调用四种内部算法中的其中一种，返回统一数据模型 (dict)
                res = func(self.grey_np, params)
                
                # 处理绘图渲染以及提示信息的着色
                if res['success']:
                    out_rgb, txt = self._draw_result(self.color_np, res, method_name)
                    
                    # 界面上依然只展示原图不受文字框影响，文字交由Label展示
                    pil_display = Image.fromarray(out_rgb)
                    self.display_image(pil_display, self.right_single_label)
                    self.right_single_text.config(text=txt, foreground="green")  # 成功染绿
                    
                    # 导出对象则单独生成一张带有白底黑字的拼接全景图
                    self.export_image_pil = self._create_export_image_with_text(pil_display, txt)
                    
                    self.status_var.set(f"检测成功！ {txt} | 耗时: {res['diagnostics']['elapsed_ms']:.1f}ms")
                else:
                    self.export_image_pil = None
                    self.right_single_label.config(image="", text="检测失败：\n" + res['diagnostics']['message'])
                    self.right_single_text.config(text=f"{method_name} - 检测失败", foreground="red") # 失败染红
                    messagebox.showerror("失败", "当前方法未能检测到有效圆。\n请尝试调整参数。")
                    self.status_var.set(f"检测失败: {res['diagnostics']['message']}")
                    
            # ======= 【情形B：勾选了对比模式，四个算法并行同时展示】 =======        
            else:
                self.right_single_frame.pack_forget()
                self.right_grid_frame.pack(fill=tk.BOTH, expand=True)
                self.root.update_idletasks()
                
                # 预先获取统一的绘图区域大小，避免在循环中因前面图像的置入导致后续方块尺寸剧变
                grid_w = self.right_grid_frame.winfo_width() // 2 - 10
                grid_h = self.right_grid_frame.winfo_height() // 2 - 30 # 留出文本的高
                target_size = (grid_w, grid_h)
                
                success_count = 0
                
                # 为了支持包含单独文字的对比大图导出，收集所有的带文字图块
                collage_blocks = []
                
                # 轮询自带的四个特征提取方法执行侦测
                for idx, (m_name, m_info) in enumerate(METHODS.items()):
                    func = m_info["func"]
                    m_params = {k: self.param_vars[k].get() for k in m_info["params"]}
                    if "binary_thresh" in m_params and m_params["binary_thresh"] == 0:
                        m_params["binary_thresh"] = None
                    if m_name == "多尺度边缘融合检测":
                        m_params["canny_pairs"] = [(30, 90), (50, 150), (80, 200)]
                        
                    res = func(self.grey_np, m_params)
                    out_rgb, txt = self._draw_result(self.color_np, res, m_name)
                    
                    if res['success']: 
                        success_count += 1
                        self.grid_blocks[idx]["txt"].config(text=txt, foreground="green")
                    else:
                        self.grid_blocks[idx]["txt"].config(text=txt, foreground="red")
                    
                    # 放入屏幕矩阵控件中
                    pil_img = Image.fromarray(out_rgb)
                    self.display_image(pil_img, self.grid_blocks[idx]["img"], target_size=target_size)
                    
                    # 生成用于导出的单带字小卡片放入数组
                    collage_blocks.append(self._create_export_image_with_text(pil_img, txt))
                    
                # 以具有小缝隙的方式将四张图卡进行拼接 (用于保存的导出图)
                if len(collage_blocks) == 4:
                    block_w, block_h = collage_blocks[0].size
                    padding = int(block_w * 0.02) # 用于分清边界的间隙
                    bg_w = block_w * 2 + padding * 3
                    bg_h = block_h * 2 + padding * 3
                    # 使用容易区分边境的淡灰色作为分隔背景
                    collage_img = Image.new('RGB', (bg_w, bg_h), color=(230, 230, 230))
                    
                    for idx, b_img in enumerate(collage_blocks):
                        row, col = idx // 2, idx % 2
                        x_pos = padding + col * (block_w + padding)
                        y_pos = padding + row * (block_h + padding)
                        collage_img.paste(b_img, (x_pos, y_pos))
                        
                    self.export_image_pil = collage_img
                    
                self.status_var.set(f"对比模式已完成。 {success_count}/4 种算法成功检测并找到了圆心。")
                
            # 开启允许按动底部的导出图片操作并恢复指针形状    
            self.btn_export.config(
                state=tk.NORMAL if self.export_image_pil else tk.DISABLED,
                text="保存对比图" if self.compare_var.get() else "保存当前结果图"
            )
            
        except Exception as e:
            messagebox.showerror("异常", f"发生意外错误：{str(e)}")
        finally:
            self.root.config(cursor="")
            
    def export_results(self):
        """利用PIL自身库将带标记物的成品图储存"""
        if not self.export_image_pil:
            return
            
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG 图像", "*.png")])
        if path:
            self.export_image_pil.save(path)
            messagebox.showinfo("导出成功", f"结果图像已保存至： {path}")
            self.status_var.set(f"已将结果视图保存为 {path}")

    def save_config(self):
        """将用户自调好的各项超参数一键打包储存在一个json文件中方便长期继承"""
        data = {k: v.get() for k, v in self.param_vars.items()}
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON 配置文件", "*.json")])
        if path:
            with open(path, 'w') as f:
                json.dump(data, f, indent=4)
            self.status_var.set(f"当前配置参数已安全保存至 {path}")
            
    def load_config(self):
        path = filedialog.askopenfilename(filetypes=[("JSON 配置文件", "*.json")])
        if path:
            with open(path, 'r') as f:
                data = json.load(f)
            for k, v in data.items():
                if k in self.param_vars:
                    self.param_vars[k].set(v)
            self.status_var.set(f"本地参数已成功从 {path} 中加载并覆盖更新。")

if __name__ == "__main__":
    root = tk.Tk()
    app = CircleApp(root)
    root.mainloop()
