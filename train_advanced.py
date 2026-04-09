from ultralytics import YOLO
import torch

def train_model():
    print(f"[*] 环境检查 -> CUDA (显卡) 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[*] 显卡型号: {torch.cuda.get_device_name(0)}")

    # 1. 加载 YOLOv8 分割模型 (加载预训练权重，站在巨人的肩膀上)
    model = YOLO('yolov8n-seg.pt')  # 如果你的显卡算力很强，回头我们可以升级成 yolov8s-seg.pt

    # 2. 专业版训练参数配置
    print("[*] 开始执行进阶版训练策略 (学术级深度学习训练)...")
    results = model.train(
        # ================== 基础数据流 ==================
        data='my_dataset/data.yaml',   # 数据集配置文件路径 (务必确认你的路径正确)
        epochs=200,                    # 训练最大轮数 (专业级一般 200-300 轮起步)，让模型充分学习
        imgsz=640,                     # 输入图像缩放尺寸 (默认640，保持和绝大多数基准一致)
        batch=8,                       # 批次大小 (如果显存不够报错，可以改成 4，如果充裕改成 16)
        device=0,                      # 使用第 0 号显卡
        workers=0,                     # Windows 防止多线程读取死锁，先设为0

        # ================== 核心优化与高级超参数 (这是你的论文卖点) ==================
        optimizer='AdamW',             # 优化器: 使用目前学术界最主流、收敛更稳定且抗过拟合最好的 AdamW
        lr0=0.001,                     # 初始学习率 (Learning Rate): 决定模型学习的跨度步伐
        lrf=0.01,                      # 最终学习率衰减因子 (lr0 * lrf)
        patience=50,                   # 早停机制 (Early Stopping): 如果连续 50 轮指标毫无提升，自动停止，防止模型死记硬背(过拟合)
        
        # ================== 高级数据增强策略 (提升论文要求的"鲁棒性") ==================
        mosaic=1.0,                    # 开启马赛克增强 (把4张图拼成1张，极大提升复杂背景和遮挡的识别能力)
        mixup=0.1,                     # 图像混合增强 (进一步提升模型对泥浆等遮挡物的泛化)
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, # HSV 颜色空间随机干扰 (模拟真实的矿井/钻孔各种光照、过曝和极暗的情况)
        degrees=10.0,                  # 随机旋转 (-10 到 10度，模拟探头旋转拍摄带来的倾斜)
        flipud=0.5,                    # 随机上下翻转 (模拟探头上下移动的各种倒置视角)
        fliplr=0.5,                    # 随机左右翻转的概率
        
        # ================== 实验记录与工程化管理 (直接能在论文里秀出来的图表) ==================
        project='Borehole_Training',   # 保存整个实验的主文件夹名字
        name='YOLOv8n_Seg_Run1',       # 本次实验的名称 (方便以后和不同的实验做对比)
        save=True,                     # 保存最优模型和最后一轮的模型权重
        plots=True,                    # 自动生成能直接贴进毕业论文的训练图表 (F1-curve, PR-curve, 混淆矩阵等)
        val=True,                      # 开启训练过程中的交叉验证评估
        amp=True                       # 使用自动混合精度 (AMP) 训练，节省一半显存并加速训练过程
    )
    
    print("\n[*] 🎉 训练任务圆满完成！")
    print("[*] 等训练结束后，你最好的模型权重会被保存在: Borehole_Training/YOLOv8n_Seg_Run1/weights/best.pt")
    print("[*] 你的所有曲线图、实验数据都被存放在 Borehole_Training/YOLOv8n_Seg_Run1 文件夹下，写论文直接去里面拿图！")

if __name__ == '__main__':
    train_model()