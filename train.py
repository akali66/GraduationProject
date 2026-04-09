from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print(f"CUDA 可用状态: {torch.cuda.is_available()}")
    
    # 1. 加载官方的 YOLOv8 Nano 分割模型作为预训练基础
    model = YOLO('yolov8n-seg.pt')

    # 2. 开始训练！
    # 注意：这里的 data 路径必须指向你解压的数据集里的 data.yaml
    results = model.train(
        data='my_dataset/data.yaml',  # 我们假设你把解压的文件夹命名为 my_dataset
        epochs=50,                    # 测试跑通流程，先训练 50 轮
        imgsz=640,                    # 图像缩放尺寸
        batch=4,                      # 批次大小，数字越小越省显存
        device=0,                     # 使用 0 号 NVIDIA 显卡
        workers=0                     # Windows 系统下建议设为 0，防止多线程数据加载报错
    )
    
    print("训练测试完成！")
