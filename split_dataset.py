import os
import random
import shutil
from pathlib import Path

def split_dataset(base_path, val_ratio=0.2, test_ratio=0.1, seed=42):
    random.seed(seed)
    
    # 定义基础路径
    base_dir = Path(base_path)
    train_img_dir = base_dir / "train" / "images"
    train_lbl_dir = base_dir / "train" / "labels"
    
    # 验证原路径是否存在
    if not train_img_dir.exists() or not train_lbl_dir.exists():
        print(f"错误: 找不到 {train_img_dir} 或 {train_lbl_dir}")
        return

    # 定义并创建输出目标文件夹
    val_img_dir = base_dir / "valid" / "images"
    val_lbl_dir = base_dir / "valid" / "labels"
    test_img_dir = base_dir / "test" / "images"
    test_lbl_dir = base_dir / "test" / "labels"
    
    for d in [val_img_dir, val_lbl_dir, test_img_dir, test_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 获取所有的图片文件（支持常见的图片格式）
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_files = [f for f in train_img_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    # 提取并配对图片与对应的标注txt文件
    file_pairs = []
    for img_path in image_files:
        base_name = img_path.stem # 无后缀的文件名
        lbl_path = train_lbl_dir / f"{base_name}.txt"
        if lbl_path.exists():
            file_pairs.append((img_path, lbl_path))
        else:
            print(f"⚠️ 警告: 找不到图片 {img_path.name} 对应的 .txt 标签文件！已跳过该文件。")
            
    # 打乱顺序，准备分割
    random.shuffle(file_pairs)
    total = len(file_pairs)
    
    if total == 0:
        print("错误：没有找到任何可用的 图片-标签 匹配对。")
        return
        
    # 计算切分数量
    test_count = int(total * test_ratio)
    val_count = int(total * val_ratio)
    
    # 切片获取各部分组
    test_pairs = file_pairs[:test_count]
    val_pairs = file_pairs[test_count: test_count + val_count]
    
    # 移动文件的核心函数
    def move_files(pairs, dest_img_dir, dest_lbl_dir):
        for img_path, lbl_path in pairs:
            shutil.move(str(img_path), str(dest_img_dir / img_path.name))
            shutil.move(str(lbl_path), str(dest_lbl_dir / lbl_path.name))
            
    # 执行移动
    move_files(test_pairs, test_img_dir, test_lbl_dir)
    move_files(val_pairs, val_img_dir, val_lbl_dir)
    
    train_count = total - len(test_pairs) - len(val_pairs)
    
    print("================== 分割完成 ==================")
    print(f"总图片与标签对数: {total}")
    print(f"-> 训练集 (Train) 剩余: {train_count} 张 (约 {100 - (val_ratio+test_ratio)*100:.0f}%)")
    print(f"-> 验证集 (Valid) 移入: {len(val_pairs)} 张 (约 {val_ratio*100:.0f}%)")
    print(f"-> 测试集 (Test)  移入: {len(test_pairs)} 张 (约 {test_ratio*100:.0f}%)")
    print("==============================================")

    # 同步自动修改 data.yaml 配置文件
    yaml_path = base_dir / "data.yaml"
    if yaml_path.exists():
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 修正错误的覆盖路径
        content = content.replace("val: train/images", "val: valid/images")
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ 已自动修正 data.yaml 文件中的 val 验证集路径！")

if __name__ == "__main__":
    # 执行数据集分割，指向我的数据集目录
    dataset_directory = r"C:\Users\16288\Desktop\study\zhx\my_dataset"
    split_dataset(dataset_directory, val_ratio=0.2, test_ratio=0.1)
