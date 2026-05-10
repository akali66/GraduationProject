from pathlib import Path
import math

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"
OUT.mkdir(exist_ok=True)
IMG_DIR = ROOT / "my_dataset" / "test" / "images"
LABEL_DIR = ROOT / "my_dataset" / "test" / "labels"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
if not FONT_PATH.exists():
    FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size)


F_TITLE = font(28)
F_LABEL = font(22)
F_SMALL = font(18)


def fit_to_cell(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    tw, th = size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def draw_centered(draw: ImageDraw.ImageDraw, xy, text, font_obj, fill=(20, 20, 20)):
    x1, y1, x2, y2 = xy
    lines = text.split("\n")
    sizes = [draw.textbbox((0, 0), line, font=font_obj) for line in lines]
    heights = [b[3] - b[1] for b in sizes]
    total_h = sum(heights) + 6 * (len(lines) - 1)
    y = y1 + (y2 - y1 - total_h) / 2
    for line, bbox, h in zip(lines, sizes, heights):
        tw = bbox[2] - bbox[0]
        draw.text((x1 + (x2 - x1 - tw) / 2, y), line, font=font_obj, fill=fill)
        y += h + 6


def draw_box(draw, xy, text, fill="#EAF2F8", outline="#356A9A", font_obj=None):
    font_obj = font_obj or F_SMALL
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=2)
    draw_centered(draw, xy, text, font_obj)


def arrow(draw, p1, p2, fill="#333333"):
    draw.line([p1, p2], fill=fill, width=3)
    x1, y1 = p1
    x2, y2 = p2
    ang = math.atan2(y2 - y1, x2 - x1)
    for a in [ang + math.pi * 0.82, ang - math.pi * 0.82]:
        draw.line([p2, (x2 + 14 * math.cos(a), y2 + 14 * math.sin(a))], fill=fill, width=3)


def sample_data():
    imgs = sorted(IMG_DIR.glob("*.jpg"))
    if len(imgs) < 16:
        raise RuntimeError("测试集图片不足 16 张，无法生成 4×4 数据集图")
    sample = imgs[0]
    bgr = cv2.imread(str(sample))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imgs, sample, bgr, gray, blur, binary


def fig_dataset_grid(imgs):
    cell = (260, 195)
    label_h = 34
    margin = 28
    grid = Image.new("RGB", (margin * 2 + cell[0] * 4, margin * 2 + (cell[1] + label_h) * 4), "white")
    draw = ImageDraw.Draw(grid)
    letters = "abcdefghijklmnop"
    for idx, path in enumerate(imgs[:16]):
        im = Image.open(path)
        crop = fit_to_cell(im, cell)
        r, c = divmod(idx, 4)
        x = margin + c * cell[0]
        y = margin + r * (cell[1] + label_h)
        grid.paste(crop, (x, y))
        stem = path.name.split("_jpg")[0].replace("frame_", "")
        draw.text((x + 8, y + cell[1] + 5), f"({letters[idx]}) frame_{stem}", fill=(40, 40, 40), font=F_SMALL)
    grid.save(OUT / "fig5-1_dataset_grid_4x4.png")


def fig_preprocess(bgr, gray, blur, binary):
    items = [
        ("原图", cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)),
        ("灰度化", cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)),
        ("高斯平滑", cv2.cvtColor(blur, cv2.COLOR_GRAY2RGB)),
        ("Otsu二值化", cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)),
    ]
    cell = (260, 195)
    label_h = 42
    margin = 24
    fig = Image.new("RGB", (margin * 2 + cell[0] * 4, margin * 2 + cell[1] + label_h), "white")
    draw = ImageDraw.Draw(fig)
    for i, (name, arr) in enumerate(items):
        x, y = margin + i * cell[0], margin
        fig.paste(fit_to_cell(Image.fromarray(arr), cell), (x, y))
        tw = draw.textlength(name, font=F_LABEL)
        draw.text((x + cell[0] / 2 - tw / 2, y + cell[1] + 8), name, fill=(30, 30, 30), font=F_LABEL)
    fig.save(OUT / "fig2-2_preprocess_comparison.png")


def fig_contour_min_circle(bgr, binary):
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = bgr.copy()
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(contour_img, [cnt], -1, (0, 255, 0), 4)
        (x, y), r = cv2.minEnclosingCircle(cnt)
        cv2.circle(contour_img, (int(x), int(y)), int(r), (0, 0, 255), 4)
        cv2.circle(contour_img, (int(x), int(y)), 6, (255, 0, 0), -1)
    canvas = Image.new("RGB", (620, 460), "white")
    canvas.paste(fit_to_cell(Image.fromarray(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)), (560, 380)), (30, 30))
    draw = ImageDraw.Draw(canvas)
    draw.text((30, 420), "绿色为提取轮廓，红色为最小外接圆，蓝点为估计圆心", fill=(40, 40, 40), font=F_SMALL)
    canvas.save(OUT / "fig2-3_contour_min_enclosing.png")


def fig_preprocess_flow():
    can = Image.new("RGB", (1100, 260), "white")
    draw = ImageDraw.Draw(can)
    boxes = [
        (40, 75, 190, 155, "输入图像"),
        (245, 75, 395, 155, "灰度化"),
        (450, 75, 600, 155, "高斯平滑"),
        (655, 75, 805, 155, "二值化"),
        (860, 75, 1010, 155, "轮廓提取"),
    ]
    for box in boxes:
        draw_box(draw, box[:4], box[4])
    for i in range(len(boxes) - 1):
        arrow(draw, (boxes[i][2] + 10, 115), (boxes[i + 1][0] - 10, 115))
    draw.text((40, 205), "该流程对应传统方法，也对应 YOLO Mask 后处理中的二值化与轮廓提取步骤。", fill=(50, 50, 50), font=F_SMALL)
    can.save(OUT / "fig2-1_preprocess_flow.png")


def fig_hough_voting():
    can = Image.new("RGB", (900, 460), "white")
    draw = ImageDraw.Draw(can)
    cx, cy = 210, 210
    for pt in [(170, 150), (250, 155), (285, 225), (210, 285), (135, 225)]:
        draw.ellipse((pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5), fill="#D94801")
        draw.ellipse((pt[0] - 70, pt[1] - 70, pt[0] + 70, pt[1] + 70), outline="#9ECAE1", width=2)
    draw.ellipse((cx - 70, cy - 70, cx + 70, cy + 70), outline="#3182BD", width=4)
    draw.text((88, 330), "图像空间：边缘点对应多个可能圆心", fill=(40, 40, 40), font=F_SMALL)
    arrow(draw, (390, 210), (500, 210))
    for x in range(560, 820, 40):
        draw.line((x, 100, x, 320), fill="#E5E5E5")
    for y in range(100, 340, 40):
        draw.line((560, y, 820, y), fill="#E5E5E5")
    draw.ellipse((670, 200, 690, 220), fill="#2171B5")
    draw.text((585, 330), "参数空间：投票集中位置作为圆心候选", fill=(40, 40, 40), font=F_SMALL)
    can.save(OUT / "fig2-4_hough_circle_voting.png")


def fig_canny_flow():
    can = Image.new("RGB", (1120, 300), "white")
    draw = ImageDraw.Draw(can)
    boxes = [
        (30, 80, 170, 160, "输入灰度图"),
        (220, 80, 360, 160, "高斯滤波\n降噪"),
        (410, 80, 550, 160, "梯度计算"),
        (600, 80, 760, 160, "非极大值\n抑制"),
        (810, 80, 970, 160, "双阈值\n连接"),
        (1010, 80, 1090, 160, "边缘图"),
    ]
    for box in boxes:
        draw_box(draw, box[:4], box[4])
    for i in range(len(boxes) - 1):
        arrow(draw, (boxes[i][2] + 10, 120), (boxes[i + 1][0] - 10, 120))
    draw.text((30, 220), "方法三显式调用 Canny 生成边缘图，再将边缘图送入霍夫圆检测。", fill=(50, 50, 50), font=F_SMALL)
    can.save(OUT / "fig2-5_canny_flow.png")


def fig_bbox_vs_mask(sample, bgr, gray):
    h, w = gray.shape
    label = LABEL_DIR / (sample.stem + ".txt")
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    left, right = img_rgb.copy(), img_rgb.copy()
    if label.exists():
        vals = list(map(float, label.read_text().split()))
        coords = np.array(vals[1:]).reshape(-1, 2)
        poly = np.column_stack([coords[:, 0] * w, coords[:, 1] * h]).astype(np.int32)
        x, y, ww, hh = cv2.boundingRect(poly)
        cv2.rectangle(left, (x, y), (x + ww, y + hh), (255, 0, 0), 5)
        cv2.circle(left, (x + ww // 2, y + hh // 2), 8, (255, 0, 0), -1)
        overlay = right.copy()
        cv2.fillPoly(overlay, [poly], (0, 180, 0))
        right = cv2.addWeighted(overlay, 0.35, right, 0.65, 0)
        cv2.polylines(right, [poly], True, (0, 150, 0), 5)
    fig = Image.new("RGB", (720, 360), "white")
    draw = ImageDraw.Draw(fig)
    fig.paste(fit_to_cell(Image.fromarray(left), (340, 255)), (20, 35))
    fig.paste(fit_to_cell(Image.fromarray(right), (340, 255)), (360, 35))
    draw.text((100, 300), "检测框中心", fill=(50, 50, 50), font=F_LABEL)
    draw.text((455, 300), "实例分割 Mask", fill=(50, 50, 50), font=F_LABEL)
    fig.save(OUT / "fig2-6_bbox_vs_mask.png")


def fig_metrics():
    can = Image.new("RGB", (950, 440), "white")
    draw = ImageDraw.Draw(can)
    draw.ellipse((110, 100, 310, 300), outline="#3182BD", width=4)
    draw.ellipse((140, 115, 320, 295), outline="#E6550D", width=4)
    draw.ellipse((205, 195, 215, 205), fill="#3182BD")
    draw.ellipse((225, 205, 235, 215), fill="#E6550D")
    draw.line((210, 200, 230, 210), fill="#222", width=3)
    draw.text((90, 325), "中心误差：预测圆心与标注圆心距离", fill=(40, 40, 40), font=F_SMALL)
    draw.text((105, 360), "半径误差：预测半径与标注半径差值", fill=(40, 40, 40), font=F_SMALL)
    draw.ellipse((520, 120, 700, 300), fill="#9ECAE1", outline="#3182BD", width=3)
    draw.ellipse((620, 120, 800, 300), fill="#FDD0A2", outline="#E6550D", width=3)
    draw.text((588, 205), "交集", fill=(30, 30, 30), font=F_LABEL)
    draw.text((515, 325), "IoU = 交集面积 / 并集面积", fill=(40, 40, 40), font=F_SMALL)
    can.save(OUT / "fig2-7_metrics_diagram.png")


def flow_chart(filename, title, steps, width=1200):
    can = Image.new("RGB", (width, 260), "white")
    draw = ImageDraw.Draw(can)
    draw.text((30, 20), title, fill=(20, 20, 20), font=F_LABEL)
    n, bw = len(steps), 150
    gap = (width - 60 - n * bw) / (n - 1)
    y1, y2 = 95, 170
    boxes = []
    for i, step in enumerate(steps):
        x1, x2 = 30 + i * (bw + gap), 30 + i * (bw + gap) + bw
        boxes.append((x1, x2))
        draw_box(draw, (x1, y1, x2, y2), step, fill="#F7FBFF")
        if i > 0:
            arrow(draw, (boxes[i - 1][1] + 8, 132), (x1 - 8, 132))
    can.save(OUT / filename)


def fig_software():
    flow_chart("fig4-2_frontend_interaction_flow.png", "前端交互模块流程", ["选择/拖拽\n图片", "选择算法\n或对比模式", "填写参数\n面板", "封装\nFormData", "发送\n/api/detect", "接收JSON\n结果", "渲染/导出"])
    flow_chart("fig4-3_backend_api_flow.png", "后端接口模块流程", ["接收请求", "读取图片\n字节流", "OpenCV\n解码", "灰度化\nRGB转换", "解析参数\n方法字典", "运行检测\n计算指标", "JSON/Base64\n返回"])
    flow_chart("fig4-4_metric_flow.png", "指标计算模块流程", ["读取检测\n结果", "判断方法\n类型", "覆盖率/\n置信度", "Mask IoU", "返回指标"])
    flow_chart("fig4-5_export_flow.png", "结果展示与导出流程", ["Base64\n结果图", "前端渲染", "展示圆心\n半径指标", "CSV导出", "PNG导出"])

    can = Image.new("RGB", (1200, 620), "white")
    draw = ImageDraw.Draw(can)
    draw.text((35, 25), "系统总体框架与数据流", fill=(20, 20, 20), font=F_TITLE)
    draw_box(draw, (60, 110, 250, 210), "前端界面\n上传/参数/对比", fill="#EAF2F8")
    draw_box(draw, (330, 110, 520, 210), "FastAPI后端\n/api/detect", fill="#EAF2F8")
    draw_box(draw, (600, 110, 790, 210), "图像解码\n灰度化/转换", fill="#EAF2F8")
    draw_box(draw, (870, 110, 1060, 210), "结果返回\nBase64/JSON", fill="#EAF2F8")
    for p1, p2 in [((250, 160), (330, 160)), ((520, 160), (600, 160)), ((790, 160), (870, 160))]:
        arrow(draw, p1, p2)
    draw_box(draw, (435, 300, 765, 390), "算法调度模块\n单方法模式 / 四方法对比模式", fill="#F7FCF5", outline="#31A354")
    for x, y, text in [
        (80, 470, "方法一\n霍夫圆"),
        (310, 470, "方法二\n轮廓外接圆"),
        (540, 470, "方法三\nCanny+霍夫"),
        (770, 470, "方法四\nYOLOv8分割"),
        (1000, 470, "指标计算\n可视化/导出"),
    ]:
        draw_box(draw, (x, y, x + 150, y + 80), text, fill="#FFF7EC", outline="#E6550D")
    arrow(draw, (695, 210), (600, 300))
    for target in [(155, 470), (385, 470), (615, 470), (845, 470)]:
        arrow(draw, (600, 390), target)
    arrow(draw, (920, 510), (1000, 510))
    arrow(draw, (1075, 470), (970, 210))
    can.save(OUT / "fig4-1_system_framework.png")

    can = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(can)
    draw.text((35, 25), "软件界面区域划分示意", fill=(20, 20, 20), font=F_TITLE)
    draw.rounded_rectangle((40, 80, 1060, 140), radius=8, fill="#DDEAF6", outline="#4C78A8", width=2)
    draw.text((60, 100), "顶部操作区：图片上传 / 算法选择 / 对比模式 / 开始检测 / 导出", fill=(30, 30, 30), font=F_SMALL)
    draw.rounded_rectangle((40, 170, 320, 590), radius=8, fill="#F7F7F7", outline="#999999", width=2)
    draw.text((70, 190), "左侧参数区", fill=(30, 30, 30), font=F_LABEL)
    draw.text((70, 235), "根据不同算法\n动态显示参数滑块\n和选择项", fill=(60, 60, 60), font=F_SMALL)
    draw.rounded_rectangle((350, 170, 1060, 590), radius=8, fill="#FFFFFF", outline="#999999", width=2)
    draw.text((380, 190), "结果展示区", fill=(30, 30, 30), font=F_LABEL)
    for i, (x, y) in enumerate([(380, 250), (710, 250), (380, 430), (710, 430)]):
        draw.rounded_rectangle((x, y, x + 290, y + 130), radius=8, fill="#F2F2F2", outline="#BBBBBB")
        draw.text((x + 20, y + 20), f"方法{i + 1}结果图 / 指标 / 诊断信息", fill=(70, 70, 70), font=F_SMALL)
    can.save(OUT / "fig4-6_ui_layout.png")


def fig_training_and_bars():
    font_manager.fontManager.addfont(str(FONT_PATH))
    font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False
    df = pd.read_csv(ROOT / "runs" / "segment" / "Borehole_Training" / "YOLOv8n_Seg_Run1" / "results.csv")
    df.columns = [c.strip() for c in df.columns]

    def plot_group(cols, title, ylabel, filename):
        fig, ax = plt.subplots(figsize=(8, 5), dpi=180)
        for col in cols:
            ax.plot(df["epoch"], df[col], label=col, linewidth=1.8)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / filename)
        plt.close(fig)

    plot_group(["train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss"], "训练损失变化曲线", "Loss", "fig5-3a_train_loss.png")
    plot_group(["val/box_loss", "val/seg_loss", "val/cls_loss", "val/dfl_loss"], "验证损失变化曲线", "Loss", "fig5-3b_val_loss.png")
    plot_group(["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"], "检测指标变化曲线", "Metric", "fig5-3c_box_metrics.png")
    plot_group(["metrics/precision(M)", "metrics/recall(M)", "metrics/mAP50(M)", "metrics/mAP50-95(M)"], "分割指标变化曲线", "Metric", "fig5-3d_mask_metrics.png")

    summary = pd.read_csv(ROOT / "runs" / "eval" / "batch_eval_summary.csv")
    labels = summary["method_label"].tolist()
    center = summary["mean_center_error_px"].tolist()
    iou = summary["mean_circle_iou_gt"].tolist()
    colors = ["#9ecae1", "#9ecae1", "#9ecae1", "#fb6a4a"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=180)
    ax1.bar(labels, center, color=colors)
    ax1.set_title("平均中心误差对比")
    ax1.set_ylabel("中心误差 / px")
    ax1.tick_params(axis="x", rotation=25)
    for i, v in enumerate(center):
        ax1.text(i, v + max(center) * 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax2.bar(labels, iou, color=colors)
    ax2.set_title("预测圆-标注掩码 IoU 对比")
    ax2.set_ylabel("IoU")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="x", rotation=25)
    for i, v in enumerate(iou):
        ax2.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-4_center_error_iou_bar.png")
    plt.close(fig)


def main():
    imgs, sample, bgr, gray, blur, binary = sample_data()
    fig_dataset_grid(imgs)
    fig_preprocess_flow()
    fig_preprocess(bgr, gray, blur, binary)
    fig_contour_min_circle(bgr, binary)
    fig_hough_voting()
    fig_canny_flow()
    fig_bbox_vs_mask(sample, bgr, gray)
    fig_metrics()
    fig_software()
    fig_training_and_bars()
    print(f"regenerated figures in {OUT}")


if __name__ == "__main__":
    main()
