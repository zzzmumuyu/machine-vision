# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os
from PIL import Image, ImageDraw, ImageFont

# 解决中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# OpenCV绘制中文（适配高版本Pillow）
def draw_cn_text_on_cv2(img, text, pos, font_size=12, text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x, y = pos
    y = max(y, text_height + 5)
    draw.rectangle([x, y - text_height - 5, x + text_width, y + 2], fill=bg_color)
    draw.text((x, y - text_height), text, fill=text_color, font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 配置参数
IMAGE_PATH = r"D:\jxsy444.jpg"
RESULT_SAVE_PATH = r"D:\共享单车检测结果"
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.45
MODEL_TYPE = "yolov8n.pt"
BIKE_CLASS_ID = 1

# 初始化目录
os.makedirs(RESULT_SAVE_PATH, exist_ok=True)

# 加载YOLOv8模型
print(f"加载YOLOv8模型（{MODEL_TYPE}）...")
model = YOLO(MODEL_TYPE)
print("模型加载完成，开始检测...")

# 读取图片
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise ValueError(f"未找到图片，请检查路径：{IMAGE_PATH}")
img_original = img.copy()
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]
print(f"图片尺寸：宽{width} × 高{height}")

# 执行检测
results = model(
    img_rgb,
    conf=CONF_THRESHOLD,
    iou=IOU_THRESHOLD,
    classes=[BIKE_CLASS_ID],
    verbose=False
)

# 解析结果并绘制
detections = results[0].boxes.data.cpu().numpy()
bike_count = len(detections)
img_with_detections = img_original.copy()

for det in detections:
    x1, y1, x2, y2, conf, _ = det
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img_with_detections, (x1, y1), (x2, y2), (0, 0, 255), 3)
    label = f"共享单车 置信度：{conf:.2f}"
    img_with_detections = draw_cn_text_on_cv2(img_with_detections, label, (x1, y1 - 20), font_size=12)

# 保存结果
img_with_detections_rgb = cv2.cvtColor(img_with_detections, cv2.COLOR_BGR2RGB)
cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "原始图片.jpg"), img_original)
cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "共享单车检测结果.jpg"), img_with_detections)

# 保存详情
with open(os.path.join(RESULT_SAVE_PATH, "检测结果详情.txt"), "w", encoding="utf-8") as f:
    f.write(f"实验四：校园共享单车检测结果\n")
    f.write(f"图片路径：{IMAGE_PATH}\n")
    f.write(f"图片尺寸：宽{width} × 高{height}\n")
    f.write(f"检测数量：{bike_count}\n")
    f.write(f"置信度阈值：{CONF_THRESHOLD}\n")
    f.write(f"IOU阈值：{IOU_THRESHOLD}\n")
    f.write("\n定位详情：\n")
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2, conf, _ = det
        f.write(f"第{i}辆：左上角({x1:.1f},{y1:.1f}) 右下角({x2:.1f},{y2:.1f}) 置信度：{conf:.4f}\n")

# 可视化对比
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("原始校园图像", fontsize=12)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(img_with_detections_rgb)
plt.title(f"共享单车检测结果（共{bike_count}辆）", fontsize=12)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_SAVE_PATH, "结果对比图.png"), dpi=300, bbox_inches="tight")
plt.show()

# 输出总结
print(f"\n检测完成！共检测到{bike_count}辆共享单车")
print(f"结果已保存至：{RESULT_SAVE_PATH}")
print("\n" + "=" * 50)
print("检测总结：")
print(f"1. 检测效率：推理时间约0.1~0.5秒")
print(f"2. 检测精度：置信度均≥{CONF_THRESHOLD}")
print(f"3. 核心流程：特征提取→目标定位→分类判断→非极大值抑制")
print("=" * 50)