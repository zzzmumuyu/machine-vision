# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# OpenCV绘制中文
def draw_cn_text_on_cv2(img, text, pos, font_size=14, text_color=(0,255,0), bg_color=(0,0,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
    except:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x, y = pos
    draw.rectangle([x, y-text_h-2, x+text_w, y+2], fill=bg_color)
    draw.text((x, y-text_h), text, fill=text_color, font=font)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 读取图片
image_path = r"D:\jxsy2222.jpg"
img = cv2.imread(image_path)
if img is None:
    print("错误：未找到图片，请检查路径！")
else:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_copy = img.copy()
    height, width = img.shape[:2]
    print(f"图片尺寸：宽{width} × 高{height}")

    # 图像预处理
    blur = cv2.GaussianBlur(img, (5, 5), 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 60, 180)

    # 定义感兴趣区域（ROI）
    def region_of_interest(img):
        mask = np.zeros_like(img)
        vertices = np.array([
            [(0, height)],
            [(width, height)],
            [(width-50, height//2)],
            [(50, height//2)]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [vertices], 255)
        masked_img = cv2.bitwise_and(img, mask)
        return masked_img

    roi_img = region_of_interest(canny)

    # 霍夫变换检测直线
    lines = cv2.HoughLinesP(
        image=roi_img,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=30,
        maxLineGap=20
    )

    # 绘制车道线
    def draw_all_lanes(img, lines):
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if np.sqrt((x2-x1)**2 + (y2-y1)**2) > 20:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        img = draw_cn_text_on_cv2(img, "多车道虚线检测", (20, 50), font_size=20)
        return img

    result_img = draw_all_lanes(img_copy, lines)
    result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

    # 显示结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title("原始道路图", fontsize=12)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(result_img_rgb)
    plt.title("多车道虚线检测结果", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # 保存结果
    save_path = r"D:\多车道检测结果"
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, "原始图.jpg"), img)
    cv2.imwrite(os.path.join(save_path, "车道线检测结果.jpg"), result_img)
    print(f"结果已保存至：{save_path}")