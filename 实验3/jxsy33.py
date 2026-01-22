# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 配置参数
IMAGE_PATH = r"D:\jxsy333.jpg"
MODEL_SAVE_PATH = r"D:\学号识别模型"
RESULT_SAVE_PATH = r"D:\学号识别结果"
IMG_SIZE = 28
ADAPTIVE_BLOCK_SIZE = 15
ADAPTIVE_C = 2
DILATE_KERNEL = (3, 3)
ERODE_KERNEL = (2, 2)

# 准备MNIST数据
def prepare_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape((x_train.shape[0], IMG_SIZE, IMG_SIZE, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((x_test.shape[0], IMG_SIZE, IMG_SIZE, 1)).astype("float32") / 255.0
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# 构建CNN模型
def build_cnn_model():
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    model_path = os.path.join(MODEL_SAVE_PATH, "mnist_cnn_model.h5")
    if os.path.exists(model_path):
        print("加载已训练的CNN模型...")
        model = load_model(model_path)
        return model
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    (x_train, y_train), (x_test, y_test) = prepare_mnist_data()
    print("开始训练CNN模型...")
    history = model.fit(
        x_train, y_train, epochs=6, batch_size=64, validation_data=(x_test, y_test), verbose=1
    )
    # 保存训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="训练准确率")
    plt.plot(history.history["val_accuracy"], label="测试准确率")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="训练损失")
    plt.plot(history.history["val_loss"], label="测试损失")
    plt.legend()
    plt.savefig(os.path.join(RESULT_SAVE_PATH, "模型训练曲线.png"))
    model.save(model_path)
    print(f"模型保存至：{model_path}")
    return model

# 图片预处理（含倾斜矫正）
def preprocess_student_id_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"未找到图片，请检查路径：{img_path}")
    img_original = img.copy()

    # 倾斜矫正
    def rotate_image(img):
        edges = cv2.Canny(img, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
        if lines is None:
            return img
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            if abs(angle) < 10:
                angles.append(angle)
        if not angles:
            return img
        avg_angle = np.mean(angles)
        h, w = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    img_rotated = rotate_image(img)
    # 自适应二值化
    img_binary = cv2.adaptiveThreshold(
        img_rotated, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK_SIZE, C=ADAPTIVE_C
    )
    # 形态学操作
    kernel_dilate = np.ones(DILATE_KERNEL, np.uint8)
    img_dilate = cv2.dilate(img_binary, kernel_dilate, iterations=1)
    kernel_erode = np.ones(ERODE_KERNEL, np.uint8)
    img_morph = cv2.erode(img_dilate, kernel_erode, iterations=1)
    # 轮廓检测
    contours, _ = cv2.findContours(img_morph.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digit_contours = []
    h, w = img.shape
    for cnt in contours:
        x, y, w_cnt, h_cnt = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if (5 < w_cnt < w // 2) and (10 < h_cnt < h // 2) and (area > 20):
            aspect_ratio = w_cnt / h_cnt
            if 0.2 < aspect_ratio < 1.5:
                digit_contours.append((x, y, w_cnt, h_cnt))
    # 排序并检查
    digit_contours.sort(key=lambda x: x[0])
    if len(digit_contours) == 0:
        os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
        cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "矫正后图片.png"), img_rotated)
        cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "二值化图片.png"), img_binary)
        cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "形态学处理后.png"), img_morph)
        raise ValueError("未检测到数字！请调整参数或重新拍摄")
    # 分割数字
    digit_imgs = []
    img_with_contours = cv2.cvtColor(img_rotated, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w_cnt, h_cnt) in enumerate(digit_contours):
        cv2.rectangle(img_with_contours, (x, y), (x + w_cnt, y + h_cnt), (0, 0, 255), 2)
        cv2.putText(img_with_contours, f"{i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # 裁剪与预处理
        pad = 5
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_end = min(w, x + w_cnt + pad)
        y_end = min(h, y + h_cnt + pad)
        digit_roi = img_morph[y_start:y_end, x_start:x_end]
        # 填充为正方形
        h_roi, w_roi = digit_roi.shape
        max_side = max(h_roi, w_roi)
        pad_h = (max_side - h_roi) // 2
        pad_w = (max_side - w_roi) // 2
        digit_padded = cv2.copyMakeBorder(digit_roi, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0)
        # 缩放与归一化
        digit_resized = cv2.resize(digit_padded, (IMG_SIZE, IMG_SIZE))
        digit_normalized = digit_resized / 255.0
        digit_imgs.append(digit_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1))
    # 保存中间结果
    os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
    cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "原始学号图.png"), img_original)
    cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "矫正后图片.png"), img_rotated)
    cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "二值化图片.png"), img_binary)
    cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "形态学处理后.png"), img_morph)
    cv2.imwrite(os.path.join(RESULT_SAVE_PATH, "数字分割图.png"), img_with_contours)
    print(f"预处理完成，检测到{len(digit_imgs)}个数字！")
    return digit_imgs, img_with_contours

# 学号识别
def recognize_student_id():
    model = build_cnn_model()
    digit_imgs, img_with_contours = preprocess_student_id_img(IMAGE_PATH)
    # 预测
    student_id = ""
    pred_details = []
    for i, digit_img in enumerate(digit_imgs):
        pred_prob = model.predict(digit_img, verbose=0)
        pred_digit = str(np.argmax(pred_prob))
        pred_confidence = pred_prob[0][np.argmax(pred_prob)]
        student_id += pred_digit
        pred_details.append((i + 1, pred_digit, pred_confidence))
    # 可视化结果
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.title("数字分割结果", fontsize=12)
    plt.axis("off")
    plt.subplot(1, 3, 2)
    details_text = "数字预测详情：\n"
    for idx, digit, conf in pred_details:
        details_text += f"第{idx}个：{digit}（置信度：{conf:.3f}）\n"
    plt.text(0.1, 0.9, details_text, fontsize=10, verticalalignment="top")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.5, f"识别到的学号：\n{student_id}", fontsize=22, ha="center", va="center",
             transform=plt.gca().transAxes)
    plt.title("最终识别结果", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_SAVE_PATH, "最终识别结果.png"))
    plt.show()
    # 输出结果
    print("\n" + "=" * 30)
    print(f"最终识别学号：{student_id}")
    print("=" * 30)
    for idx, digit, conf in pred_details:
        print(f"第{idx}个数字：预测为{digit}，置信度：{conf:.4f}")
    return student_id

# 主函数
if __name__ == "__main__":
    try:
        os.makedirs(RESULT_SAVE_PATH, exist_ok=True)
        recognize_student_id()
        print(f"\n所有结果已保存至：{RESULT_SAVE_PATH}")
    except Exception as e:
        print(f"\n运行出错：{str(e)}")