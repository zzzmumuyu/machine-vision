import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import cv2

# 单通道图像卷积（手动实现）
def convolve_single_channel(channel, kernel, padding=1):
    kernel_size = kernel.shape[0]
    padded_channel = np.pad(channel, padding, mode='constant')
    h, w = channel.shape
    output = np.zeros_like(channel, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded_channel[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.sum(region * kernel)
    return np.clip(output, 0, 255).astype(np.uint8)

# RGB图像卷积（分通道处理）
def convolve_rgb_image(image, kernel):
    r, g, b = cv2.split(image)
    r_conv = convolve_single_channel(r, kernel)
    g_conv = convolve_single_channel(g, kernel)
    b_conv = convolve_single_channel(b, kernel)
    return cv2.merge([r_conv, g_conv, b_conv])

# 定义卷积核
sobel_x_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
sobel_y_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
given_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)

# 读取并预处理图像
img_path = r"D:\jxsy11.jpg"
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise ValueError("图像读取失败！请检查路径或文件")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# 卷积滤波
sobel_x_rgb = convolve_rgb_image(img_bgr, sobel_x_kernel)
sobel_y_rgb = convolve_rgb_image(img_bgr, sobel_y_kernel)

# 转为单通道计算梯度幅值
sobel_x_gray = cv2.cvtColor(sobel_x_rgb, cv2.COLOR_BGR2GRAY)
sobel_y_gray = cv2.cvtColor(sobel_y_rgb, cv2.COLOR_BGR2GRAY)
sobel_mag = np.sqrt(np.square(sobel_x_gray.astype(np.float32)) + np.square(sobel_y_gray.astype(np.float32)))
sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)
sobel_mag_rgb = cv2.cvtColor(sobel_mag, cv2.COLOR_GRAY2RGB)
given_kernel_rgb = convolve_rgb_image(img_bgr, given_kernel)

# 手动计算RGB颜色直方图
def compute_color_histogram(image):
    r_hist = np.zeros(256, dtype=int)
    g_hist = np.zeros(256, dtype=int)
    b_hist = np.zeros(256, dtype=int)
    h, w, _ = image.shape
    for i in range(h):
        for j in range(w):
            r = image[i, j, 0]
            g = image[i, j, 1]
            b = image[i, j, 2]
            r_hist[r] += 1
            g_hist[g] += 1
            b_hist[b] += 1
    return r_hist, g_hist, b_hist

r_hist, g_hist, b_hist = compute_color_histogram(img_rgb)

# 计算灰度共生矩阵（GLCM）
def compute_glcm(gray_img, distance=1, angle=0):
    glcm = np.zeros((256, 256), dtype=int)
    h, w = gray_img.shape
    for i in range(h):
        for j in range(w - distance):
            pixel1 = gray_img[i, j]
            pixel2 = gray_img[i, j + distance]
            glcm[pixel1, pixel2] += 1
    return glcm / np.sum(glcm)

# 提取GLCM纹理特征
def extract_glcm_features(glcm):
    contrast, energy, entropy, correlation = 0, 0, 0, 0
    mean_i = np.sum(np.arange(256) * np.sum(glcm, axis=1))
    mean_j = np.sum(np.arange(256) * np.sum(glcm, axis=0))
    var_i = np.sum(((np.arange(256) - mean_i) ** 2) * np.sum(glcm, axis=1))
    var_j = np.sum(((np.arange(256) - mean_j) ** 2) * np.sum(glcm, axis=0))
    for i in range(256):
        for j in range(256):
            p = glcm[i, j]
            if p == 0:
                continue
            contrast += (i - j) ** 2 * p
            energy += p ** 2
            entropy -= p * np.log2(p)
            correlation += (i - mean_i) * (j - mean_j) * p / np.sqrt(var_i * var_j) if (var_i * var_j) != 0 else 0
    return {"contrast": contrast, "energy": energy, "entropy": entropy, "correlation": correlation}

glcm = compute_glcm(img_gray)
texture_features = extract_glcm_features(glcm)
np.save("texture_features.npy", texture_features)

# 结果可视化
plt.figure(figsize=(16, 12))
plt.subplot(2, 3, 1)
plt.imshow(img_rgb)
plt.title("原始图像")
plt.axis("off")
plt.subplot(2, 3, 2)
plt.imshow(cv2.cvtColor(sobel_x_rgb, cv2.COLOR_BGR2RGB))
plt.title("Sobel X方向滤波")
plt.axis("off")
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(sobel_y_rgb, cv2.COLOR_BGR2RGB))
plt.title("Sobel Y方向滤波")
plt.axis("off")
plt.subplot(2, 3, 4)
plt.imshow(sobel_mag_rgb, cmap="gray")
plt.title("Sobel梯度幅值")
plt.axis("off")
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(given_kernel_rgb, cv2.COLOR_BGR2RGB))
plt.title("给定卷积核滤波")
plt.axis("off")
plt.subplot(2, 3, 6)
plt.plot(r_hist, color="red", label="Red")
plt.plot(g_hist, color="green", label="Green")
plt.plot(b_hist, color="blue", label="Blue")
plt.title("RGB颜色直方图")
plt.xlabel("像素值")
plt.ylabel("像素数量")
plt.legend()
plt.tight_layout()
plt.show()

# 保存结果
cv2.imwrite("sobel_x_filtered.jpg", sobel_x_rgb)
cv2.imwrite("sobel_y_filtered.jpg", sobel_y_rgb)
cv2.imwrite("sobel_mag_filtered.jpg", sobel_mag)
cv2.imwrite("given_kernel_filtered.jpg", given_kernel_rgb)

# 输出分析结果
print("=== 实验一：图像滤波与特征提取 结果分析 ===")
print("1. 卷积滤波修复说明：")
print("   - 原错误：3通道Sobel结果导致转换失败")
print("   - 修复：先转单通道再计算梯度幅值")
print("\n2. 纹理特征结果（GLCM）：")
for key, val in texture_features.items():
    print(f"   - {key}：{val:.4f}")
print("\n3. 结果已保存至当前目录")