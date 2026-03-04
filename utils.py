# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
from torchvision import transforms

def calculate_firing_rate(peak_count, target_peaks, base_rate=20):
    if peak_count <= 0:
        return base_rate
    ratio = target_peaks / peak_count
    adjusted_rate = base_rate * ratio
    return np.clip(adjusted_rate, 1.0, 100.0)

def plot_input_data(time_data, current_data, filename="input_data_plot.png", dpi=300):
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, current_data, 'b-', linewidth=1)
    plt.title("输入电流-时间数据")
    plt.xlabel("时间 (ms)")
    plt.ylabel("电流 (nA)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def fuzzy_process_image(img_tensor):
    """
    对图像进行模糊和简单二值化处理的示例函数：
    1. 使用高斯模糊对图像进行模糊处理。
    2. 使用中值为阈值进行简单二值化（>阈值则前景，否则背景）。

    img_tensor: [C,H,W]，假设为单通道灰度图 [1,28,28]

    返回处理后的图像张量 [1,H,W], 值为0或1的二值图
    """
    # 将张量转为PIL图像（记得先cpu()）
    img_pil = transforms.ToPILImage()(img_tensor.cpu())

    # 模糊处理
    img_blur = img_pil.filter(ImageFilter.GaussianBlur(radius=2))

    # 转numpy进行阈值化
    img_np = np.array(img_blur)
    threshold = np.median(img_np)
    img_bin = (img_np > threshold).astype(np.float32)

    # 转回Tensor
    img_out = torch.tensor(img_bin).unsqueeze(0)  # [1,H,W]
    return img_out
