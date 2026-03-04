import os
import numpy as np
from PIL import Image
import shutil
import random

def generate_fingerprint_dataset(base_dir, num_samples_per_class=1000):
    """
    生成8类指纹方向数据集 (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5 度)
    图像为 28x28 灰度图，模拟 MNIST 格式
    """
    angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    classes = [str(a) for a in angles]
    
    # Clean up existing directory
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    # Create train and test directories
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(train_dir)
    os.makedirs(test_dir)
    
    print(f"Generating fingerprint dataset in {base_dir}...")
    
    for idx, angle in enumerate(angles):
        # Create class directories in train and test
        train_class_dir = os.path.join(train_dir, classes[idx])
        test_class_dir = os.path.join(test_dir, classes[idx])
        os.makedirs(train_class_dir)
        os.makedirs(test_class_dir)
        
        # 将角度转换为弧度
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        
        for i in range(num_samples_per_class):
            # ... (generation logic same as before) ...
            # 创建 28x28 网格
            x = np.linspace(-1, 1, 28)
            y = np.linspace(-1, 1, 28)
            X, Y = np.meshgrid(x, y)
            
            # 旋转坐标系
            X_rot = X * c + Y * s
            Y_rot = -X * s + Y * c
            
            # 生成正弦波光栅 (模拟指纹脊线)
            freq = np.random.uniform(4.5, 5.5) 
            # 减小相位随机性，使特征在空间上更固定，便于简单SNN学习
            phase = np.random.uniform(-0.5, 0.5)
            
            grating = np.sin(2 * np.pi * freq * X_rot + phase)
            
            noise = np.random.normal(0, 0.2, grating.shape)
            img_data = grating + noise
            
            img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min())
            img_data = (img_data * 255).astype(np.uint8)
            
            img_data = np.where(img_data > 127, 
                              np.minimum(img_data + 50, 255), 
                              np.maximum(img_data - 50, 0)).astype(np.uint8)
            
            # Save to train (80%) or test (20%)
            img = Image.fromarray(img_data)
            if i < int(num_samples_per_class * 0.8):
                img.save(os.path.join(train_class_dir, f"sample_{i:04d}.png"))
            else:
                img.save(os.path.join(test_class_dir, f"sample_{i:04d}.png"))
            
        print(f"  Class {angle}°: Generated {num_samples_per_class} images (800 train, 200 test)")

if __name__ == "__main__":
    dataset_dir = os.path.join(os.getcwd(), "fingerprint_orientation")
    generate_fingerprint_dataset(dataset_dir)
