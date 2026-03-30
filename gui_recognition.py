import tkinter as tk
from tkinter import ttk, filedialog, StringVar, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageTk
from torchvision import transforms
import os
from data_processing import SynapticDataProcessor
from utils import fuzzy_process_image
import scipy

from tkinter import ttk, filedialog, StringVar, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


from datetime import datetime
from torchvision.datasets import ImageFolder
import os
import json
import torch
import torch.nn.functional as F
import pandas as pd



def calculate_adaptive_threshold(self, image):
    """计算自适应阈值"""
    local_mean = torch.mean(image)
    local_std = torch.std(image)
    return local_mean - 0.2 * local_std


def bilateral_filter(self, x, kernel_size, sigma_space, sigma_intensity):
    """改进的双边滤波实现"""
    height, width = x.shape
    pad = kernel_size // 2
    padded = torch.zeros((height + 2 * pad, width + 2 * pad), device=x.device)
    padded[pad:-pad, pad:-pad] = x
    result = torch.zeros_like(x)

    # 创建空间权重核
    space_kernel = torch.zeros((kernel_size, kernel_size), device=x.device)
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = ((i - center) ** 2 + (j - center) ** 2)
            space_kernel[i, j] = torch.exp(-distance / (2 * sigma_space ** 2))

    # 应用滤波器
    for i in range(height):
        for j in range(width):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            intensity_diff = (window - x[i, j]) ** 2
            intensity_weights = torch.exp(-intensity_diff / (2 * sigma_intensity ** 2))
            weights = space_kernel * intensity_weights
            weights = weights / weights.sum()
            result[i, j] = (window * weights).sum()

    return result

def apply_heatmap_style(img_tensor, style='fashion'):
        """应用热力图风格的颜色映射
        Args:
            img_tensor: 输入图像张量
            style: 'fashion' 或 'mnist'
        Returns:
            colored_tensor: 添加热力图风格后的图像张量
        """
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.squeeze(0)

        # 确保值域在[0,1]范围内
        if img_tensor.min() < 0:
            img_tensor = (img_tensor + 1) / 2

        # 创建RGB通道
        colored = torch.zeros(3, img_tensor.shape[0], img_tensor.shape[1])

        if style == 'fashion':
            # Fashion-MNIST使用绿黄-紫色配色方案
            # 绿色通道 (高值区域偏黄绿)
            colored[1] = torch.where(img_tensor > 0.5,
                                     2 * (img_tensor - 0.5),
                                     img_tensor)

            # 红色通道 (高值区域偏黄)
            colored[0] = torch.where(img_tensor > 0.7,
                                     3 * (img_tensor - 0.7),
                                     0)

            # 蓝色通道 (低值区域偏紫)
            colored[2] = torch.where(img_tensor < 0.3,
                                     1 - img_tensor,
                                     0)
        else:  # MNIST
            # MNIST使用黄-紫色配色方案
            # 紫色背景，黄色前景

            # 红色通道 (高值区域偏黄)
            colored[0] = img_tensor

            # 绿色通道 (高值区域偏黄)
            colored[1] = img_tensor

            # 蓝色通道 (低值区域偏紫)
            colored[2] = torch.where(img_tensor < 0.5,
                                     0.8 * (1 - img_tensor),
                                     0)

        # 增强对比度
        colored = torch.clamp((colored - colored.mean()) * 1.5 + colored.mean(), 0, 1)

        return colored

class RecognitionDisplay(ttk.Frame):
    def __init__(self, parent, dataset_manager=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.results_buffer = []
        self.weight_matrices = []
        self.confusion_matrix = np.zeros((10, 10))
        self.test_datasets = {}
        self.ltp_processor = SynapticDataProcessor()
        self.ltp_data_loaded = False

        self.create_result_text()
        self.load_default_datasets()
        self.setup_ui()

    # 在 gui_recognition.py 中的 RecognitionDisplay 类中添加新方法

    def setup_custom_dataset(self):
        """设置自定义数据集支持"""
        dataset_frame = ttk.LabelFrame(self, text="数据集设置", padding=10)
        dataset_frame.pack(fill='x', padx=5, pady=5)

        # 数据集选择
        selection_frame = ttk.Frame(dataset_frame)
        selection_frame.pack(fill='x', pady=5)

        self.dataset_var = StringVar(value='mnist')
        ttk.Label(selection_frame, text="数据集选择:").pack(side='left', padx=5)
        self.dataset_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.dataset_var,
            values=['mnist', 'fmnist', 'custom'],
            state='readonly'
        )
        self.dataset_combo.pack(side='left', padx=5)
        self.dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_change)

        # 自定义数据集导入
        custom_frame = ttk.Frame(dataset_frame)
        custom_frame.pack(fill='x', pady=5)

        self.custom_path_var = StringVar()
        ttk.Label(custom_frame, text="自定义数据集路径:").pack(side='left', padx=5)
        ttk.Entry(custom_frame, textvariable=self.custom_path_var).pack(side='left', fill='x', expand=True, padx=5)
        ttk.Button(custom_frame, text="浏览", command=self.browse_custom_dataset).pack(side='left')

    def browse_custom_dataset(self):
        """浏览并加载自定义数据集"""
        dir_path = filedialog.askdirectory(title="选择自定义数据集目录")
        if not dir_path:
            return

        try:
            # 验证数据集结构
            is_valid, message = self.validate_custom_dataset(dir_path)
            if not is_valid:
                messagebox.showerror("错误", f"无效的数据集结构: {message}")
                return

            self.custom_path_var.set(dir_path)
            self.load_custom_dataset(dir_path)
            messagebox.showinfo("成功", "自定义数据集加载成功")
        except Exception as e:
            messagebox.showerror("错误", f"加载数据集失败: {str(e)}")

    def validate_custom_dataset(self, dir_path):
        """验证自定义数据集的结构"""
        try:
            # 检查目录结构
            if not os.path.isdir(dir_path):
                return False, "不是有效的目录"

            # 检查是否包含必要的子目录
            required_dirs = ['train', 'test']
            for subdir in required_dirs:
                if not os.path.isdir(os.path.join(dir_path, subdir)):
                    return False, f"缺少{subdir}目录"

            # 检查图像文件
            valid_extensions = {'.jpg', '.jpeg', '.png'}
            for subdir in required_dirs:
                path = os.path.join(dir_path, subdir)
                files = os.listdir(path)
                image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
                if not image_files:
                    return False, f"{subdir}目录中没有有效的图像文件"

            return True, "数据集结构有效"
        except Exception as e:
            return False, str(e)

    def load_custom_dataset(self, dir_path):
        """加载自定义数据集"""
        try:
            # 设置数据转换
            transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            # 加载训练集和测试集
            train_dataset = ImageFolder(
                os.path.join(dir_path, 'train'),
                transform=transform
            )
            test_dataset = ImageFolder(
                os.path.join(dir_path, 'test'),
                transform=transform
            )

            # 保存数据集信息
            self.custom_dataset = {
                'train': train_dataset,
                'test': test_dataset,
                'classes': train_dataset.classes
            }

            # 更新界面显示
            self.update_dataset_info()

        except Exception as e:
            raise Exception(f"加载自定义数据集失败: {str(e)}")

    def update_dataset_info(self):
        """更新数据集信息显示"""
        if hasattr(self, 'custom_dataset'):
            info_text = (
                f"数据集信息:\n"
                f"类别数: {len(self.custom_dataset['classes'])}\n"
                f"训练集样本数: {len(self.custom_dataset['train'])}\n"
                f"测试集样本数: {len(self.custom_dataset['test'])}\n"
                f"类别: {', '.join(self.custom_dataset['classes'])}"
            )
            self.update_result_text(info_text)

    def on_dataset_change(self, event=None):
        """数据集选择变更处理"""
        selected = self.dataset_var.get()
        if selected == 'custom':
            self.browse_custom_dataset()
        else:
            # 使用预设数据集
            if hasattr(self, 'custom_dataset'):
                del self.custom_dataset
            self.update_result_text(f"使用{selected}数据集")

    def process_dataset(self, dataset, noise_level):
        """处理数据集图像"""
        processed_samples = []
        for data, label in dataset:
            # 添加噪声
            noisy_data = self.add_noise(data, noise_level)
            # 降噪处理
            denoised_data = self.denoise_with_ltp(noisy_data)
            processed_samples.append({
                'original': data,
                'noisy': noisy_data,
                'denoised': denoised_data,
                'label': label
            })
        return processed_samples

    def add_noise(self, image, noise_level, seed=None):
        """添加噪声到图像
        
        Args:
            image: 输入图像张量
            noise_level: 噪声强度
            seed: 随机种子，如果为None则使用随机种子
            
        Returns:
            添加噪声后的图像张量
        """
        if seed is not None:
            # 保存当前随机状态
            numpy_state = np.random.get_state()
            torch_state = torch.random.get_rng_state()
            
            # 设置固定种子
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # 生成噪声
        noise = torch.randn_like(image) * noise_level
        noisy_image = image + noise
        
        if seed is not None:
            # 恢复之前的随机状态
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            
        return torch.clamp(noisy_image, -1, 1)

    def show_confusion_matrix(self):
        """显示混淆矩阵及其数据表"""
        dialog = tk.Toplevel(self)
        dialog.title("混淆矩阵可视化与数据")
        dialog.geometry("1200x800")

        # 创建左右分隔的框架
        main_frame = ttk.PanedWindow(dialog, orient='horizontal')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧：图形显示
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=1)

        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(self.confusion_matrix, cmap='Blues')
        fig.colorbar(im)

        classes = range(10)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title('混淆矩阵图示')

        # 添加数值标注
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{self.confusion_matrix[i, j]:.0f}',
                               ha="center", va="center",
                               color="white" if self.confusion_matrix[i, j] >
                                                self.confusion_matrix.max() / 2 else "black")

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()

        # 右侧：数据表格
        right_frame = ttk.Frame(main_frame)
        main_frame.add(right_frame, weight=1)

        # 创建表格
        table_frame = ttk.Frame(right_frame)
        table_frame.pack(fill='both', expand=True, padx=5)

        # 添加水平和垂直滚动条
        h_scroll = ttk.Scrollbar(table_frame, orient='horizontal')
        v_scroll = ttk.Scrollbar(table_frame, orient='vertical')

        # 创建表格视图
        table = ttk.Treeview(table_frame, columns=['pred_' + str(i) for i in range(10)],
                             show='headings',
                             xscrollcommand=h_scroll.set,
                             yscrollcommand=v_scroll.set)

        # 配置滚动条
        h_scroll.config(command=table.xview)
        h_scroll.pack(side='bottom', fill='x')
        v_scroll.config(command=table.yview)
        v_scroll.pack(side='right', fill='y')

        # 配置列标题
        for i in range(10):
            table.heading(f'pred_{i}', text=f'预测 {i}')
            table.column(f'pred_{i}', width=70, anchor='center')

        # 插入数据
        for i in range(10):
            values = [f'{self.confusion_matrix[i, j]:.0f}' for j in range(10)]
            table.insert('', 'end', values=values, tags=('row',))

        table.pack(fill='both', expand=True)

        # 添加统计信息
        stats_frame = ttk.LabelFrame(right_frame, text="统计信息")
        stats_frame.pack(fill='x', padx=5, pady=5)

        total_samples = np.sum(self.confusion_matrix)
        correct_predictions = np.sum(np.diag(self.confusion_matrix))
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        stats_text = (
            f"总样本数: {total_samples:.0f}\n"
            f"正确预测: {correct_predictions:.0f}\n"
            f"准确率: {accuracy:.2%}\n"
            f"每类准确率:\n"
        )

        for i in range(10):
            class_accuracy = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i, :])
            stats_text += f"类别 {i}: {class_accuracy:.2%}\n"

        ttk.Label(stats_frame, text=stats_text, justify='left').pack(padx=5, pady=5)

        # 添加保存按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)

        def save_data():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="保存混淆矩阵数据"
            )
            if file_path:
                np.savetxt(file_path, self.confusion_matrix,
                           delimiter=',', fmt='%.0f',
                           header=','.join([f'预测_{i}' for i in range(10)]),
                           comments='')
                messagebox.showinfo("成功", "数据已保存")

        ttk.Button(button_frame, text="保存数据", command=save_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side='right', padx=5)

    def save_denoising_results(self):
        """保存所有降噪处理结果"""
        if not hasattr(self, 'denoised_samples'):
            messagebox.showerror("错误", "没有可保存的降噪结果")
            return

        try:
            # 创建保存目录
            save_dir = filedialog.askdirectory(title="选择保存目录")
            if not save_dir:
                return

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_dir = os.path.join(save_dir, f"denoising_results_{timestamp}")
            os.makedirs(result_dir, exist_ok=True)

            # 保存图像和评估结果
            results_data = []
            for i, sample in enumerate(self.denoised_samples):
                # 保存图像
                img_dir = os.path.join(result_dir, f"sample_{i + 1}")
                os.makedirs(img_dir, exist_ok=True)

                # 原始图像
                orig_img = self.original_samples[i][0]
                orig_path = os.path.join(img_dir, "original.png")
                self.save_image(orig_img, orig_path)

                # 噪声图像
                noisy_img = self.noisy_samples[i][0]
                noisy_path = os.path.join(img_dir, "noisy.png")
                self.save_image(noisy_img, noisy_path)

                # 降噪后图像
                denoised_img = sample['image']
                denoised_path = os.path.join(img_dir, "denoised.png")
                self.save_image(denoised_img, denoised_path)

                # 计算所有评估指标
                noise_mse = torch.mean((noisy_img - orig_img) ** 2).item()
                noise_psnr = 10 * torch.log10(1.0 / noise_mse) if noise_mse > 0 else float('inf')
                noise_ssim = self.calculate_ssim(orig_img, noisy_img)

                denoised_mse = sample['metrics']['mse']
                denoised_psnr = sample['metrics']['psnr']
                denoised_ssim = self.calculate_ssim(orig_img, denoised_img)

                # 收集评估结果
                result_entry = {
                    'sample_id': i + 1,
                    'noise_metrics': {
                        'mse': noise_mse,
                        'psnr': noise_psnr,
                        'ssim': noise_ssim
                    },
                    'denoised_metrics': {
                        'mse': denoised_mse,
                        'psnr': denoised_psnr,
                        'ssim': denoised_ssim
                    },
                    'improvement': {
                        'mse_reduction': ((noise_mse - denoised_mse) / noise_mse * 100),
                        'psnr_increase': (denoised_psnr - noise_psnr),
                        'ssim_increase': ((denoised_ssim - noise_ssim) / noise_ssim * 100)
                    }
                }
                results_data.append(result_entry)

            # 保存评估结果到CSV
            results_df = pd.DataFrame(results_data)
            csv_path = os.path.join(result_dir, "evaluation_results.csv")
            results_df.to_csv(csv_path, index=False)

            # 生成并保存评估报告
            report_path = os.path.join(result_dir, "denoising_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("降噪处理评估报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"样本数量: {len(results_data)}\n\n")

                # 写入平均指标
                avg_metrics = results_df.mean()
                f.write("平均评估指标:\n")
                f.write("-" * 30 + "\n")
                f.write(f"噪声图像 MSE: {avg_metrics['noise_metrics.mse']:.6f}\n")
                f.write(f"噪声图像 PSNR: {avg_metrics['noise_metrics.psnr']:.2f}dB\n")
                f.write(f"噪声图像 SSIM: {avg_metrics['noise_metrics.ssim']:.4f}\n\n")
                f.write(f"降噪后 MSE: {avg_metrics['denoised_metrics.mse']:.6f}\n")
                f.write(f"降噪后 PSNR: {avg_metrics['denoised_metrics.psnr']:.2f}dB\n")
                f.write(f"降噪后 SSIM: {avg_metrics['denoised_metrics.ssim']:.4f}\n\n")
                f.write(f"平均改善:\n")
                f.write(f"MSE减少: {avg_metrics['improvement.mse_reduction']:.2f}%\n")
                f.write(f"PSNR提升: {avg_metrics['improvement.psnr_increase']:.2f}dB\n")
                f.write(f"SSIM提升: {avg_metrics['improvement.ssim_increase']:.2f}%\n")

            messagebox.showinfo("成功", f"结果已保存至目录：\n{result_dir}")

        except Exception as e:
            messagebox.showerror("错误", f"保存结果失败: {str(e)}")
            print(f"Error details: {str(e)}")

    def save_image(self, img_tensor, save_path):
        """保存图像张量为PNG文件"""
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.squeeze(0)

        img_np = (img_tensor.cpu().numpy() + 1) / 2  # 转换到[0,1]范围
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
        img_pil.save(save_path)

    def apply_heatmap_style(self, img_tensor):
        """将灰度图像转换为黄-紫热力图样式
        Args:
            img_tensor: 输入图像张量
        Returns:
            colored: 热力图风格的图像张量 [3, H, W]
        """
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.squeeze(0)

        # 确保值域在[0,1]范围内
        img_tensor = img_tensor.float()
        if img_tensor.min() < 0:
            img_tensor = (img_tensor + 1) / 2

        # 创建颜色通道
        colored = torch.zeros((3, img_tensor.shape[0], img_tensor.shape[1]), dtype=torch.float32)

        # 黄色（低值）到紫色（高值）的映射
        # 红色通道：高值减少
        colored[0] = torch.where(img_tensor < 0.5,
                                 torch.ones_like(img_tensor),
                                 1.0 - (img_tensor - 0.5) * 2)

        # 绿色通道：高值减少
        colored[1] = torch.where(img_tensor < 0.5,
                                 torch.ones_like(img_tensor),
                                 1.0 - (img_tensor - 0.5) * 2)

        # 蓝色通道：高值增加
        colored[2] = torch.where(img_tensor < 0.5,
                                 img_tensor * 2,
                                 torch.ones_like(img_tensor))

        # 增强对比度
        colored = torch.clamp(colored, 0, 1)

        return colored

    def show_samples(self, original_samples, noisy_samples, orig_title="原始图像", noisy_title="噪声图像"):
        """显示热力图样式的样本对比"""
        dialog = tk.Toplevel(self)
        dialog.title("图像对比")
        dialog.geometry("800x600")

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        for i in range(len(original_samples)):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', pady=10)

            # 原始图像
            orig_img, label = original_samples[i]
            orig_frame = ttk.LabelFrame(frame, text=f"{orig_title} (标签: {label})")
            orig_frame.pack(side='left', padx=10)

            try:
                # 应用热力图样式
                orig_colored = self.apply_heatmap_style(orig_img)
                orig_img_pil = transforms.ToPILImage()(orig_colored)
                orig_img_pil = orig_img_pil.resize((150, 150), Image.LANCZOS)
                orig_img_tk = ImageTk.PhotoImage(orig_img_pil)

                orig_label = ttk.Label(orig_frame, image=orig_img_tk)
                orig_label.image = orig_img_tk
                orig_label.pack(padx=5, pady=5)

                # 噪声图像
                noisy_img, _ = noisy_samples[i]
                noisy_frame = ttk.LabelFrame(frame, text=noisy_title)
                noisy_frame.pack(side='left', padx=10)

                # 应用热力图样式
                noisy_colored = self.apply_heatmap_style(noisy_img)
                noisy_img_pil = transforms.ToPILImage()(noisy_colored)
                noisy_img_pil = noisy_img_pil.resize((150, 150), Image.LANCZOS)
                noisy_img_tk = ImageTk.PhotoImage(noisy_img_pil)

                noisy_label = ttk.Label(noisy_frame, image=noisy_img_tk)
                noisy_label.image = noisy_img_tk
                noisy_label.pack(padx=5, pady=5)

                # 质量评估
                stats_frame = ttk.LabelFrame(frame, text="质量评估")
                stats_frame.pack(side='left', padx=10)

                mse = torch.mean((noisy_img - orig_img) ** 2).item()
                psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

                stats_text = f"MSE: {mse:.6f}\nPSNR: {psnr:.2f}dB"
                ttk.Label(stats_frame, text=stats_text, justify='left').pack(padx=5, pady=5)

            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=10)

    # 在 gui_recognition.py 的 RecognitionDisplay 类中

    def show_denoising_results(self):
        """显示降噪处理的详细结果"""
        if not hasattr(self, 'denoised_samples'):
            messagebox.showerror("错误", "没有降噪结果")
            return

        dialog = tk.Toplevel(self)
        dialog.title("降噪处理结果")
        dialog.geometry("1200x800")

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 创建滚动区域
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # 显示图像和评估结果
        for i, sample in enumerate(self.denoised_samples):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', pady=10)

            try:
                # 原始图像
                orig_img = self.original_samples[i][0]
                orig_frame = ttk.LabelFrame(frame, text="原始图像")
                orig_frame.pack(side='left', padx=10)
                self.display_image(orig_frame, orig_img)

                # 噪声图像
                noisy_img = self.noisy_samples[i][0]
                noisy_frame = ttk.LabelFrame(frame, text="噪声图像")
                noisy_frame.pack(side='left', padx=10)
                self.display_image(noisy_frame, noisy_img)

                # 降噪后图像
                denoised_img = sample['image']
                denoised_frame = ttk.LabelFrame(frame, text="降噪后图像")
                denoised_frame.pack(side='left', padx=10)
                self.display_image(denoised_frame, denoised_img)

                # 评估指标
                metrics_frame = ttk.LabelFrame(frame, text="质量评估")
                metrics_frame.pack(side='left', padx=10, fill='y')

                # 确保数据在[-1,1]范围内
                orig_img = torch.clamp(orig_img, -1, 1)
                noisy_img = torch.clamp(noisy_img, -1, 1)
                denoised_img = torch.clamp(denoised_img, -1, 1)

                # 原始-噪声对比
                noise_mse = torch.mean((noisy_img - orig_img) ** 2).item()
                # 使用正确的数据范围2.0（因为范围是-1到1）
                noise_psnr = 10 * torch.log10(torch.tensor(4.0 / noise_mse)) if noise_mse > 0 else float('inf')
                noise_psnr = noise_psnr.item()

                # 原始-降噪对比
                denoised_mse = torch.mean((denoised_img - orig_img) ** 2).item()
                denoised_psnr = 10 * torch.log10(torch.tensor(4.0 / denoised_mse)) if denoised_mse > 0 else float('inf')
                denoised_psnr = denoised_psnr.item()

                # SSIM计算
                noise_ssim = self.calculate_ssim(orig_img, noisy_img)
                denoised_ssim = self.calculate_ssim(orig_img, denoised_img)

                # 计算改善程度（添加了max(1e-10, ...)避免除零）
                mse_improvement = ((noise_mse - denoised_mse) / max(1e-10, noise_mse) * 100) if noise_mse > 0 else 0
                psnr_improvement = (denoised_psnr - noise_psnr) if noise_psnr != float('inf') else 0
                ssim_improvement = ((denoised_ssim - noise_ssim) / max(1e-10, 1 - noise_ssim) * 100) if noise_ssim < 1 else 0
                
                metrics_text = (
                    f"噪声图像评估:\n"
                    f"MSE: {noise_mse:.6f}\n"
                    f"PSNR: {noise_psnr:.2f} dB\n"
                    f"SSIM: {noise_ssim:.4f}\n\n"
                    f"降噪后评估:\n"
                    f"MSE: {denoised_mse:.6f}\n"
                    f"PSNR: {denoised_psnr:.2f} dB\n"
                    f"SSIM: {denoised_ssim:.4f}\n\n"
                    f"改善程度:\n"
                    f"MSE减少: {mse_improvement:.2f}%\n"
                    f"PSNR提升: {psnr_improvement:.2f} dB\n"
                    f"SSIM提升: {ssim_improvement:.2f}%"
                )
                ttk.Label(metrics_frame, text=metrics_text, justify='left').pack(padx=5, pady=5)

            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                continue

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 保存结果按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)
        ttk.Button(button_frame, text="保存所有结果",
                   command=self.save_denoising_results).pack(side='left', padx=5)
        ttk.Button(button_frame, text="关闭",
                   command=dialog.destroy).pack(side='right', padx=5)

    def display_image(self, frame, img_tensor, size=(150, 150)):
        """在指定框架中显示图像"""
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor.squeeze(0)

        img_np = (img_tensor.cpu().numpy() + 1) / 2  # 转换到[0,1]范围
        img_pil = Image.fromarray((img_np * 255).astype('uint8'))
        img_pil = img_pil.resize(size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)

        label = ttk.Label(frame, image=img_tk)
        label.image = img_tk  # 保持引用
        label.pack(padx=5, pady=5)

    def calculate_ssim(self, img1, img2, window_size=11):
        """计算结构相似性指数(SSIM)
        
        Args:
            img1: 输入图像1，可以是2D、3D或4D张量
            img2: 输入图像2，维度需要与img1匹配
            window_size: 高斯窗口大小
            
        Returns:
            float: 平均SSIM值
        """
        from torch.nn.functional import conv2d
        from math import exp
        
        # 确保输入是4D张量 [batch, channel, height, width]
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)
        elif img1.dim() == 3:
            img1 = img1.unsqueeze(0)
            
        if img2.dim() == 2:
            img2 = img2.unsqueeze(0).unsqueeze(0)
        elif img2.dim() == 3:
            img2 = img2.unsqueeze(0)
        
        # 确保输入在[0,1]范围内
        img1 = (img1 + 1) / 2  # 从[-1,1]映射到[0,1]
        img2 = (img2 + 1) / 2  # 从[-1,1]映射到[0,1]
            
        # 确保两个张量在同一设备上
        device = img1.device
        img2 = img2.to(device)
        
        # 创建高斯核
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
            return gauss/gauss.sum()
            
        def create_window(window_size, channel):
            _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
            return window.to(device)
            
        # 参数
        channel = img1.size(1)
        window = create_window(window_size, channel)
        
        # 计算均值和方差
        mu1 = conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        # SSIM参数 - 使用[0,1]范围的值
        C1 = 0.01 ** 2  # 对于范围在[0,1]的数据
        C2 = 0.03 ** 2
        
        ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8)
        
        return ssim_map.mean().item()
    def update_show_samples(self, original_samples, noisy_samples, orig_title="原始图像", noisy_title="噪声图像"):
        """显示热力图风格的样本图像对比"""
        dialog = tk.Toplevel(self)
        dialog.title("图像对比")
        dialog.geometry("800x600")

        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # 确定数据集类型
        style = 'fashion' if 'fmnist' in self.dataset_var.get() else 'mnist'

        for i in range(len(original_samples)):
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', pady=10)

            # 原始图像
            orig_img, label = original_samples[i]
            orig_frame = ttk.LabelFrame(frame, text=f"{orig_title} (标签: {label})")
            orig_frame.pack(side='left', padx=10)

            # 应用热力图风格
            orig_colored = apply_heatmap_style(orig_img, style)
            orig_img_pil = transforms.ToPILImage()(orig_colored)
            orig_img_pil = orig_img_pil.resize((150, 150), Image.LANCZOS)
            orig_img_tk = ImageTk.PhotoImage(orig_img_pil)

            orig_label = ttk.Label(orig_frame, image=orig_img_tk)
            orig_label.image = orig_img_tk
            orig_label.pack(padx=5, pady=5)

            # 噪声图像
            noisy_img, _ = noisy_samples[i]
            noisy_frame = ttk.LabelFrame(frame, text=noisy_title)
            noisy_frame.pack(side='left', padx=10)

            # 应用热力图风格
            noisy_colored = apply_heatmap_style(noisy_img, style)
            noisy_img_pil = transforms.ToPILImage()(noisy_colored)
            noisy_img_pil = noisy_img_pil.resize((150, 150), Image.LANCZOS)
            noisy_img_tk = ImageTk.PhotoImage(noisy_img_pil)

            noisy_label = ttk.Label(noisy_frame, image=noisy_img_tk)
            noisy_label.image = noisy_img_tk
            noisy_label.pack(padx=5, pady=5)

            # 质量评估
            stats_frame = ttk.LabelFrame(frame, text="质量评估")
            stats_frame.pack(side='left', padx=10)

            mse = torch.mean((noisy_img - orig_img) ** 2).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

            stats_text = f"MSE: {mse:.6f}\nPSNR: {psnr:.2f}dB"
            ttk.Label(stats_frame, text=stats_text, justify='left').pack(padx=5, pady=5)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=10)

    def denoise_with_ltp(self):
        """使用改进的降噪处理方法"""
        if not hasattr(self, 'noisy_samples') or not self.noisy_samples:
            messagebox.showerror("错误", "请先生成噪声数据")
            return

        self.denoised_samples = []
        try:
            for i, (noisy_img, label) in enumerate(self.noisy_samples):
                try:
                    # 确保输入是正确的形状和类型
                    if len(noisy_img.shape) == 4:
                        noisy_img = noisy_img.squeeze(0)
                    if len(noisy_img.shape) == 3:
                        noisy_img = noisy_img.squeeze(0)
                    
                    # 确保张量在CPU上处理
                    if torch.is_tensor(noisy_img):
                        noisy_img = noisy_img.cpu()
                    
                    # 保存原始图像用于后续比较
                    orig_img = self.original_samples[i][0].cpu()
                    if orig_img.dim() == 4:
                        orig_img = orig_img.squeeze(0)
                    if orig_img.dim() == 3:
                        orig_img = orig_img.squeeze(0)
                    
                    # 1. 将图像转换到[0,1]范围进行处理
                    img_min = noisy_img.min()
                    img_max = noisy_img.max()
                    img_scaled = (noisy_img - img_min) / (img_max - img_min + 1e-8)

                    # 2. 基于LTP的自适应阈值去噪
                    local_mean = torch.mean(img_scaled)
                    local_std = torch.std(img_scaled)
                    adaptive_threshold = local_mean - 0.2 * local_std

                    # 使用PyTorch的where进行阈值处理
                    background_mask = img_scaled < adaptive_threshold
                    img_denoised = torch.where(background_mask, torch.zeros_like(img_scaled), img_scaled)

                    # 3. 细节保持和增强
                    detail_threshold = adaptive_threshold * 1.5
                    detail_mask = img_denoised > detail_threshold
                    img_denoised = torch.where(detail_mask, img_denoised * 1.2, img_denoised)

                    # 4. 应用双边滤波进行边缘保持降噪
                    img_denoised = self.bilateral_filter(
                        img_denoised,
                        kernel_size=5,
                        sigma_space=1.0,
                        sigma_intensity=0.1
                    )

                    # 5. 将处理后的图像缩放到原始范围
                    result = img_denoised * (img_max - img_min) + img_min
                    result = torch.clamp(result, -1, 1)  # 确保在[-1,1]范围内
                    result = result.unsqueeze(0)  # 添加通道维度

                    # 6. 计算质量评估指标
                    # 确保张量形状匹配
                    if result.dim() > orig_img.dim():
                        result = result.squeeze(0)
                    
                    # 计算MSE（与原始图像比较）
                    mse = torch.mean((result - orig_img) ** 2).item()
                    
                    # 计算PSNR（使用原始图像作为参考）
                    data_range = 2.0  # 因为数据范围是[-1,1]
                    mse_for_psnr = torch.mean((result - orig_img) ** 2)
                    psnr = 10 * torch.log10(torch.tensor(data_range**2) / (mse_for_psnr + 1e-8)).item()
                    
                    # 计算SSIM
                    ssim = self.calculate_ssim(orig_img.unsqueeze(0), result.unsqueeze(0))

                    # 保存降噪结果和评估指标
                    self.denoised_samples.append({
                        'image': result,
                        'label': label,
                        'metrics': {
                            'mse': mse,
                            'psnr': psnr,
                            'ssim': ssim
                        }
                    })

                except Exception as e:
                    print(f"Error processing image {i}: {str(e)}")
                    continue

            if not self.denoised_samples:
                messagebox.showerror("错误", "未能成功处理任何图像")
                return

            # 显示结果并更新UI
            self.show_denoising_results()
            self.update_result_text("降噪处理完成")

        except Exception as e:
            error_msg = f"降噪处理失败: {str(e)}"
            messagebox.showerror("错误", error_msg)
            self.update_result_text(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

    def calculate_adaptive_threshold(self, image):
        """计算自适应阈值"""
        local_mean = torch.mean(image)
        local_std = torch.std(image)
        return local_mean - 0.2 * local_std

    def bilateral_filter(self, x, kernel_size, sigma_space, sigma_intensity):
        """改进的双边滤波实现，确保所有计算使用 PyTorch 张量

        Args:
            x (torch.Tensor): 输入图像张量
            kernel_size (int): 滤波核大小
            sigma_space (float): 空间高斯核的标准差
            sigma_intensity (float): 值域高斯核的标准差

        Returns:
            torch.Tensor: 滤波后的图像张量
        """
        # 确保输入是PyTorch张量
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        height, width = x.shape
        pad = kernel_size // 2
        padded = torch.zeros((height + 2 * pad, width + 2 * pad), device=x.device, dtype=torch.float32)
        padded[pad:-pad, pad:-pad] = x
        result = torch.zeros_like(x, dtype=torch.float32)

        # 创建空间权重核
        space_kernel = torch.zeros((kernel_size, kernel_size), device=x.device, dtype=torch.float32)
        center = kernel_size // 2

        # 使用网格坐标计算距离
        y_coords, x_coords = torch.meshgrid(
            torch.arange(kernel_size, device=x.device, dtype=torch.float32),
            torch.arange(kernel_size, device=x.device, dtype=torch.float32),
            indexing='ij'
        )
        distances = ((y_coords - center) ** 2 + (x_coords - center) ** 2)
        space_kernel = torch.exp(-distances / (2 * sigma_space ** 2))

        # 应用滤波器
        for i in range(height):
            for j in range(width):
                window = padded[i:i + kernel_size, j:j + kernel_size]
                center_val = x[i, j]

                # 计算强度权重
                intensity_diff = (window - center_val) ** 2
                intensity_weights = torch.exp(-intensity_diff / (2 * sigma_intensity ** 2))

                # 组合空间和强度权重
                weights = space_kernel * intensity_weights
                weights = weights / weights.sum()

                # 应用权重并求和
                result[i, j] = (window * weights).sum()

        return result


    def update_result_text(self, message):
        if hasattr(self, 'result_text'):
            self.result_text.configure(state='normal')
            self.result_text.insert(tk.END, f"{message}\n")
            self.result_text.see(tk.END)
            self.result_text.configure(state='disabled')

    def create_result_text(self):
        result_frame = ttk.LabelFrame(self, text="处理结果", padding=10)
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(result_frame, orient='vertical',
                                  command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def setup_ui(self):
        """设置UI界面"""
        # LTP数据加载区域
        ltp_frame = ttk.LabelFrame(self, text="LTP数据加载", padding=10)
        ltp_frame.pack(fill='x', padx=5, pady=5)

        file_frame = ttk.Frame(ltp_frame)
        file_frame.pack(fill='x')

        self.ltp_path_var = StringVar()
        ttk.Label(file_frame, text="LTP数据:").pack(side='left', padx=5)
        ttk.Entry(file_frame, textvariable=self.ltp_path_var).pack(side='left',
                                                                   fill='x',
                                                                   expand=True,
                                                                   padx=5)
        ttk.Button(file_frame, text="加载数据",
                   command=self.load_ltp_data).pack(side='left', padx=5)

        # 数据集选择区域
        dataset_frame = ttk.LabelFrame(self, text="数据集操作", padding=10)
        dataset_frame.pack(fill='x', padx=5, pady=5)

        dataset_controls = ttk.Frame(dataset_frame)
        dataset_controls.pack(fill='x')

        self.dataset_var = StringVar(value='mnist')
        ttk.Label(dataset_controls, text="当前数据集:").pack(side='left', padx=5)
        self.dataset_combo = ttk.Combobox(dataset_controls,
                                          textvariable=self.dataset_var,
                                          values=['mnist', 'fmnist'],
                                          state='readonly')
        self.dataset_combo.pack(side='left', padx=5)

        # 噪声控制
        noise_frame = ttk.LabelFrame(self, text="降噪处理", padding=10)
        noise_frame.pack(fill='x', padx=5, pady=5)

        noise_controls = ttk.Frame(noise_frame)
        noise_controls.pack(fill='x')

        ttk.Label(noise_controls, text="噪声强度:").pack(side='left', padx=5)
        self.noise_var = tk.DoubleVar(value=0.1)
        noise_scale = ttk.Scale(noise_controls, from_=0.0, to=1.0,
                                variable=self.noise_var, orient='horizontal')
        noise_scale.pack(side='left', fill='x', expand=True, padx=5)
        
        # 添加固定随机种子选项
        self.fixed_seed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(noise_controls, text="固定随机种子", 
                       variable=self.fixed_seed_var).pack(side='left', padx=5)

        button_frame = ttk.Frame(noise_frame)
        button_frame.pack(fill='x', pady=5)

        # 创建并配置按钮，初始状态设置
        self.generate_noise_btn = ttk.Button(button_frame,
                                             text="生成噪声数据",
                                             command=self.generate_noisy_data)
        self.generate_noise_btn.pack(side='left', padx=5)

        self.denoise_btn = ttk.Button(button_frame,
                                      text="LTP降噪",
                                      command=self.denoise_with_ltp,
                                      state='disabled')
        self.denoise_btn.pack(side='left', padx=5)


        # 可视化区域
        viz_frame = ttk.LabelFrame(self, text="训练可视化", padding=10)
        viz_frame.pack(fill='x', padx=5, pady=5)

        ttk.Button(viz_frame, text="显示混淆矩阵",
                   command=self.show_confusion_matrix).pack(side='left', padx=5)
        ttk.Button(viz_frame, text="显示权重热力图",
                   command=self.show_weight_heatmap).pack(side='left', padx=5)
        ttk.Button(viz_frame, text="显示LTP曲线",
                   command=self.show_ltp_curves).pack(side='left', padx=5)

        # 结果显示区域
        result_frame = ttk.LabelFrame(self, text="处理结果", padding=10)
        result_frame.pack(fill='both', expand=True, padx=5, pady=5)

        self.result_text = tk.Text(result_frame, height=10, width=50)
        scrollbar = ttk.Scrollbar(result_frame,
                                  orient='vertical',
                                  command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)

        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 图像显示区域
        self.image_frame = ttk.Frame(self)
        self.image_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 初始化状态变量
        self.is_data_loaded = False
        self.is_noise_generated = False
        self.is_denoised = False
    def load_ltp_data(self):
        dialog = tk.Toplevel(self)
        dialog.title("加载LTP数据")
        dialog.geometry("300x200")
        dialog.grab_set()

        def load_file():
            file_path = filedialog.askopenfilename(
                title="选择LTP数据文件",
                filetypes=[
                    ("Excel files", "*.xlsx *.xls"),
                    ("CSV files", "*.csv"),
                    ("All files", "*.*")
                ]
            )
            if file_path:
                try:
                    if self.ltp_processor.load_data(file_path):
                        self.ltp_processor.normalize_data(num_points=100)
                        self.ltp_data_loaded = True
                        self.ltp_path_var.set(file_path)
                        self.update_result_text("LTP数据文件加载成功")
                        dialog.destroy()
                        self.show_ltp_curves()
                    else:
                        raise Exception("数据加载失败")
                except Exception as e:
                    messagebox.showerror("错误", f"数据加载失败: {str(e)}")

        def load_folder():
            folder_path = filedialog.askdirectory(title="选择LTP数据文件夹")
            if folder_path:
                try:
                    if self.ltp_processor.load_batch_data(folder_path):
                        self.ltp_processor.normalize_data(num_points=100)
                        self.ltp_data_loaded = True
                        self.ltp_path_var.set(folder_path)
                        self.update_result_text("LTP数据文件夹加载成功")
                        dialog.destroy()
                        self.show_ltp_curves()
                    else:
                        raise Exception("数据加载失败")
                except Exception as e:
                    messagebox.showerror("错误", f"数据加载失败: {str(e)}")

        ttk.Button(dialog, text="选择单个文件",
                   command=load_file).pack(pady=10)
        ttk.Button(dialog, text="选择文件夹",
                   command=load_folder).pack(pady=10)
        ttk.Button(dialog, text="取消",
                   command=dialog.destroy).pack(pady=10)

    def load_default_datasets(self):
        if self.dataset_manager is None:
            self.update_result_text("警告：数据集管理器未初始化")
            return

        try:
            # 加载MNIST测试集
            mnist_loader = self.dataset_manager.get_dataloader('mnist', train=False, batch_size=1)
            self.test_datasets['mnist'] = []
            for i, (data, target) in enumerate(mnist_loader):
                if i < 100:  # 只加载100个样本用于演示
                    self.test_datasets['mnist'].append((data[0], target.item()))
                else:
                    break

            # 加载Fashion-MNIST测试集
            fmnist_loader = self.dataset_manager.get_dataloader('fmnist', train=False, batch_size=1)
            self.test_datasets['fmnist'] = []
            for i, (data, target) in enumerate(fmnist_loader):
                if i < 100:  # 只加载100个样本用于演示
                    self.test_datasets['fmnist'].append((data[0], target.item()))
                else:
                    break

            self.update_result_text("默认数据集加载完成")
        except Exception as e:
            self.update_result_text(f"加载默认数据集失败: {str(e)}")
            messagebox.showerror("错误", f"加载数据集失败: {str(e)}")

    def show_confusion_matrix(self):
        """显示混淆矩阵及其数据表"""
        if not hasattr(self, 'confusion_matrix') or self.confusion_matrix is None:
            messagebox.showerror("错误", "没有可用的混淆矩阵数据")
            return

        dialog = tk.Toplevel(self)
        dialog.title("混淆矩阵可视化与数据")
        dialog.geometry("1200x800")

        # 创建左右分隔的框架
        main_frame = ttk.PanedWindow(dialog, orient='horizontal')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧：图形显示
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=1)

        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(self.confusion_matrix, cmap='Blues')
        fig.colorbar(im)

        classes = range(10)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)

        ax.set_xlabel('预测类别')
        ax.set_ylabel('真实类别')
        ax.set_title('混淆矩阵图示')

        # 添加数值标注
        for i in range(len(classes)):
            for j in range(len(classes)):
                text = ax.text(j, i, f'{self.confusion_matrix[i, j]:.0f}',
                               ha="center", va="center",
                               color="white" if self.confusion_matrix[i, j] >
                                                self.confusion_matrix.max() / 2 else "black")

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()

        # 右侧：数据表格
        right_frame = ttk.Frame(main_frame)
        main_frame.add(right_frame, weight=1)

        # 创建表格
        table_frame = ttk.Frame(right_frame)
        table_frame.pack(fill='both', expand=True, padx=5)

        # 添加水平和垂直滚动条
        h_scroll = ttk.Scrollbar(table_frame, orient='horizontal')
        v_scroll = ttk.Scrollbar(table_frame, orient='vertical')

        # 创建表格视图
        table = ttk.Treeview(table_frame, columns=['pred_' + str(i) for i in range(10)],
                             show='headings',
                             xscrollcommand=h_scroll.set,
                             yscrollcommand=v_scroll.set)

        # 配置滚动条
        h_scroll.config(command=table.xview)
        h_scroll.pack(side='bottom', fill='x')
        v_scroll.config(command=table.yview)
        v_scroll.pack(side='right', fill='y')

        # 配置列标题
        for i in range(10):
            table.heading(f'pred_{i}', text=f'预测 {i}')
            table.column(f'pred_{i}', width=70, anchor='center')

        # 插入数据
        for i in range(10):
            values = [f'{self.confusion_matrix[i, j]:.0f}' for j in range(10)]
            table.insert('', 'end', values=values, tags=('row',))

        table.pack(fill='both', expand=True)

        # 添加统计信息
        stats_frame = ttk.LabelFrame(right_frame, text="统计信息")
        stats_frame.pack(fill='x', padx=5, pady=5)

        total_samples = np.sum(self.confusion_matrix)
        correct_predictions = np.sum(np.diag(self.confusion_matrix))
        accuracy = correct_predictions / total_samples if total_samples > 0 else 0

        stats_text = (
            f"总样本数: {total_samples:.0f}\n"
            f"正确预测: {correct_predictions:.0f}\n"
            f"准确率: {accuracy:.2%}\n"
            f"每类准确率:\n"
        )

        for i in range(10):
            class_accuracy = self.confusion_matrix[i, i] / np.sum(self.confusion_matrix[i, :])
            stats_text += f"类别 {i}: {class_accuracy:.2%}\n"

        ttk.Label(stats_frame, text=stats_text, justify='left').pack(padx=5, pady=5)

        # 添加保存按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side='right', padx=5)

    def generate_noisy_data(self):
        """生成噪声数据"""
        if not self.test_datasets:
            messagebox.showerror("错误", "数据集未加载")
            return

        current_dataset = self.dataset_var.get()
        dataset = self.test_datasets.get(current_dataset)
        if not dataset:
            messagebox.showerror("错误", "无法获取当前数据集")
            return

        try:
            # 获取当前设置
            noise_level = self.noise_var.get()
            use_fixed_seed = self.fixed_seed_var.get()
            
            # 如果使用固定种子，则设置固定种子
            seed = 42 if use_fixed_seed else None
            
            if seed is not None:
                # 设置随机种子用于样本选择，确保相同的样本被选中
                np.random.seed(seed)
                
            # 随机选择样本
            indices = np.random.choice(len(dataset), 5, replace=False)
            self.noisy_samples = []
            self.original_samples = []

            # 生成噪声数据
            for idx in indices:
                img, label = dataset[idx]
                self.original_samples.append((img, label))
                # 为每个样本使用不同的种子（基于主种子和样本索引）
                sample_seed = seed + idx if seed is not None else None
                noisy_img = self.add_noise(img, noise_level, sample_seed)
                self.noisy_samples.append((noisy_img, label))

            # 显示噪声样本
            self.show_samples(self.original_samples, self.noisy_samples,
                             "原始图像", "噪声图像")

            # 更新状态
            self.is_noise_generated = True
            self.denoise_btn['state'] = 'normal'

            # 记录噪声统计信息
            noise_stats = {
                'noise_level': noise_level,
                'num_samples': len(indices),
                'dataset': current_dataset,
                'use_fixed_seed': use_fixed_seed,
                'seed': seed,
                'mean_noise': torch.mean(torch.stack([n[0] for n in self.noisy_samples])).item(),
                'std_noise': torch.std(torch.stack([n[0] for n in self.noisy_samples])).item()
            }

            self.current_stats = noise_stats
            
            # 在结果中显示随机种子信息
            seed_info = f" (随机种子: {seed})" if seed is not None else ""
            self.update_result_text(f"已生成噪声数据，噪声强度: {noise_level:.2f}{seed_info}")

            return True

        except Exception as e:
            error_msg = f"生成噪声数据时出错: {str(e)}"
            messagebox.showerror("错误", error_msg)
            self.update_result_text(error_msg)
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def show_weight_heatmap(self):
        """显示权重矩阵热力图和数据表"""
        if not self.weight_matrices:
            messagebox.showerror("错误", "暂无权重矩阵数据")
            return

        dialog = tk.Toplevel(self)
        dialog.title("权重矩阵热力图与数据")
        dialog.geometry("1400x800")

        # 创建左右分隔的框架
        main_frame = ttk.PanedWindow(dialog, orient='horizontal')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧：热力图显示
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=1)

        # 计算平均权重矩阵
        avg_weight = np.mean(self.weight_matrices, axis=0)

        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        im = ax.imshow(avg_weight, cmap='seismic', aspect='auto')
        fig.colorbar(im, label='权重值 (Siemens)')

        ax.set_xlabel('输出神经元')
        ax.set_ylabel('输入神经元')
        ax.set_title('平均权重矩阵热力图')

        # 添加统计信息到图上
        stats_text = (
            f'统计信息:\n'
            f'最大值: {np.max(avg_weight):.2e} S\n'
            f'最小值: {np.min(avg_weight):.2e} S\n'
            f'平均值: {np.mean(avg_weight):.2e} S\n'
            f'标准差: {np.std(avg_weight):.2e} S'
        )
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='bottom')

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()

        # 右侧：数据表格和时间演化
        right_frame = ttk.Notebook(main_frame)
        main_frame.add(right_frame, weight=1)

        # 数据表格标签页
        table_frame = ttk.Frame(right_frame)
        right_frame.add(table_frame, text='权重数据表')

        # 创建表格
        table_container = ttk.Frame(table_frame)
        table_container.pack(fill='both', expand=True, padx=5, pady=5)

        # 添加滚动条
        h_scroll = ttk.Scrollbar(table_container, orient='horizontal')
        v_scroll = ttk.Scrollbar(table_container, orient='vertical')

        # 创建表格视图
        cols = [f'neuron_{i}' for i in range(avg_weight.shape[1])]
        table = ttk.Treeview(table_container, columns=cols,
                             show='headings',
                             xscrollcommand=h_scroll.set,
                             yscrollcommand=v_scroll.set)

        # 配置滚动条
        h_scroll.config(command=table.xview)
        h_scroll.pack(side='bottom', fill='x')
        v_scroll.config(command=table.yview)
        v_scroll.pack(side='right', fill='y')

        # 配置列标题
        for col in cols:
            table.heading(col, text=col)
            table.column(col, width=70, anchor='center')

        # 插入数据
        for i in range(avg_weight.shape[0]):
            values = [f'{avg_weight[i, j]:.2e}' for j in range(avg_weight.shape[1])]
            table.insert('', 'end', values=values, tags=('row',))

        table.pack(fill='both', expand=True)

        # 时间演化标签页
        evolution_frame = ttk.Frame(right_frame)
        right_frame.add(evolution_frame, text='权重演化')

        # 创建时间演化图
        fig_evo = plt.Figure(figsize=(8, 6))
        ax_evo = fig_evo.add_subplot(111)

        # 选择几个代表性神经元显示权重变化
        sample_neurons = [(0, 0), (0, -1), (-1, 0), (-1, -1)]  # 角落的神经元
        for i, j in sample_neurons:
            weights = [matrix[i, j] for matrix in self.weight_matrices]
            ax_evo.plot(weights, label=f'神经元 ({i},{j})')

        ax_evo.set_xlabel('训练步数')
        ax_evo.set_ylabel('权重值 (Siemens)')
        ax_evo.set_title('代表性神经元权重演化')
        ax_evo.legend()
        ax_evo.grid(True)

        canvas_evo = FigureCanvasTkAgg(fig_evo, master=evolution_frame)
        canvas_evo.draw()
        canvas_evo.get_tk_widget().pack(fill='both', expand=True)

        # 添加工具栏
        toolbar_evo = NavigationToolbar2Tk(canvas_evo, evolution_frame)
        toolbar_evo.update()

        # 添加保存按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)

        def save_weight_data():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".npz",
                filetypes=[("NumPy压缩文件", "*.npz"), ("All files", "*.*")],
                title="保存权重矩阵数据"
            )
            if file_path:
                np.savez(file_path,
                         average_weights=avg_weight,
                         weight_history=np.array(self.weight_matrices),
                         stats={
                             'max': np.max(avg_weight),
                             'min': np.min(avg_weight),
                             'mean': np.mean(avg_weight),
                             'std': np.std(avg_weight)
                         })
                messagebox.showinfo("成功", "权重数据已保存")

        ttk.Button(button_frame, text="保存数据", command=save_weight_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side='right', padx=5)

    def show_sample_pair(self, parent_frame, img1, label1, img2, label2, title1, title2):
        """显示一对图像样本的通用方法"""
        frame = ttk.Frame(parent_frame)
        frame.pack(fill='x', pady=10)

        # 显示第一张图像
        frame1 = ttk.LabelFrame(frame, text=f"{title1} ({label1})")
        frame1.pack(side='left', padx=10)

        img1_pil = transforms.ToPILImage()(img1.cpu())
        img1_pil = img1_pil.resize((150, 150), Image.LANCZOS)
        img1_tk = ImageTk.PhotoImage(img1_pil)

        label1_widget = ttk.Label(frame1, image=img1_tk)
        label1_widget.image = img1_tk
        label1_widget.pack(padx=5, pady=5)

        # 显示第二张图像
        frame2 = ttk.LabelFrame(frame, text=f"{title2} ({label2})")
        frame2.pack(side='left', padx=10)

        img2_pil = transforms.ToPILImage()(img2.cpu())
        img2_pil = img2_pil.resize((150, 150), Image.LANCZOS)
        img2_tk = ImageTk.PhotoImage(img2_pil)

        label2_widget = ttk.Label(frame2, image=img2_tk)
        label2_widget.image = img2_tk
        label2_widget.pack(padx=5, pady=5)

        # 计算并显示图像质量指标
        mse = torch.mean((img2 - img1) ** 2).item()
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')

        stats_frame = ttk.LabelFrame(frame, text="质量评估")
        stats_frame.pack(side='left', padx=10)
        stats_text = f"MSE: {mse:.6f}\nPSNR: {psnr:.2f}dB"
        ttk.Label(stats_frame, text=stats_text, justify='left').pack(padx=5, pady=5)

        return frame

    def show_samples_dialog(self, samples1, samples2, title1, title2, dialog_title="图像对比"):
        """显示两组样本的对比对话框"""
        dialog = tk.Toplevel(self)
        dialog.title(dialog_title)
        dialog.geometry("800x600")
        dialog.grab_set()

        # 创建滚动区域
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient='vertical', command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        # 显示样本对
        for (img1, label1), (img2, label2) in zip(samples1, samples2):
            self.show_sample_pair(scrollable_frame, img1, label1, img2, label2, title1, title2)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=10)

    def show_ltp_curves(self):
        """显示LTP/LTD特性曲线和数据表"""
        if not self.ltp_data_loaded:
            messagebox.showerror("错误", "请先加载LTP数据")
            return

        dialog = tk.Toplevel(self)
        dialog.title("LTP/LTD特性曲线与数据")
        dialog.geometry("1400x800")

        # 创建左右分隔的框架
        main_frame = ttk.PanedWindow(dialog, orient='horizontal')
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 左侧：曲线显示
        left_frame = ttk.Frame(main_frame)
        main_frame.add(left_frame, weight=1)

        fig = plt.Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        ltp_data = self.ltp_processor.normalized_data
        x_data = ltp_data['x']
        ltp_values = ltp_data['ltp']
        ltd_values = ltp_data['ltd']

        # 绘制主曲线
        ltp_line = ax.plot(x_data, ltp_values, 'b-', label='LTP', linewidth=2)[0]
        ltd_line = ax.plot(x_data, ltd_values, 'r-', label='LTD', linewidth=2)[0]

        ax.set_xlabel('归一化脉冲数')
        ax.set_ylabel('归一化权重变化')
        ax.set_title('LTP/LTD特性曲线')
        ax.grid(True)
        ax.legend()

        # 计算MSE和PSNR
        mse = np.mean((ltp_values - ltd_values) ** 2)
        psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
        
        # 添加统计信息
        stats_text = (
            f'LTP统计:\n'
            f'最大值: {np.max(ltp_values):.3f}\n'
            f'最小值: {np.min(ltp_values):.3f}\n'
            f'平均值: {np.mean(ltp_values):.3f}\n'
            f'标准差: {np.std(ltp_values):.3f}\n\n'
            f'LTD统计:\n'
            f'最大值: {np.max(ltd_values):.3f}\n'
            f'最小值: {np.min(ltd_values):.3f}\n'
            f'平均值: {np.mean(ltd_values):.3f}\n'
            f'标准差: {np.std(ltd_values):.3f}\n\n'
            f'质量评估:\n'
            f'MSE: {mse:.6f}\n'
            f'PSNR: {psnr:.2f} dB'
        )
        ax.text(1.02, 0.98, stats_text, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.8),
                fontsize=10, verticalalignment='top')

        canvas = FigureCanvasTkAgg(fig, master=left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(canvas, left_frame)
        toolbar.update()

        # 右侧：数据表格和分析
        right_frame = ttk.Notebook(main_frame)
        main_frame.add(right_frame, weight=1)

        # 数据表格标签页
        table_frame = ttk.Frame(right_frame)
        right_frame.add(table_frame, text='数据表')

        # 创建表格
        table_container = ttk.Frame(table_frame)
        table_container.pack(fill='both', expand=True, padx=5, pady=5)

        # 添加滚动条
        h_scroll = ttk.Scrollbar(table_container, orient='horizontal')
        v_scroll = ttk.Scrollbar(table_container, orient='vertical')

        # 创建表格视图
        table = ttk.Treeview(table_container,
                             columns=('index', 'pulse', 'ltp', 'ltd'),
                             show='headings',
                             xscrollcommand=h_scroll.set,
                             yscrollcommand=v_scroll.set)

        # 配置滚动条
        h_scroll.config(command=table.xview)
        h_scroll.pack(side='bottom', fill='x')
        v_scroll.config(command=table.yview)
        v_scroll.pack(side='right', fill='y')

        # 配置列
        table.heading('index', text='序号')
        table.heading('pulse', text='归一化脉冲数')
        table.heading('ltp', text='LTP值')
        table.heading('ltd', text='LTD值')

        table.column('index', width=70, anchor='center')
        table.column('pulse', width=100, anchor='center')
        table.column('ltp', width=100, anchor='center')
        table.column('ltd', width=100, anchor='center')

        # 插入数据
        for i in range(len(x_data)):
            values = (
                f'{i + 1}',
                f'{x_data[i]:.3f}',
                f'{ltp_values[i]:.3f}',
                f'{ltd_values[i]:.3f}'
            )
            table.insert('', 'end', values=values)

        table.pack(fill='both', expand=True)

        # 分析标签页
        analysis_frame = ttk.Frame(right_frame)
        right_frame.add(analysis_frame, text='数据分析')

        # 创建分析图
        fig_analysis = plt.Figure(figsize=(8, 6))
        ax1 = fig_analysis.add_subplot(211)
        ax2 = fig_analysis.add_subplot(212)

        # LTP-LTD差值分析
        difference = ltp_values - ltd_values
        ax1.plot(x_data, difference, 'g-', label='LTP-LTD差值')
        ax1.set_xlabel('归一化脉冲数')
        ax1.set_ylabel('差值')
        ax1.set_title('LTP-LTD差值分析')
        ax1.grid(True)
        ax1.legend()

        # 变化率分析
        ltp_rate = np.gradient(ltp_values, x_data)
        ltd_rate = np.gradient(ltd_values, x_data)
        ax2.plot(x_data, ltp_rate, 'b-', label='LTP变化率')
        ax2.plot(x_data, ltd_rate, 'r-', label='LTD变化率')
        ax2.set_xlabel('归一化脉冲数')
        ax2.set_ylabel('变化率')
        ax2.set_title('变化率分析')
        ax2.grid(True)
        ax2.legend()

        fig_analysis.tight_layout()

        canvas_analysis = FigureCanvasTkAgg(fig_analysis, master=analysis_frame)
        canvas_analysis.draw()
        canvas_analysis.get_tk_widget().pack(fill='both', expand=True)

        # 添加工具栏
        toolbar_analysis = NavigationToolbar2Tk(canvas_analysis, analysis_frame)
        toolbar_analysis.update()

        # 添加保存按钮
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill='x', pady=10)

        def save_ltp_data():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("所有文件", "*.*")],
                title="保存LTP/LTD数据"
            )
            if file_path:
                if file_path.endswith('.csv'):
                    pd.DataFrame({
                        '归一化脉冲数': x_data,
                        'LTP值': ltp_values,
                        'LTD值': ltd_values,
                        'LTP-LTD差值': difference,
                        'LTP变化率': ltp_rate,
                        'LTD变化率': ltd_rate
                    }).to_csv(file_path, index=False)
                else:
                    pd.DataFrame({
                        '归一化脉冲数': x_data,
                        'LTP值': ltp_values,
                        'LTD值': ltd_values,
                        'LTP-LTD差值': difference,
                        'LTP变化率': ltp_rate,
                        'LTD变化率': ltd_rate
                    }).to_excel(file_path, index=False)
                messagebox.showinfo("成功", "数据已保存")

        ttk.Button(button_frame, text="保存数据", command=save_ltp_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="关闭", command=dialog.destroy).pack(side='right', padx=5)





