
from gui_config import PresetConfig
import tkinter as tk
from tkinter import filedialog, StringVar, IntVar, DoubleVar, Text, END, messagebox

import threading

import random

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime



import pandas as pd

from tkinter import ttk, filedialog, Text, END, messagebox, StringVar
from PIL import Image, ImageTk

import torch

from torchvision import transforms
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
from data_processing import SynapticDataProcessor, load_current_time_data
from scipy.signal import find_peaks
from peak_detection_window import show_peak_detection_window

class ParameterSection(ttk.LabelFrame):
    """参数配置区域组件"""
    def __init__(self, parent, title, parameters, presets=None):
        super().__init__(parent, text=title, padding=10)
        self.parameters = {}

        if presets:
            preset_frame = ttk.Frame(self)
            preset_frame.pack(fill='x', pady=(0, 10))
            ttk.Label(preset_frame, text="预设配置:").pack(side='left', padx=5)
            self.preset_var = StringVar(value="optimal")
            preset_combo = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                      values=["optimal", "fast"], state='readonly', width=15)
            preset_combo.pack(side='left', padx=5)
            preset_combo.bind('<<ComboboxSelected>>',
                            lambda e: self.apply_preset(self.preset_var.get()))

        param_frame = ttk.Frame(self)
        param_frame.pack(fill='x')

        for i, (name, param_type, default, width) in enumerate(parameters):
            row = i // 2
            col = (i % 2) * 2

            ttk.Label(param_frame, text=f"{name}:").grid(row=row, column=col,
                                                       padx=5, pady=3, sticky='e')

            if param_type == "int":
                var = IntVar(value=default)
            elif param_type == "float":
                var = DoubleVar(value=default)
            else:
                var = StringVar(value=default)

            entry = ttk.Entry(param_frame, textvariable=var, width=width)
            entry.grid(row=row, column=col + 1, padx=5, pady=3, sticky='w')
            self.parameters[name] = var

    def apply_preset(self, preset_name):
        if preset_name == "optimal":
            config = PresetConfig.OPTIMAL
        else:
            config = PresetConfig.FAST

        for name, var in self.parameters.items():
            key = name.lower().replace(" ", "_")
            if key in config:
                var.set(config[key])

    def get_values(self):
        return {name: var.get() for name, var in self.parameters.items()}


class SynapticDataSection(ttk.LabelFrame):
    """突触数据配置区域组件"""

    def __init__(self, parent, log_callback=None):
        super().__init__(parent, text="突触数据配置", padding=10)
        self.log_callback = log_callback if log_callback else lambda x: None
        self.synaptic_processor = SynapticDataProcessor()
        self.use_manual_data = False  # 添加手动数据标志
        self.setup_ui()

    def toggle_manual_mode(self):
        """切换手动数据模式"""
        self.use_manual_data = self.manual_var.get()
        # 禁用/启用文件选择相关控件
        state = 'disabled' if self.use_manual_data else 'normal'
        self.file_entry.config(state=state)
        self.file_button.config(state=state)
        self.folder_button.config(state=state)
        
        if self.use_manual_data:
            self.log_callback("已切换到手动数据模式")
        else:
            self.log_callback("已切换到文件数据模式")

    def process_data(self):
        """处理数据时添加详细的错误捕获"""
        if not self.use_manual_data:
            file_path = self.synapse_file_var.get()
            if not file_path:
                messagebox.showerror("错误", "请选择突触数据文件或文件夹")
                return
        else:
            file_path = "手动输入数据"
            
        try:
            # 记录开始处理
            self.log_callback(f"开始处理{'手动' if self.use_manual_data else '文件'}数据...")

            points = int(self.points_var.get())
            self.log_callback(f"设定采样点数: {points}")

            if points <= 0:
                raise ValueError("采样点数必须大于0")

            # 1. 加载数据
            try:
                if self.use_manual_data:
                    success = self.synaptic_processor.load_manual_data()
                    self.log_callback("手动数据加载成功")
                else:
                    # 获取峰值检测参数
                    min_height = self.min_height_var.get()
                    min_distance = self.min_distance_var.get()
                    prominence = self.prominence_var.get()
                    
                    self.log_callback(f"处理数据参数: Height={min_height}, Distance={min_distance}, Prominence={prominence}")
                    
                    success = self.synaptic_processor.load_data(file_path, height=min_height, distance=min_distance, prominence=prominence)
                    self.log_callback(f"文件数据加载成功: {file_path}")
                    
                if not success:
                    raise ValueError("数据加载失败")
                    
            except Exception as e:
                error_msg = f"{'手动' if self.use_manual_data else '文件'}数据加载错误: {str(e)}"
                self.log_callback(error_msg)
                raise ValueError(error_msg)

            # 2. 获取峰值计数
            try:
                peak_count = self.synaptic_processor.get_peak_count()
                self.log_callback(f"检测到峰值数量: {peak_count}")
            except Exception as e:
                self.log_callback(f"峰值检测错误: {str(e)}")
                raise

            # 3. 验证采样点数
            try:
                self.synaptic_processor.validate_sampling_points(points)
                self.log_callback("采样点数验证通过")
            except Exception as e:
                self.log_callback(f"采样点数验证错误: {str(e)}")
                raise

            # 4. 归一化数据
            try:
                normalized_data = self.synaptic_processor.normalize_data(num_points=points)
                if normalized_data is None:
                    raise ValueError("归一化数据失败")
                self.log_callback("数据归一化完成")
            except Exception as e:
                self.log_callback(f"数据归一化错误: {str(e)}")
                raise

            # 处理成功
            self.log_callback("数据处理完成")
            # 自动显示归一化图表
            self.show_normalized_data()
            
            # messagebox.showinfo("成功", "数据处理完成，可以点击'保存归一化数据'保存结果")

        except ValueError as ve:
            error_msg = str(ve)
            self.log_callback(f"处理错误: {error_msg}")
            messagebox.showerror("错误", error_msg)
        except Exception as e:
            error_msg = f"处理数据时发生未预期的错误: {str(e)}"
            self.log_callback(error_msg)
            messagebox.showerror("错误", error_msg)

    def validate_data_format(data):
        """验证数据格式"""
        # 检查基本结构
        if data.shape[1] < 2:
            raise ValueError(f"数据格式错误：需要至少2列数据，当前只有{data.shape[1]}列")

        # 检查数据类型
        if not np.issubdtype(data.dtypes[0], np.number):
            raise ValueError("第一列(时间数据)必须是数值类型")
        if not np.issubdtype(data.dtypes[1], np.number):
            raise ValueError("第二列(电流数据)必须是数值类型")

        # 检查是否有空值
        if data.isnull().any().any():
            raise ValueError("数据中存在空值，请检查数据完整性")

    def setup_ui(self):
        # 手动数据选项
        manual_frame = ttk.Frame(self)
        manual_frame.pack(fill='x', pady=5)
        self.manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(manual_frame, text="使用手动输入数据", variable=self.manual_var,
                       command=self.toggle_manual_mode).pack(side='left', padx=5)
        
        # 数据文件选择
        self.file_frame = ttk.Frame(self)
        self.file_frame.pack(fill='x', pady=5)

        ttk.Label(self.file_frame, text="突触数据:").pack(side='left', padx=5)
        self.synapse_file_var = StringVar()
        self.file_entry = ttk.Entry(self.file_frame, textvariable=self.synapse_file_var)
        self.file_entry.pack(side='left', fill='x', expand=True, padx=5)
        self.file_button = ttk.Button(self.file_frame, text="选择文件", command=self.browse_file)
        self.file_button.pack(side='left', padx=2)
        self.folder_button = ttk.Button(self.file_frame, text="选择文件夹", command=self.browse_folder)
        self.folder_button.pack(side='left', padx=2)

        # LTP点数配置
        config_frame = ttk.Frame(self)
        config_frame.pack(fill='x', pady=5)

        ttk.Label(config_frame, text="LTP采样点数:").pack(side='left', padx=5)
        self.points_var = StringVar(value="100")
        ttk.Entry(config_frame, textvariable=self.points_var, width=10).pack(side='left', padx=5)

        # 峰值检测参数 (Min Height, Min Distance)
        ttk.Label(config_frame, text="最小高度:").pack(side='left', padx=5)
        self.min_height_var = DoubleVar(value=0.0)
        ttk.Entry(config_frame, textvariable=self.min_height_var, width=8).pack(side='left', padx=2)

        ttk.Label(config_frame, text="最小间距:").pack(side='left', padx=5)
        self.min_distance_var = IntVar(value=10)
        ttk.Entry(config_frame, textvariable=self.min_distance_var, width=8).pack(side='left', padx=2)
        
        # 隐藏的 Prominence 变量，由峰值检测窗口更新
        self.prominence_var = DoubleVar(value=0.0)

        # 处理按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', pady=5)

        ttk.Button(button_frame, text="峰值检测", command=self.show_peak_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="处理数据", command=self.process_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="保存归一化数据", command=self.save_normalized_data).pack(side='left', padx=5)
        # ttk.Button(button_frame, text="保存结果", command=self.save_results).pack(side='left', padx=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="选择突触数据文件",
            filetypes=[
                ("Excel文件", "*.xlsx *.xls"),
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.synapse_file_var.set(file_path)
            self.log_callback(f"选择突触数据文件: {file_path}")

    def browse_folder(self):
        folder_path = filedialog.askdirectory(title="选择突触数据文件夹")
        if folder_path:
            self.synapse_file_var.set(folder_path)
            self.log_callback(f"选择突触数据文件夹: {folder_path}")

    def show_peak_detection(self):
        """显示增强版峰值检测窗口"""
        file_path = self.synapse_file_var.get()
        if not file_path:
            messagebox.showwarning("警告", "请先选择数据文件")
            return
            
        try:
            # 启动峰值检测窗口
            show_peak_detection_window(self, self.synaptic_processor, file_path)
            
        except Exception as e:
            messagebox.showerror("错误", f"无法启动峰值检测窗口: {str(e)}")
            self.log_callback(f"启动峰值检测窗口失败: {str(e)}")



    def save_normalized_data(self):
        """保存归一化后的LTP和LTD数据到output文件夹"""
        if not hasattr(self.synaptic_processor, 'normalized_data') or \
                self.synaptic_processor.normalized_data is None:
            messagebox.showerror("错误", "请先处理数据")
            return

        try:
            # 确保output目录存在
            from run_simulation import get_output_dir
            output_dir = get_output_dir()
            
            # 使用固定文件名，覆盖旧文件
            filename = "normalized_synaptic_data.csv"
            save_path = os.path.join(output_dir, filename)

            # 准备数据：只有 ltp 和 ltd 两列
            data = {
                'ltp': self.synaptic_processor.normalized_data['ltp'],
                'ltd': self.synaptic_processor.normalized_data['ltd']
            }
            
            df = pd.DataFrame(data)
            df.to_csv(save_path, index=False)
            
            self.log_callback(f"归一化数据已保存至: {save_path}")
            messagebox.showinfo(
                "保存成功", 
                f"数据已保存至:\n{save_path}\n\n注意：该文件是固定文件名，下次保存时将直接覆盖此文件，请知悉。"
            )
            
        except Exception as e:
            error_msg = f"保存归一化数据失败: {str(e)}"
            self.log_callback(error_msg)
            messagebox.showerror("错误", error_msg)

    def show_normalized_data(self):
        if not hasattr(self.synaptic_processor, 'normalized_data') or \
                self.synaptic_processor.normalized_data is None:
            messagebox.showerror("错误", "请先处理数据")
            return

        # 创建新窗口显示归一化后的数据
        top = tk.Toplevel(self)
        top.title("归一化数据可视化")
        top.geometry("800x600")

        # 创建图形区域
        fig = plt.Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # 绘制归一化数据
        x_data = self.synaptic_processor.normalized_data['x']
        ltp_data = self.synaptic_processor.normalized_data['ltp']
        ltd_data = self.synaptic_processor.normalized_data['ltd']

        ax.plot(x_data, ltp_data, 'b-', label='LTP (归一化)')
        ax.plot(x_data, ltd_data, 'r-', label='LTD (归一化)')
        ax.set_xlabel('归一化脉冲数')
        ax.set_ylabel('归一化权重变化')
        ax.set_title('突触权重变化的归一化表示')
        ax.grid(True)
        ax.legend()

        # 将图形嵌入Tkinter窗口
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def save_results(self):
        if not hasattr(self.synaptic_processor, 'normalized_data') or \
                self.synaptic_processor.normalized_data is None:
            messagebox.showerror("错误", "没有可保存的数据")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data = {
                'x': self.synaptic_processor.normalized_data['x'],
                'ltp': self.synaptic_processor.normalized_data['ltp'],
                'ltd': self.synaptic_processor.normalized_data['ltd']
            }
            df = pd.DataFrame(data)
            df.to_csv(f"normalized_data_{timestamp}.csv", index=False)
            self.log_callback("结果已保存")
            messagebox.showinfo("成功", "数据已保存")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def get_normalized_data(self):
        """返回归一化后的数据供训练使用"""
        return self.synaptic_processor.normalized_data