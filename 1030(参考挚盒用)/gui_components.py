
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
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processing import SynapticDataProcessor

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
        self.use_bluetooth_data = False  # 添加蓝牙数据标志
        self.setup_ui()

    def toggle_manual_mode(self):
        """切换手动数据模式"""
        self.use_manual_data = self.manual_var.get()
        
        # 如果切换到手动模式，自动关闭蓝牙模式
        if self.use_manual_data and hasattr(self, 'bluetooth_var'):
            self.bluetooth_var.set(False)
            self.use_bluetooth_data = False
        
        # 禁用/启用文件选择相关控件
        state = 'disabled' if self.use_manual_data else 'normal'
        self.file_entry.config(state=state)
        self.file_button.config(state=state)
        self.folder_button.config(state=state)
        
        if self.use_manual_data:
            self.log_callback("已切换到手动数据模式")
        else:
            self.log_callback("已切换到文件数据模式")
    
    def toggle_bluetooth_mode(self):
        """切换蓝牙数据模式"""
        self.use_bluetooth_data = self.bluetooth_var.get()
        
        # 如果切换到蓝牙模式，自动关闭手动模式
        if self.use_bluetooth_data:
            self.manual_var.set(False)
            self.use_manual_data = False
        
        # 禁用/启用文件选择相关控件
        state = 'disabled' if self.use_bluetooth_data else 'normal'
        if not self.use_manual_data:
            self.file_entry.config(state=state)
            self.file_button.config(state=state)
            self.folder_button.config(state=state)
        
        if self.use_bluetooth_data:
            self.log_callback("已切换到蓝牙数据模式")
        else:
            self.log_callback("已关闭蓝牙数据模式")

    def process_data(self):
        """处理数据时添加详细的错误捕获，支持蓝牙直接数据和文件数据"""
        # 使用实例变量直接访问
        use_bluetooth_data = self.use_bluetooth_data
        
        if not self.use_manual_data and not use_bluetooth_data:
            file_path = self.synapse_file_var.get()
            if not file_path:
                messagebox.showerror("错误", "请选择突触数据文件或文件夹")
                return
        elif self.use_manual_data:
            file_path = "手动输入数据"
        else:
            file_path = "蓝牙直接数据"
            
        try:
            # 记录开始处理
            if use_bluetooth_data:
                self.log_callback("开始处理蓝牙直接数据...")
            else:
                self.log_callback(f"开始处理{'手动' if self.use_manual_data else '文件'}数据...")

            points = int(self.points_var.get())
            self.log_callback(f"设定采样点数: {points}")

            if points <= 0:
                raise ValueError("采样点数必须大于0")

            # 1. 加载数据
            try:
                if use_bluetooth_data:
                    # 使用直接设置的蓝牙数据
                    if not hasattr(self.synaptic_processor, 'time_data') or not hasattr(self.synaptic_processor, 'current_data'):
                        raise ValueError("蓝牙数据未设置或不完整")
                    
                    # 使用蓝牙数据进行处理
                    time_data = self.synaptic_processor.time_data
                    current_data = self.synaptic_processor.current_data
                    self.log_callback(f"蓝牙数据加载成功，共 {len(time_data)} 个数据点")
                    
                    # 提取峰值点 - 使用与文件数据相同的方法
                    peaks, peak_times, peak_currents = self.synaptic_processor.extract_peak_points(time_data, current_data)
                    self.synaptic_processor.peak_count = len(peaks)
                    self.log_callback(f"检测到 {self.synaptic_processor.peak_count} 个峰值点")
                    
                    # 验证峰值数量
                    if self.synaptic_processor.peak_count < 2:
                        raise ValueError("检测到的峰值点太少，无法进行LTP/LTD分析")
                    
                    # 将峰值点分为LTD和LTP两组 - 使用与文件数据相同的方法
                    ltd_times, ltd_currents, ltp_times, ltp_currents = self.synaptic_processor.split_peaks_into_ltd_ltp(peak_times, peak_currents)
                    self.log_callback(f"LTD组: {len(ltd_times)} 个点, LTP组: {len(ltp_times)} 个点")
                    
                    # 创建平滑的LTD和LTP曲线 - 使用与文件数据相同的方法
                    ltd_time_norm, ltd_smooth = self.synaptic_processor.create_preserving_ltd_curve(ltd_times, ltd_currents)
                    ltp_time_norm, ltp_smooth = self.synaptic_processor.create_preserving_ltp_curve(ltp_times, ltp_currents)
                    
                    # 为LTD和LTP创建独立的时间序列，范围都是[0,1]
                    ltd_time_full = np.linspace(0, 1, len(ltd_smooth))
                    ltp_time_full = np.linspace(0, 1, len(ltp_smooth))
                    
                    # 保存处理后的数据 - 保持LTD和LTP的独立性，与文件数据处理方式一致
                    self.synaptic_processor.ltp_data = np.column_stack((ltp_time_full, ltp_smooth))
                    self.synaptic_processor.ltd_data = np.column_stack((ltd_time_full, ltd_smooth))
                    
                    success = True
                elif self.use_manual_data:
                    success = self.synaptic_processor.load_manual_data()
                    self.log_callback("手动数据加载成功")
                    # 验证手动数据格式 - 只对手动数据进行验证
                    self.validate_data_format(self.synaptic_processor.raw_data)
                    self.log_callback("手动数据格式验证通过")
                else:
                    success = self.synaptic_processor.load_data(file_path)
                    self.log_callback(f"文件数据加载成功: {file_path}")
                    # 不再对文件数据进行格式验证，因为文件数据与手动数据格式不同
                    self.log_callback("文件数据处理完成")
                    
                if not success:
                    raise ValueError("数据加载失败")
                    
            except Exception as e:
                if use_bluetooth_data:
                    error_msg = f"蓝牙数据处理错误: {str(e)}"
                else:
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
            messagebox.showinfo("成功", "数据处理完成，可以点击'显示归一化数据'查看结果")

        except ValueError as ve:
            error_msg = str(ve)
            self.log_callback(f"处理错误: {error_msg}")
            messagebox.showerror("错误", error_msg)
        except Exception as e:
            error_msg = f"处理数据时发生未预期的错误: {str(e)}"
            self.log_callback(error_msg)
            messagebox.showerror("错误", error_msg)

    # 修改validate_data_format函数，添加self参数使其成为实例方法
    def validate_data_format(self, data):
        """验证数据格式"""
        # 首先检查数据是否为None
        if data is None:
            # 对于文件数据，load_data方法没有设置raw_data
            # 所以我们直接返回，不进行验证
            return
        
        # 对于手动数据，data是字典类型
        if isinstance(data, dict):
            # 检查字典中是否包含必要的键
            required_keys = ['pulses', 'ltp', 'ltd']
            for key in required_keys:
                if key not in data:
                    raise ValueError(f"手动数据格式错误：缺少{key}键")
            
            # 检查每个列表是否为空
            for key in required_keys:
                if len(data[key]) == 0:
                    raise ValueError(f"手动数据格式错误：{key}列表为空")
            
            return
        
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

    # 在按钮框架部分修改，添加峰值检测按钮
    def setup_ui(self):
        # 手动数据和蓝牙数据选项
        data_mode_frame = ttk.Frame(self)
        data_mode_frame.pack(fill='x', pady=5)
        
        # 手动数据选项
        self.manual_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_mode_frame, text="使用手动输入数据", variable=self.manual_var,
                       command=self.toggle_manual_mode).pack(side='left', padx=5)
        
        # 蓝牙数据选项
        self.bluetooth_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(data_mode_frame, text="使用蓝牙采集数据", variable=self.bluetooth_var,
                       command=self.toggle_bluetooth_mode).pack(side='left', padx=5)
        
        # 加载CSV文件按钮
        csv_frame = ttk.Frame(self)
        csv_frame.pack(fill='x', pady=5)
        ttk.Button(csv_frame, text="加载CSV数据(a.csv)", 
                  command=lambda: self.load_csv_data("a.csv")).pack(side='left', padx=5)
        
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
    
        # 处理按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', pady=5)
    
        # 在处理数据按钮左边添加峰值检测按钮
        ttk.Button(button_frame, text="峰值检测", command=self.show_peak_detection).pack(side='left', padx=5)
        ttk.Button(button_frame, text="处理数据", command=self.process_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="显示归一化数据", command=self.show_normalized_data).pack(side='left', padx=5)
        ttk.Button(button_frame, text="保存结果", command=self.save_results).pack(side='left', padx=5)
    
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
        
    def set_bluetooth_data(self, time_data, current_data):
        """直接设置蓝牙采集的时间-电流数据，绕过文件读取
        
        参数:
            time_data: 时间数据数组
            current_data: 电流数据数组
        """
        try:
            # 验证数据有效性
            if time_data is None or current_data is None:
                raise ValueError("时间数据或电流数据为空")
                
            if len(time_data) != len(current_data):
                raise ValueError("时间数据和电流数据长度不匹配")
            
            if len(time_data) == 0:
                raise ValueError("数据长度为0")
            
            # 直接设置处理器的原始数据
            if hasattr(self.synaptic_processor, 'raw_data'):
                # 创建一个临时字典来存储原始数据
                self.synaptic_processor.raw_data = {
                    'time': time_data,
                    'current': current_data
                }
            
            # 保存数据到synaptic_processor中供后续处理
            self.synaptic_processor.time_data = time_data
            self.synaptic_processor.current_data = current_data
            
            # 标记为蓝牙数据模式
            self.use_bluetooth_data = True
            
            # 如果有bluetooth_var，也设置它
            if hasattr(self, 'bluetooth_var'):
                self.bluetooth_var.set(True)
                # 自动禁用手动模式
                if hasattr(self, 'manual_var'):
                    self.manual_var.set(False)
                    self.use_manual_data = False
            
            self.log_callback("成功设置蓝牙采集数据，共 {} 个数据点".format(len(time_data)))
            self.log_callback("已启用蓝牙数据模式，可直接进行峰值检测和数据处理")
            return True
        except Exception as e:
            error_msg = "设置蓝牙数据失败: {}".format(str(e))
            self.log_callback(error_msg)
            return False
    
    def load_csv_data(self, csv_file_path):
        """从CSV文件加载数据并设置到突触数据处理器
        
        参数:
            csv_file_path: CSV文件路径
        """
        try:
            # 添加按钮点击反馈
            self.log_callback(f"正在加载CSV文件: {csv_file_path}")
            
            # 检查文件是否存在
            if not os.path.exists(csv_file_path):
                raise FileNotFoundError(f"文件不存在: {csv_file_path}")
            
            # 读取CSV文件
            df = pd.read_csv(csv_file_path)
            self.log_callback(f"CSV文件读取成功，共{len(df)}行数据，{len(df.columns)}列数据")
            
            # 打印前几行数据用于调试
            self.log_callback(f"CSV文件前3行数据:\n{df.head(3).to_string()}")
            
            # 尝试不同的列名组合来提取时间和电流数据
            time_column = None
            current_column = None
            
            # 检查常见的列名
            possible_time_columns = ['Time', 'time', '时间', 'Time(ms)', 'Time(s)']
            possible_current_columns = ['Current', 'current', '电流', 'Data', 'data']
            
            for col in possible_time_columns:
                if col in df.columns:
                    time_column = col
                    break
            
            for col in possible_current_columns:
                if col in df.columns:
                    current_column = col
                    break
            
            # 如果没有找到标准列名，使用前两列
            if time_column is None or current_column is None:
                if len(df.columns) >= 2:
                    time_column = df.columns[0]
                    current_column = df.columns[1]
                    self.log_callback(f"未找到标准列名，使用前两列: '{time_column}' 和 '{current_column}'")
                else:
                    raise ValueError("CSV文件至少需要包含两列数据")
            
            # 提取数据
            time_data = df[time_column].values
            current_data = df[current_column].values
            
            self.log_callback(f"从CSV文件加载数据：时间列='{time_column}'，电流列='{current_column}'，共{len(time_data)}个数据点")
            
            # 使用已有的set_bluetooth_data方法设置数据
            # 这样可以复用相同的数据处理逻辑
            success = self.set_bluetooth_data(time_data, current_data)
            
            if success:
                self.log_callback(f"成功从CSV文件 {csv_file_path} 加载数据")
                messagebox.showinfo("成功", f"数据加载成功！已从{csv_file_path}文件加载{len(time_data)}个数据点\n现在可以点击'峰值检测'或'处理数据'按钮")
            
            return success
        except Exception as e:
            error_msg = f"加载CSV文件失败: {str(e)}"
            self.log_callback(error_msg)
            messagebox.showerror("错误", error_msg)
            return False

    def show_peak_detection(self):
        """改进的峰值检测可视化功能，支持查看原始数据、处理后的数据和蓝牙直接数据"""
        # 添加详细的调试信息
        self.log_callback("=== 开始峰值检测功能 ===")
        
        # 检查synaptic_processor对象是否存在
        if not hasattr(self, 'synaptic_processor'):
            self.log_callback("错误: synaptic_processor对象不存在")
            messagebox.showerror("错误", "系统错误: 数据处理器不存在")
            return
        
        # 使用实例变量而不是重新计算
        use_bluetooth_data = self.use_bluetooth_data
        
        # 记录操作日志
        if use_bluetooth_data:
            self.log_callback("开始峰值检测可视化（蓝牙数据）...")
        elif self.use_manual_data:
            self.log_callback("开始峰值检测可视化（手动数据）...")
        else:
            self.log_callback("开始峰值检测可视化...")
            
        # 尝试从多个可能的数据位置获取数据
        import numpy as np
        time_data = None
        current_data = None
        data_source = "未知"
        
        # 如果使用蓝牙数据，从synaptic_processor获取数据
        if use_bluetooth_data and hasattr(self.synaptic_processor, 'time_data') and hasattr(self.synaptic_processor, 'current_data'):
            try:
                time_data = self.synaptic_processor.time_data
                current_data = self.synaptic_processor.current_data
                if time_data is not None and current_data is not None and len(time_data) > 0 and len(current_data) > 0:
                    data_source = "蓝牙直接数据"
                    self.log_callback(f"成功获取蓝牙数据，数据点数量: {len(time_data)}")
                else:
                    self.log_callback("警告: 蓝牙数据为空或不完整")
            except Exception as e:
                self.log_callback(f"读取蓝牙数据时出错: {str(e)}")
        
        # 首先检查是否有原始数据（直接从文件加载，未经过处理的数据）
        # 这里我们尝试直接从文件重新加载原始数据，以确保看到的是未处理的原始数据
        file_path = self.synapse_file_var.get() if hasattr(self, 'synapse_file_var') else ""
        if time_data is None and current_data is None and file_path and not self.use_manual_data:
            try:
                import pandas as pd
                # 尝试直接读取Excel文件获取原始数据
                data = pd.read_excel(file_path)
                if data.shape[1] >= 2:
                    # 直接使用前两列作为时间和电流数据
                    time_data = data.iloc[:, 0].values
                    current_data = data.iloc[:, 1].values
                    data_source = "原始文件数据"
                    self.log_callback(f"成功读取原始文件数据，形状: {data.shape}")
            except Exception as e:
                self.log_callback(f"读取原始文件数据时出错: {str(e)}")
        
        # 如果没有获取到原始数据，再尝试使用处理后的数据
        if time_data is None and current_data is None:
            if hasattr(self.synaptic_processor, 'ltp_data') and self.synaptic_processor.ltp_data is not None:
                try:
                    if isinstance(self.synaptic_processor.ltp_data, np.ndarray):
                        if len(self.synaptic_processor.ltp_data.shape) >= 2 and self.synaptic_processor.ltp_data.shape[1] >= 2:
                            # 提取时间和电流数据
                            time_data = self.synaptic_processor.ltp_data[:, 0]
                            current_data = self.synaptic_processor.ltp_data[:, 1]
                            data_source = "处理后的LTP数据"
                            self.log_callback(f"使用处理后的LTP数据")
                except Exception as e:
                    self.log_callback(f"从ltp_data提取数据时出错: {str(e)}")
        
        # 数据准备完成，开始可视化部分
        try:
            # 先导入必要的库
            import tkinter as tk
            from tkinter import messagebox, ttk
            
            # 如果没有数据，显示错误
            if time_data is None or current_data is None:
                self.log_callback("错误: 没有找到有效数据")
                messagebox.showerror("错误", "请先通过'选择文件'加载数据")
                return
            
            self.log_callback("开始创建可视化窗口...")
            
            # 创建新窗口
            top = tk.Toplevel(self)
            top.title("峰值检测可视化")
            top.geometry("900x700")
            
            # 导入必要的可视化库
            import matplotlib
            matplotlib.use('TkAgg')  # 使用TkAgg后端
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            import matplotlib.pyplot as plt
            from scipy.signal import find_peaks
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 创建图形和子图
            fig = plt.Figure(figsize=(10, 8))
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2)
            
            # 检测峰值
            self.log_callback("执行峰值检测算法...")
            peaks, properties = find_peaks(current_data, height=0)
            self.log_callback(f"检测到 {len(peaks)} 个峰值")
            
            # 绘制原始数据与峰值
            ax1.plot(time_data, current_data, 'b-', linewidth=1, label=f'{data_source}')
            ax1.plot(time_data[peaks], current_data[peaks], 'ro', markersize=4, label=f'检测峰值 ({len(peaks)}个)')
            ax1.set_xlabel('时间')
            ax1.set_ylabel('值')
            ax1.set_title(f'原始数据与峰值检测 - {data_source}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制峰值间隔分析
            if len(peaks) > 1:
                peak_intervals = np.diff(time_data[peaks])
                ax2.plot(range(len(peak_intervals)), peak_intervals, 'g-o', markersize=3)
                ax2.set_xlabel('峰值序号')
                ax2.set_ylabel('峰值间隔')
                ax2.set_title('峰值间隔分析')
                ax2.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_interval = np.mean(peak_intervals)
                std_interval = np.std(peak_intervals)
                ax2.axhline(y=mean_interval, color='r', linestyle='--', 
                           label=f'平均间隔: {mean_interval:.6f}')
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, '峰值数量不足，无法进行间隔分析', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes)
            
            # 调整布局
            fig.tight_layout()
            
            # 创建画布并添加到窗口
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
            # 添加按钮区域
            button_frame = ttk.Frame(top)
            button_frame.pack(pady=10)
            
            # 添加重新加载原始数据按钮
            ttk.Button(button_frame, text="重新加载原始数据", command=lambda: self.reload_original_data(top, file_path)).pack(side='left', padx=5)
            # 添加关闭按钮
            ttk.Button(button_frame, text="关闭", command=top.destroy).pack(side='left', padx=5)
            
            # 添加数据信息标签
            info_frame = ttk.Frame(top)
            info_frame.pack(fill='x', padx=10, pady=5)
            ttk.Label(info_frame, text=f"数据源: {data_source}, 数据点数量: {len(time_data)}, 检测到峰值: {len(peaks)}").pack(anchor='w')
            
            self.log_callback("峰值检测可视化完成")
            
        except Exception as e:
            error_msg = f"可视化过程错误: {str(e)}"
            self.log_callback(error_msg)
            messagebox.showerror("错误", error_msg)
            
    def reload_original_data(self, top_window, file_path):
        """重新加载并显示原始文件数据"""
        if not file_path:
            messagebox.showerror("错误", "没有选择文件")
            return
        
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from scipy.signal import find_peaks
            
            # 直接读取原始数据
            data = pd.read_excel(file_path)
            if data.shape[1] >= 2:
                time_data = data.iloc[:, 0].values
                current_data = data.iloc[:, 1].values
                
                # 检测峰值
                peaks, properties = find_peaks(current_data, height=0)
                
                # 重新创建图表
                for widget in top_window.winfo_children():
                    if widget.winfo_class() == 'Canvas':  # 找到画布并删除
                        widget.destroy()
                
                # 创建新的图形
                fig = plt.Figure(figsize=(10, 8))
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
                
                # 绘制原始数据与峰值
                ax1.plot(time_data, current_data, 'b-', linewidth=1, label='原始文件数据')
                ax1.plot(time_data[peaks], current_data[peaks], 'ro', markersize=4, label=f'检测峰值 ({len(peaks)}个)')
                ax1.set_xlabel('时间')
                ax1.set_ylabel('值')
                ax1.set_title('原始数据与峰值检测 - 原始文件数据')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # 绘制峰值间隔分析
                if len(peaks) > 1:
                    peak_intervals = np.diff(time_data[peaks])
                    ax2.plot(range(len(peak_intervals)), peak_intervals, 'g-o', markersize=3)
                    ax2.set_xlabel('峰值序号')
                    ax2.set_ylabel('峰值间隔')
                    ax2.set_title('峰值间隔分析')
                    ax2.grid(True, alpha=0.3)
                    
                    # 添加统计信息
                    mean_interval = np.mean(peak_intervals)
                    std_interval = np.std(peak_intervals)
                    ax2.axhline(y=mean_interval, color='r', linestyle='--', 
                               label=f'平均间隔: {mean_interval:.6f}')
                    ax2.legend()
                else:
                    ax2.text(0.5, 0.5, '峰值数量不足，无法进行间隔分析', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax2.transAxes)
                
                # 调整布局
                fig.tight_layout()
                
                # 创建新画布并添加到窗口
                from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
                canvas = FigureCanvasTkAgg(fig, master=top_window)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)
                
                # 更新信息标签
                for widget in top_window.winfo_children():
                    if widget.winfo_class() == 'TFrame':
                        for child in widget.winfo_children():
                            if child.winfo_class() == 'TLabel':
                                child.config(text=f"数据源: 原始文件数据, 数据点数量: {len(time_data)}, 检测到峰值: {len(peaks)}")
                
                self.log_callback("已重新加载并显示原始数据")
        except Exception as e:
            error_msg = f"重新加载原始数据时出错: {str(e)}"
            self.log_callback(error_msg)
            messagebox.showerror("错误", error_msg)