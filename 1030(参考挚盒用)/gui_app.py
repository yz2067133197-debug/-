import tkinter as tk
from tkinter import ttk, filedialog, StringVar, IntVar, DoubleVar, messagebox
import threading
import time
from PIL import Image, ImageTk
from torchvision import transforms
import matplotlib.pyplot as plt
import os

# 导入蓝牙数据采集器
try:
    from bluetooth_data_collector import get_bluetooth_collector
    BLUETOOTH_AVAILABLE = True
except ImportError:
    # 允许导入失败，但不在启动时显示提示
    BLUETOOTH_AVAILABLE = False
    get_bluetooth_collector = None

from gui_config import PresetConfig
from gui_components import ParameterSection, SynapticDataSection
from gui_visualization import WeightVisualization
from gui_recognition import RecognitionDisplay
from dataset_manager import DatasetManager
from utils import plot_input_data, fuzzy_process_image
from data_processing import load_current_time_data
from run_simulation import run_simulation
import random
# 在gui_app.py的顶部添加导入
from handwriting_recognition import add_handwriting_recognition
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import torch
import torch.nn as nn
from snn_scientific_implementation import ScientificSNN
from training_manager import TrainingManager, STDPTrainingConfig
class App:
    def __init__(self, master):
        self.master = master
        self.setup_window()
        self.create_styles()
        self.dataset_manager = DatasetManager()
        # 添加训练状态标志
        self.is_training = False

        # 初始化变量
        self.recognition_results = []
        self.current_accuracy = None
        self.log_text = None
        self.train_canvas = None
        self.progress = None
        self.file_var = None
        self.dataset_var = None
        self.images_display_frame = None
        self.preview_dataset_combo = None
        self.preview_dataset_var = None
        self.recognition_display = None
        self.accuracy_var = None
        self.notebook = None
        self.image_tab = None
        
        # 初始化SNN优化参数变量
        self.snn_optimization_vars = {
            'enable_optimization': tk.BooleanVar(value=False),
            'dynamic_range_factor': tk.DoubleVar(value=1.0),
            'noise_robustness': tk.DoubleVar(value=0.0),
            'regularization_strength': tk.DoubleVar(value=0.0),
            'feature_enhancement': tk.BooleanVar(value=False),
            'adaptive_scaling': tk.BooleanVar(value=False),
            'preset_strategy': tk.StringVar(value='high_accuracy')
        }
        
        # 初始化数据处理器
        from data_processing import SynapticDataProcessor
        self.data_processor = SynapticDataProcessor()
        # 配置SNN优化参数为默认关闭状态
        self.data_processor.configure_snn_optimization(
            dynamic_range_factor=1.0,
            temporal_diversity=False,
            noise_robustness=0.0,
            feature_enhancement=False,
            adaptive_scaling=False,
            regularization_strength=0.0
        )
        
        # 初始化训练相关变量
        self.training_manager = None  # 延迟初始化
        self.optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
        # 初始化训练参数变量
        self.epochs_var = tk.StringVar(value='10')
        self.batch_size_var = tk.StringVar(value='32')
        self.learning_rate_var = tk.StringVar(value='0.001')
        self.hidden_layers_var = tk.StringVar(value='3')
        self.hidden_neurons_var = tk.StringVar(value='128')
        self.time_steps_var = tk.StringVar(value='20')
        
        # 训练状态变量
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value='准备就绪')
        
        # 蓝牙数据采集相关变量
        self.bluetooth_collector = None
        self.data_points_var = tk.StringVar(value='200')  # 默认采集200个数据
        self.auto_train_after_collection = tk.BooleanVar(value=False)  # 采集后自动训练，默认关闭
        self.continuous_collection_var = tk.BooleanVar(value=True)
        
        # LTP/LTD分析相关变量
        self.ltp_ltd_window_var = tk.StringVar(value='50')  # 分析窗口大小
        self.auto_process_var = tk.BooleanVar(value=False)  # 自动处理开关
        self.process_interval_var = tk.StringVar(value='2000')  # 自动处理间隔（毫秒）
        
        # 实时数据存储
        self.time_data = []
        self.current_data = [[] for _ in range(8)]  # 最多支持8通道
        
        # LTP/LTD数据存储
        self.ltp_data = []
        self.ltd_data = []
        self.ltp_ltd_time_data = []

        # 创建布局
        self.create_layout()

        self.append_log("系统已启动，使用优化预设参数配置")
        self.append_log("MNIST数据集：训练集60,000样本，测试集10,000样本")
        self.append_log("预设批次大小：32，对应完整训练需要1875批次，完整测试需要313批次")
        self.append_log("SNN优化功能已启用，可在数据配置中调整参数")

    # 在 gui_app.py 的 App 类中添加和修改以下方法

    def create_bluetooth_data_collection_panel(self, parent):
        """创建蓝牙数据采集面板"""
        # 标题
        title_frame = ttk.Frame(parent)
        title_frame.pack(fill='x', padx=10, pady=10)
        ttk.Label(title_frame, text="蓝牙电流数据采集", font=('Arial', 14, 'bold')).pack()
        
        # 显示说明文本
        ttk.Label(title_frame, text="点击'连接设备'按钮搜索并连接到CM1051设备", 
                 font=('Arial', 10)).pack()
        ttk.Label(title_frame, text="连接成功后可开始采集时间-电流数据对", 
                 font=('Arial', 10)).pack()
        
        # 参数设置和设备状态区域容器
        params_device_frame = ttk.Frame(parent)
        params_device_frame.pack(fill='x', padx=10, pady=5)
        
        # 参数设置区域
        params_frame = ttk.LabelFrame(params_device_frame, text="采集参数设置", padding=10)
        params_frame.pack(side='left', fill='x', expand=True, padx=(0, 5))
        
        # 数据数量设置
        ttk.Label(params_frame, text="采集数据数:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        data_points_entry = ttk.Entry(params_frame, textvariable=self.data_points_var, width=10)
        data_points_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(params_frame, text="个 (时间-电流数据)").grid(row=0, column=2, sticky='w', padx=5, pady=5)
        
        # 自动训练选项
        auto_train_check = ttk.Checkbutton(params_frame, text="采集完成后自动开始训练", 
                                          variable=self.auto_train_after_collection)
        auto_train_check.grid(row=1, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        
        # 持续采集模式选项
        continuous_check = ttk.Checkbutton(params_frame, text="使用持续采集模式", 
                                          variable=self.continuous_collection_var)
        continuous_check.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        
        # 设备信息区域
        device_frame = ttk.LabelFrame(params_device_frame, text="设备状态", padding=10)
        device_frame.pack(side='left', fill='x', expand=True, padx=(5, 0))
        
        self.bluetooth_status_var = tk.StringVar(value="未连接")
        self.collection_status_var = tk.StringVar(value="未开始")
        self.progress_var_bluetooth = tk.DoubleVar(value=0.0)
        
        ttk.Label(device_frame, text="连接状态:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(device_frame, textvariable=self.bluetooth_status_var).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        ttk.Label(device_frame, text="采集状态:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        ttk.Label(device_frame, textvariable=self.collection_status_var).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # 进度条
        ttk.Label(device_frame, text="采集进度:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        self.bluetooth_progress_bar = ttk.Progressbar(device_frame, mode='determinate', 
                                                      variable=self.progress_var_bluetooth)
        self.bluetooth_progress_bar.grid(row=2, column=1, columnspan=2, sticky='ew', padx=5, pady=2)
        
        # 控制按钮区域
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', padx=10, pady=10)
        
        self.start_bluetooth_button = ttk.Button(control_frame, text="开始采集", 
                                              command=self.start_bluetooth_collection,
                                              style='Action.TButton',
                                              state='disabled')  # 初始禁用，连接后启用
        self.start_bluetooth_button.pack(side='left', padx=5)
        
        self.stop_bluetooth_button = ttk.Button(control_frame, text="停止采集", 
                                               command=self.stop_bluetooth_collection,
                                               state='disabled')
        self.stop_bluetooth_button.pack(side='left', padx=5)
        
        self.connect_bluetooth_button = ttk.Button(control_frame, text="连接设备", 
                                                  command=self.connect_bluetooth_device)
        self.connect_bluetooth_button.pack(side='left', padx=5)
        
        # 实时曲线图区域
        graph_frame = ttk.LabelFrame(parent, text="实时数据曲线", padding=10)
        graph_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # 创建实时曲线图
        self.create_real_time_graph(graph_frame)
        
        # 数据显示区域
        data_frame = ttk.LabelFrame(parent, text="实时数据", padding=10)
        data_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_display = tk.Text(data_frame, height=10, width=50)
        data_scrollbar = ttk.Scrollbar(data_frame, orient='vertical', 
                                       command=self.data_display.yview)
        self.data_display.configure(yscrollcommand=data_scrollbar.set)
        data_scrollbar.pack(side='right', fill='y')
        self.data_display.pack(side='left', fill='both', expand=True)
        
        # LTP/LTD分析区域
        ltp_ltd_frame = ttk.LabelFrame(parent, text="LTP/LTD实时分析", padding=10)
        ltp_ltd_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # LTP/LTD分析参数设置
        params_frame = ttk.Frame(ltp_ltd_frame)
        params_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(params_frame, text="分析窗口大小: ").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.ltp_ltd_window_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="自动处理间隔(ms): ").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.process_interval_var, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Checkbutton(params_frame, text="自动处理", 
                        variable=self.auto_process_var, 
                        command=self.toggle_auto_process).grid(row=0, column=4, padx=5, pady=5)
        
        ttk.Button(params_frame, text="手动处理", 
                  command=self.process_ltp_ltd_data).grid(row=0, column=5, padx=5, pady=5)
        
        # LTP/LTD曲线图
        self.ltp_ltd_fig, self.ltp_ltd_ax = plt.subplots(figsize=(6, 4))
        self.ltp_ltd_ax.set_title('LTP/LTD 实时分析')
        self.ltp_ltd_ax.set_xlabel('时间 (s)')
        self.ltp_ltd_ax.set_ylabel('电流变化率')
        self.ltp_ltd_ax.grid(True)
        
        # 初始化LTP/LTD曲线
        self.ltp_line, = self.ltp_ltd_ax.plot([], [], 'r-', label='LTP')
        self.ltd_line, = self.ltp_ltd_ax.plot([], [], 'b-', label='LTD')
        self.ltp_ltd_ax.legend()
        
        # 创建画布
        self.ltp_ltd_canvas = FigureCanvasTkAgg(self.ltp_ltd_fig, master=ltp_ltd_frame)
        self.ltp_ltd_canvas.draw()
        self.ltp_ltd_canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # 自动处理相关变量
        self.auto_process_timer = None
    
    def append_bluetooth_log(self, message):
        """添加蓝牙日志"""
        self.append_log(f"[蓝牙采集] {message}")
        # 在数据显示区域也显示重要信息
        if "已采集" in message or "完成" in message or "错误" in message:
            self.data_display.insert('end', f"{message}\n")
            self.data_display.see('end')
    
    def update_bluetooth_data_display(self, data_info):
        """更新蓝牙数据显示"""
        def update_ui():
            if 'data_point' in data_info:
                data_point = data_info['data_point']
                time_val = data_point['Time']
                current_vals = data_point['Data']
                
                display_text = f"时间: {time_val:.3f}s, 电流: {[f'{val:.2e}' for val in current_vals]}\n"
                self.data_display.insert('end', display_text)
                self.data_display.see('end')
                
                # 更新实时曲线图
                self.update_real_time_graph(time_val, current_vals)
            
            # 更新进度
            if 'progress' in data_info:
                self.progress_var_bluetooth.set(data_info['progress'] * 100)
                self.collection_status_var.set(f"已采集 {data_info['collected_points']}/{data_info['target_points']} 个")
            else:
                # 持续采集模式
                if 'collected_points' in data_info:
                    self.collection_status_var.set(f"已采集 {data_info['collected_points']} 个")
            
            # 强制刷新UI
            self.master.update_idletasks()
        
        # 确保UI更新在主线程中进行
        self.master.after(0, update_ui)
    
    def create_real_time_graph(self, parent):
        """创建实时曲线图"""
        self.real_time_fig = plt.Figure(figsize=(10, 6))
        self.real_time_axes = self.real_time_fig.add_subplot(111)
        self.real_time_axes.set_title('实时电流数据')
        self.real_time_axes.set_xlabel('时间 (s)')
        self.real_time_axes.set_ylabel('电流 (mA)')
        self.real_time_axes.grid(True)
        
        # 初始化数据存储
        self.recent_time_data = [[] for _ in range(8)]
        self.recent_channel_data = [[] for _ in range(8)]
        
        # 创建曲线
        self.real_time_lines = []
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange']
        for i in range(8):
            line, = self.real_time_axes.plot([], [], color=colors[i], label=f'通道 {i+1}')
            self.real_time_lines.append(line)
        
        self.real_time_axes.legend()
        
        # 创建画布
        self.real_time_graph = FigureCanvasTkAgg(self.real_time_fig, master=parent)
        self.real_time_graph.draw()
        self.real_time_graph.get_tk_widget().pack(fill='both', expand=True)

    def update_real_time_graph(self):
        """更新实时曲线图"""
        if not self.real_time_graph or not self.real_time_axes:
            return

        try:
            # 获取当前时间和电流数据
            current_time = time.time() - self.start_time
            current_data = self.data_collector.get_current_data()
            
            # 添加数据到各个通道
            for i in range(8):
                self.recent_time_data[i].append(current_time)
                self.recent_channel_data[i].append(current_data[i])
                
                # 限制数据点数量，保持最新200个点
                if len(self.recent_time_data[i]) > 200:
                    self.recent_time_data[i].pop(0)
                    self.recent_channel_data[i].pop(0)
                    
                # 更新曲线数据
                self.real_time_lines[i].set_data(self.recent_time_data[i], self.recent_channel_data[i])
            
            # 自动调整坐标轴范围
            self.real_time_axes.relim()
            self.real_time_axes.autoscale_view()
            
            # 刷新画布
            self.real_time_graph.canvas.draw()
            self.real_time_graph.canvas.flush_events()
            
            # 自动处理LTP/LTD数据
            if self.auto_process_var.get():
                self.process_ltp_ltd_data()
                
        except Exception as e:
            self.append_log(f"更新实时图失败: {str(e)}")
    
    def connect_bluetooth_device(self):
        """连接蓝牙设备"""
        try:
            self.bluetooth_status_var.set("正在初始化蓝牙...")
            self.connect_bluetooth_button.config(state='disabled')
            
            # 检查蓝牙模块是否可用
            if not BLUETOOTH_AVAILABLE or get_bluetooth_collector is None:
                self.bluetooth_status_var.set("蓝牙模块不可用")
                self.append_log("蓝牙模块不可用，请安装蓝牙库以启用此功能")
                self.connect_bluetooth_button.config(state='normal')
                return
            
            # 初始化蓝牙采集器
            if self.bluetooth_collector is None:
                self.bluetooth_collector = get_bluetooth_collector(
                    log_callback=self.append_bluetooth_log,
                    data_callback=self.update_bluetooth_data_display
                )
                self.append_log("蓝牙采集器初始化成功")
            
            # 在新线程中连接设备
            def connect_thread():
                try:
                    self.bluetooth_status_var.set("正在搜索并连接设备...")
                    self.append_log("开始搜索蓝牙设备 CM1051")
                    
                    # 连接目标设备（使用从Save.py提取的默认参数）
                    success = self.bluetooth_collector.connect_target_device()
                    
                    if success:
                        self.bluetooth_status_var.set("已连接")
                        self.append_log("蓝牙设备连接成功")
                        # 启用开始采集按钮
                        self.start_bluetooth_button.config(state='normal')
                    else:
                        self.bluetooth_status_var.set("连接失败")
                        self.append_log("蓝牙设备连接失败")
                        messagebox.showwarning("连接失败", "无法找到或连接到CM1051设备，请确保设备已开启且在蓝牙范围内")
                except Exception as e:
                    self.bluetooth_status_var.set("连接错误")
                    self.append_log(f"连接过程出错: {str(e)}")
                finally:
                    self.connect_bluetooth_button.config(state='normal')
            
            threading.Thread(target=connect_thread, daemon=True).start()
            
        except Exception as e:
            self.bluetooth_status_var.set("初始化错误")
            self.append_log(f"蓝牙初始化出错: {str(e)}")
            self.connect_bluetooth_button.config(state='normal')
    
    def start_bluetooth_collection(self):
        """开始蓝牙数据采集"""
            
        # 检查蓝牙采集器是否初始化
        if self.bluetooth_collector is None:
            messagebox.showwarning("警告", "请先点击'连接设备'按钮初始化蓝牙")
            return
            
        # 检查是否已连接设备
        if not self.bluetooth_collector.is_connected():
            messagebox.showwarning("警告", "请先连接蓝牙设备")
            return
            
        # 获取目标采集点数
        target_points = 100  # 默认值
        if not self.continuous_collection_var.get():
            try:
                target_points = int(self.data_points_var.get())
                if target_points <= 0:
                    messagebox.showerror("错误", "采集数据对数必须大于0")
                    return
            except ValueError:
                messagebox.showerror("错误", "请输入有效的数字")
                return
        
        # 更新UI状态
        self.start_bluetooth_button.config(state='disabled')
        self.stop_bluetooth_button.config(state='normal')
        self.connect_bluetooth_button.config(state='disabled')
        self.collection_status_var.set("正在采集...")
        self.data_display.delete(1.0, 'end')
        
        # 清空之前的采集数据
        self.bluetooth_collector.collected_data = []
        self.bluetooth_collector.file_path = 'a.csv'  # 确保使用正确的文件名
        
        # 清空历史数据
        self.time_data = []
        self.current_data = [[] for _ in range(8)]
        self.ltp_data = []
        self.ltd_data = []
        self.ltp_ltd_time_data = []
        
        # 更新图表
        for line in self.lines:
            line.set_data([], [])
        self.canvas.draw()
        
        self.ltp_line.set_data([], [])
        self.ltd_line.set_data([], [])
        self.ltp_ltd_canvas.draw()
        
        # 开始采集
        auto_train = self.auto_train_after_collection.get()
        
        # 持续采集模式
        if self.continuous_collection_var.get():
            self.bluetooth_collector.collect_data(
                target_points=-1,  # -1表示持续采集
                auto_train=auto_train,
                app=self if auto_train else None
            )
            self.append_log("开始蓝牙持续数据采集")
        else:
            self.bluetooth_collector.collect_data(
                target_points=target_points,
                auto_train=auto_train,
                app=self if auto_train else None
            )
            self.append_log(f"开始蓝牙数据采集，目标: {target_points} 个数据")
    
    def stop_bluetooth_collection(self):
        """停止蓝牙数据采集"""
        if self.bluetooth_collector and self.bluetooth_collector.is_running():
            self.bluetooth_collector.stop_collection()
            
        # 恢复UI状态
        self.start_bluetooth_button.config(state='normal')
        self.stop_bluetooth_button.config(state='disabled')
        self.connect_bluetooth_button.config(state='normal')
        self.collection_status_var.set("已停止")
        
        # 断开蓝牙连接
        if self.bluetooth_collector and self.bluetooth_collector.is_connected():
            self.bluetooth_collector.disconnect()
            self.bluetooth_status_var.set("未连接")
        
        # 停止自动处理定时器
        self.stop_auto_process_timer()
    
    def toggle_auto_process(self):
        """切换自动处理LTP/LTD数据"""
        if self.auto_process_var.get():
            # 启动自动处理
            self.start_auto_process_timer()
            self.append_bluetooth_log("已启动LTP/LTD自动处理")
        else:
            # 停止自动处理
            self.stop_auto_process_timer()
            self.append_bluetooth_log("已停止LTP/LTD自动处理")
    
    def start_auto_process_timer(self):
        """启动自动处理定时器"""
        self.stop_auto_process_timer()
        interval = int(self.process_interval_var.get())
        self.auto_process_timer = self.master.after(interval, self.auto_process_ltp_ltd)
    
    def stop_auto_process_timer(self):
        """停止自动处理定时器"""
        if self.auto_process_timer:
            self.master.after_cancel(self.auto_process_timer)
            self.auto_process_timer = None
    
    def auto_process_ltp_ltd(self):
        """自动处理LTP/LTD数据"""
        self.process_ltp_ltd_data()
        if self.auto_process_var.get():
            self.start_auto_process_timer()
    
    def process_ltp_ltd_data(self):
        """处理LTP/LTD数据"""
        try:
            window_size = int(self.ltp_ltd_window_var.get())
            
            # 确保有数据
            if not self.time_data:
                return
                
            # 如果数据点数量少于窗口大小，只在数据量非常小时提示
            if len(self.time_data) < window_size:
                # 只在数据点非常少的时候显示信息，避免频繁提示
                if len(self.time_data) < window_size // 10:
                    self.append_bluetooth_log(f"数据正在积累中，已采集 {len(self.time_data)} 个数据点")
                return
            
            # 获取最新的window_size个数据点
            recent_time_data = self.time_data[-window_size:]
            recent_current_data = [channel[-window_size:] for channel in self.current_data if len(channel) > 0]
            
            if not recent_current_data:
                return
            
            # 计算每个通道的变化率
            ltp_values = []
            ltd_values = []
            current_rates = []
            
            for channel_data in recent_current_data:
                if len(channel_data) < 2:
                    continue
                
                # 计算相邻数据点的变化率
                for i in range(1, len(channel_data)):
                    if recent_time_data[i] != recent_time_data[i-1]:
                        rate = (channel_data[i] - channel_data[i-1]) / (recent_time_data[i] - recent_time_data[i-1])
                        current_rates.append(rate)
            
            # 分离LTP和LTD数据
            ltp_rates = [r for r in current_rates if r > 0]
            ltd_rates = [r for r in current_rates if r <= 0]
            
            # 计算LTP和LTD的平均值
            avg_ltp = sum(ltp_rates) / len(ltp_rates) if ltp_rates else 0
            avg_ltd = sum(ltd_rates) / len(ltd_rates) if ltd_rates else 0
            
            # 添加到LTP/LTD数据存储
            current_time = recent_time_data[-1]
            self.ltp_ltd_time_data.append(current_time)
            self.ltp_data.append(avg_ltp)
            self.ltd_data.append(avg_ltd)
            
            # 更新LTP/LTD曲线图
            self.update_ltp_ltd_graph()
            
            self.append_bluetooth_log(f"LTP/LTD分析完成: LTP={avg_ltp:.2e}, LTD={avg_ltd:.2e}")
            
        except Exception as e:
            self.append_bluetooth_log(f"LTP/LTD分析错误: {str(e)}")
    
    def update_ltp_ltd_graph(self):
        """更新LTP/LTD曲线图"""
        if not self.ltp_ltd_time_data:
            return
        
        # 更新曲线数据
        self.ltp_line.set_data(self.ltp_ltd_time_data, self.ltp_data)
        self.ltd_line.set_data(self.ltp_ltd_time_data, self.ltd_data)
        
        # 自动调整坐标轴范围
        self.ltp_ltd_ax.relim()
        self.ltp_ltd_ax.autoscale_view()
        
        # 更新画布
        self.ltp_ltd_canvas.draw()
        # 强制刷新画布，解决无法实时显示的问题
        self.ltp_ltd_canvas.flush_events()
        self.append_log("蓝牙设备已断开连接")
    
    def create_training_progress_page(self):
        """创建训练进度页面"""
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text='训练进度')

        # 创建多列进度显示
        progress_frame = ttk.Frame(train_frame)
        progress_frame.pack(fill='x', padx=10, pady=5)

        # 轮次进度
        epoch_frame = ttk.LabelFrame(progress_frame, text="训练轮次", padding=5)
        epoch_frame.pack(fill='x', pady=5)
        self.epoch_progress = ttk.Progressbar(epoch_frame, mode='determinate')
        self.epoch_progress.pack(fill='x')
        self.epoch_label = ttk.Label(epoch_frame, text="0/0 轮")
        self.epoch_label.pack()

        # 批次进度
        batch_frame = ttk.LabelFrame(progress_frame, text="当前轮次进度", padding=5)
        batch_frame.pack(fill='x', pady=5)
        self.batch_progress = ttk.Progressbar(batch_frame, mode='determinate')
        self.batch_progress.pack(fill='x')
        self.batch_label = ttk.Label(batch_frame, text="0/0 批次")
        self.batch_label.pack()

        # 当前准确率显示
        accuracy_frame = ttk.LabelFrame(progress_frame, text="当前准确率", padding=5)
        accuracy_frame.pack(fill='x', pady=5)
        self.current_train_acc_var = tk.StringVar(value="训练准确率: -")
        self.current_test_acc_var = tk.StringVar(value="测试准确率: -")
        ttk.Label(accuracy_frame, textvariable=self.current_train_acc_var).pack(fill='x')
        ttk.Label(accuracy_frame, textvariable=self.current_test_acc_var).pack(fill='x')

        # 实时指标显示
        metrics_frame = ttk.LabelFrame(train_frame, text="训练指标图表", padding=5)
        metrics_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 创建图表
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.figure.subplots_adjust(bottom=0.15)

        # 损失曲线
        self.ax_loss = self.figure.add_subplot(221)
        self.ax_loss.set_title('损失曲线')
        self.ax_loss.set_xlabel('迭代次数')
        self.ax_loss.set_ylabel('损失值')
        self.ax_loss.grid(True, alpha=0.3)
        self.loss_line, = self.ax_loss.plot([], [], 'b-', linewidth=2)

        # 准确率曲线
        self.ax_acc = self.figure.add_subplot(222)
        self.ax_acc.set_title('准确率曲线')
        self.ax_acc.set_xlabel('迭代次数')
        self.ax_acc.set_ylabel('准确率')
        self.ax_acc.set_ylim(0, 1)
        self.ax_acc.grid(True, alpha=0.3)
        self.acc_line, = self.ax_acc.plot([], [], 'g-', linewidth=2)
        self.test_acc_line, = self.ax_acc.plot([], [], 'r-', linewidth=2, label='测试准确率')
        self.ax_acc.legend()

        # 学习率曲线
        self.ax_lr = self.figure.add_subplot(223)
        self.ax_lr.set_title('学习率曲线')
        self.ax_lr.set_xlabel('迭代次数')
        self.ax_lr.set_ylabel('学习率')
        self.ax_lr.grid(True, alpha=0.3)
        self.lr_line, = self.ax_lr.plot([], [], 'm-', linewidth=2)

        # 训练轮次统计
        self.ax_epoch = self.figure.add_subplot(224)
        self.ax_epoch.axis('off')
        self.epoch_stats_text = self.ax_epoch.text(0.5, 0.5, '训练统计\n等待训练开始...', 
                                                 ha='center', va='center', fontsize=10)

        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.figure, master=metrics_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.ax_loss.set_title('损失值')
        self.ax_loss.set_xlabel('批次')
        self.ax_loss.set_ylabel('损失')
        self.loss_line, = self.ax_loss.plot([], [], 'b-', label='训练损失')
        self.ax_loss.legend()
        self.ax_loss.grid(True)

        # 准确率曲线
        self.ax_acc = self.figure.add_subplot(222)
        self.ax_acc.set_title('准确率')
        self.ax_acc.set_xlabel('批次')
        self.ax_acc.set_ylabel('准确率 (%)')
        self.acc_line, = self.ax_acc.plot([], [], 'g-', label='训练准确率')
        self.ax_acc.legend()
        self.ax_acc.grid(True)

        # 学习率曲线
        self.ax_lr = self.figure.add_subplot(223)
        self.ax_lr.set_title('学习率')
        self.ax_lr.set_xlabel('轮次')
        self.ax_lr.set_ylabel('学习率')
        self.lr_line, = self.ax_lr.plot([], [], 'r-')
        self.ax_lr.grid(True)

        # 创建图表画布
        self.canvas = FigureCanvasTkAgg(self.figure, master=metrics_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # 添加工具栏
        toolbar = NavigationToolbar2Tk(self.canvas, metrics_frame)
        toolbar.update()

    def update_training_progress(self, epoch, total_epochs, batch, total_batches,
                                 loss, accuracy, learning_rate):
        """更新训练进度显示"""
        # 更新轮次进度
        epoch_progress = (epoch + 1) / total_epochs * 100
        self.epoch_progress['value'] = epoch_progress
        self.epoch_label.config(text=f"{epoch + 1}/{total_epochs} 轮")

        # 更新批次进度
        batch_progress = (batch + 1) / total_batches * 100
        self.batch_progress['value'] = batch_progress
        self.batch_label.config(text=f"{batch + 1}/{total_batches} 批次")

        # 更新当前准确率显示
        train_acc = self.training_history['train_acc'][-1] if self.training_history['train_acc'] else 0
        test_acc = self.training_history['test_acc'][-1] if self.training_history['test_acc'] else 0
        self.current_train_acc_var.set(f"训练准确率: {train_acc:.2%}")
        self.current_test_acc_var.set(f"测试准确率: {test_acc:.2%}")

        # 更新损失曲线（STDP训练使用0作为损失值）
        if self.training_history['train_loss']:
            x_data = list(range(len(self.training_history['train_loss'])))
            self.loss_line.set_data(x_data, self.training_history['train_loss'])
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

        # 更新准确率曲线
        if self.training_history['train_acc']:
            x_data = list(range(len(self.training_history['train_acc'])))
            self.acc_line.set_data(x_data, self.training_history['train_acc'])
            if self.training_history['test_acc']:
                self.test_acc_line.set_data(x_data, self.training_history['test_acc'])
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()

        # 更新学习率曲线
        if self.training_history['learning_rates']:
            x_data = list(range(len(self.training_history['learning_rates'])))
            self.lr_line.set_data(x_data, self.training_history['learning_rates'])
            self.ax_lr.relim()
            self.ax_lr.autoscale_view()

        # 更新训练统计信息
        stats_text = f"训练统计:\n"
        stats_text += f"轮次: {epoch + 1}/{total_epochs}\n"
        stats_text += f"当前训练准确率: {train_acc:.2%}\n"
        stats_text += f"当前测试准确率: {test_acc:.2%}\n"
        stats_text += f"当前学习率: {learning_rate}\n"
        stats_text += f"总迭代: {len(self.training_history['train_acc'])}次"
        self.epoch_stats_text.set_text(stats_text)

        # 重绘图表
        self.canvas.draw()
        self.master.update_idletasks()

    def setup_window(self):
        self.master.title("神经突触模拟系统")
        self.master.configure(bg='#f5f5f5')
        self.master.minsize(1200, 800)

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        window_width = 1200
        window_height = 800
        center_x = int(screen_width / 2 - window_width / 2)
        center_y = int(screen_height / 2 - window_height / 2)
        self.master.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')

    def create_styles(self):
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Section.TLabelframe', padding=10)
        style.configure('Action.TButton', padding=5, font=('Arial', 10, 'bold'))

    def create_layout(self):
        main_container = ttk.Frame(self.master, padding=10)
        main_container.pack(fill='both', expand=True)

        # 标题部分
        title_frame = ttk.Frame(main_container)
        title_frame.pack(fill='x', pady=(0, 10))
        ttk.Label(title_frame, text="神经突触模拟系统",
                  style='Title.TLabel').pack(anchor='w')
        ttk.Label(title_frame, text="已预设优化参数配置，可直接开始训练或根据需要调整参数",
                  style='Subtitle.TLabel').pack(anchor='w')

        # 内容区域分割
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill='both', expand=True)

        # 左侧参数面板
        left_panel = ttk.Frame(content_frame)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 5))

        self.create_left_panel(left_panel)

        # 右侧显示面板
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side='right', fill='both', expand=True, padx=(5, 0))

        self.create_right_panel(right_panel)

    def create_right_panel(self, parent):
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True)

        # 蓝牙数据采集标签页（第一个位置）
        # 即使蓝牙模块不可用，也创建标签页但禁用相关功能
        bluetooth_frame = ttk.Frame(self.notebook)
        self.notebook.add(bluetooth_frame, text='蓝牙数据采集')
        self.create_bluetooth_data_collection_panel(bluetooth_frame)
        
        # SNN优化设置标签页（第二个位置）
        snn_frame = ttk.Frame(self.notebook)
        self.notebook.add(snn_frame, text='SNN优化设置')
        self.create_snn_optimization_panel(snn_frame)
        
        # 训练进度标签页
        self.create_training_progress_page()
        
        # 训练详细日志标签页
        train_log_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_log_frame, text='训练详细日志')
        
        # 训练日志显示
        log_frame = ttk.LabelFrame(train_log_frame, text="自动训练流程日志", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        from tkinter import scrolledtext
        self.training_log_text = scrolledtext.ScrolledText(log_frame, wrap='word', height=20)
        # 添加关键步骤的样式标签
        self.training_log_text.tag_configure("critical", foreground="red", font=('Arial', 10, 'bold'))
        self.training_log_text.pack(fill='both', expand=True)
        self.training_log_text.configure(state='disabled')
        
        # 添加控制按钮
        control_frame = ttk.Frame(train_log_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(control_frame, text="清空日志", 
                  command=lambda: self.clear_training_log()).pack(side='left', padx=5)
        ttk.Button(control_frame, text="导出日志", 
                  command=lambda: self.export_training_log()).pack(side='left', padx=5)

        # 日志输出标签页
        log_frame = ttk.Frame(self.notebook)
        self.notebook.add(log_frame, text='日志输出')
        self.log_text = tk.Text(log_frame, wrap='word', height=20)
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical',
                                  command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        self.log_text.pack(side='left', fill='both', expand=True)

        # 图像处理预览标签页
        self.image_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.image_tab, text='图像处理预览')
        self.setup_image_preview()

        # 识别结果显示标签页
        self.recognition_display = RecognitionDisplay(self.notebook, self.dataset_manager)
        self.notebook.add(self.recognition_display, text='识别结果')

        # 添加手写识别标签页
        self.handwriting_canvas = add_handwriting_recognition(self.notebook)

    def create_left_panel(self, parent):
        # 数据配置
        self.create_data_section(parent)

        # 网络参数
        network_params = [
            ("隐藏层数", "int", PresetConfig.OPTIMAL["hidden_layers"], 8),
            ("神经元数", "int", PresetConfig.OPTIMAL["neurons_per_layer"], 8),
            ("时间常数", "float", PresetConfig.OPTIMAL["tau"], 8),
            ("阈值电压", "float", PresetConfig.OPTIMAL["v_threshold"], 8),
            ("重置电压", "float", PresetConfig.OPTIMAL["v_reset"], 8),
            ("时间步长", "int", PresetConfig.OPTIMAL["time_steps"], 8)
        ]
        self.network_section = ParameterSection(parent, "网络参数",
                                              network_params, presets=True)
        self.network_section.pack(fill='x', pady=10)
        
        # 将网络参数区域与App类变量绑定
        def update_network_vars(*args):
            try:
                # 添加错误处理，避免空值问题
                hidden_layers_val = self.network_section.parameters["隐藏层数"].get()
                if hidden_layers_val is not None and str(hidden_layers_val).strip():
                    self.hidden_layers_var.set(str(hidden_layers_val))
                    
                hidden_neurons_val = self.network_section.parameters["神经元数"].get()
                if hidden_neurons_val is not None and str(hidden_neurons_val).strip():
                    self.hidden_neurons_var.set(str(hidden_neurons_val))
                    
                time_steps_val = self.network_section.parameters["时间步长"].get()
                if time_steps_val is not None and str(time_steps_val).strip():
                    self.time_steps_var.set(str(time_steps_val))
            except Exception as e:
                # 静默处理错误，避免界面崩溃
                pass
        
        # 初始化同步
        try:
            update_network_vars()
        except:
            pass
            
        # 设置参数变化监听器
        for param_name in ["隐藏层数", "神经元数", "时间步长"]:
            try:
                self.network_section.parameters[param_name].trace('w', update_network_vars)
            except:
                pass

        # 训练参数
        training_params = [
            ("训练轮次", "int", PresetConfig.OPTIMAL["epochs"], 8),
            ("批次大小", "int", PresetConfig.OPTIMAL["batch_size"], 8),
            ("学习率", "float", PresetConfig.OPTIMAL["learning_rate"], 8),
            ("训练批次", "int", PresetConfig.OPTIMAL["train_batches"], 8),
            ("测试批次", "int", PresetConfig.OPTIMAL["test_batches"], 8)
        ]
        self.training_section = ParameterSection(parent, "训练参数",
                                            training_params, presets=True)
        self.training_section.pack(fill='x', pady=10)
        
        # 将训练参数区域与App类变量绑定
        def update_training_vars(*args):
            try:
                # 添加错误处理，避免空值问题
                epochs_val = self.training_section.parameters["训练轮次"].get()
                if epochs_val is not None and str(epochs_val).strip():
                    self.epochs_var.set(str(epochs_val))
                    
                batch_size_val = self.training_section.parameters["批次大小"].get()
                if batch_size_val is not None and str(batch_size_val).strip():
                    self.batch_size_var.set(str(batch_size_val))
                    
                learning_rate_val = self.training_section.parameters["学习率"].get()
                if learning_rate_val is not None and str(learning_rate_val).strip():
                    self.learning_rate_var.set(str(learning_rate_val))
            except Exception as e:
                # 静默处理错误，避免界面崩溃
                pass
        
        # 初始化同步
        try:
            update_training_vars()
        except:
            pass
            
        # 设置参数变化监听器
        for param_name in ["训练轮次", "批次大小", "学习率"]:
            try:
                self.training_section.parameters[param_name].trace('w', update_training_vars)
            except:
                pass

        # 突触数据配置
        self.synaptic_section = SynapticDataSection(parent, log_callback=self.append_log)
        self.synaptic_section.pack(fill='x', pady=10)
        
        # 将电流数据文件路径与突触数据文件路径关联起来
        def sync_file_paths(*args):
            try:
                file_path = self.file_var.get()
                if file_path and hasattr(self.synaptic_section, 'synapse_file_var'):
                    self.synaptic_section.synapse_file_var.set(file_path)
            except Exception as e:
                # 静默处理错误，避免界面崩溃
                pass
        
        # 设置同步监听器
        try:
            self.file_var.trace('w', sync_file_paths)
            # 初始化同步
            sync_file_paths()
        except:
            pass

        # 控制按钮
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=10)

        self.start_button = ttk.Button(control_frame, text="开始训练", style='Action.TButton',
                command=self.start_training)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止",
                command=self.stop_training)
        self.stop_button.pack(side='left')
        self.stop_button.config(state='disabled')  # 初始状态为禁用

        # 进度条
        self.progress = ttk.Progressbar(parent, mode='determinate')
        self.progress.pack(fill='x', pady=5)

        # 准确度显示
        self.accuracy_var = StringVar(value="准确度: -")
        ttk.Label(parent, textvariable=self.accuracy_var).pack(pady=5)


    def create_data_section(self, parent):
        data_frame = ttk.LabelFrame(parent, text="数据配置", padding=10)
        data_frame.pack(fill='x', pady=(0, 10))

        # 数据集选择
        dataset_frame = ttk.Frame(data_frame)
        dataset_frame.pack(fill='x', pady=5)

        self.dataset_var = StringVar(value="mnist")
        ttk.Radiobutton(dataset_frame, text="MNIST", variable=self.dataset_var,
                     value="mnist").pack(side='left', padx=10)
        ttk.Radiobutton(dataset_frame, text="Fashion-MNIST", variable=self.dataset_var,
                     value="fmnist").pack(side='left', padx=10)

        # 文件选择
        file_frame = ttk.Frame(data_frame)
        file_frame.pack(fill='x', pady=5)

        ttk.Label(file_frame, text="电流数据:").pack(side='left', padx=5)
        self.file_var = StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(side='left',
                                                            fill='x', expand=True, padx=5)
        ttk.Button(file_frame, text="浏览",
                command=self.browse_file).pack(side='left')

        # 自定义数据集导入
        ttk.Button(data_frame, text="导入自定义数据集",
                command=self.import_custom_dataset).pack(fill='x', pady=5)

    def setup_image_preview(self):
        control_frame = ttk.Frame(self.image_tab)
        control_frame.pack(fill='x', pady=5)

        ttk.Label(control_frame, text="选择数据集:").pack(side='left', padx=5)
        self.preview_dataset_var = StringVar(value="mnist")
        dataset_options = ["mnist", "fmnist"]
        if "custom" in self.dataset_manager.datasets:
            dataset_options.append("custom")

        self.preview_dataset_combo = ttk.Combobox(
            control_frame,
            values=dataset_options,
            textvariable=self.preview_dataset_var,
            state='readonly'
        )
        self.preview_dataset_combo.pack(side='left', padx=5)

        ttk.Button(control_frame, text="加载样例图像",
                command=self.load_sample_images).pack(side='left', padx=5)
        ttk.Button(control_frame, text="选择并处理单张图像",
                command=self.choose_and_process_image).pack(side='left', padx=5)

        self.images_display_frame = ttk.Frame(self.image_tab)
        self.images_display_frame.pack(fill='both', expand=True, padx=5, pady=5)

    def browse_file(self):
        # 如果是手动模式，直接返回True
        if hasattr(self, 'synapse_panel') and self.synapse_panel.use_manual_data:
            return True
        
        file_path = filedialog.askopenfilename(
            title="选择电流数据文件",
            filetypes=[
                ("Excel文件", "*.xlsx *.xls"),
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.file_var.set(file_path)
            self.append_log(f"已选择电流数据文件: {file_path}")
            return True
        return False

    def import_custom_dataset(self):
        """导入自定义数据集"""
        try:
            folder_path = filedialog.askdirectory(title="选择数据集文件夹")
            if folder_path:
                # 验证数据集格式
                if self.dataset_manager.validate_custom_dataset(folder_path):
                    # 导入数据集
                    dataset_name = os.path.basename(folder_path)
                    success = self.dataset_manager.import_custom_dataset(folder_path, dataset_name)
                    
                    if success:
                        self.append_log(f"成功导入自定义数据集: {dataset_name}")
                        # 更新数据集选项
                        self.dataset_var.set(dataset_name)
                    else:
                        self.append_log("导入数据集失败")
                else:
                    messagebox.showerror("错误", "数据集格式不正确")
                    
        except Exception as e:
            self.append_log(f"导入数据集出错: {str(e)}")
            messagebox.showerror("错误", f"导入失败: {str(e)}")
    

    
    def create_snn_optimization_panel(self, parent):
        """创建SNN优化参数控制面板"""
        # SNN优化参数面板
        snn_frame = ttk.LabelFrame(parent, text="SNN识别优化设置", padding=10)
        snn_frame.pack(fill='x', pady=5)
        
        # 启用/禁用优化
        enable_frame = ttk.Frame(snn_frame)
        enable_frame.pack(fill='x', pady=2)
        ttk.Checkbutton(enable_frame, text="启用SNN优化", 
                       variable=self.snn_optimization_vars['enable_optimization'],
                       command=self.on_snn_optimization_toggle).pack(side='left')
        
        # 预设策略选择
        preset_frame = ttk.Frame(snn_frame)
        preset_frame.pack(fill='x', pady=2)
        ttk.Label(preset_frame, text="优化策略:").pack(side='left', padx=(0, 5))
        
        preset_combo = ttk.Combobox(preset_frame, 
                                   textvariable=self.snn_optimization_vars['preset_strategy'],
                                   values=['high_accuracy', 'fast_training', 'robust_generalization'],
                                   state='readonly', width=20)
        preset_combo.pack(side='left', padx=5)
        preset_combo.bind('<<ComboboxSelected>>', self.on_preset_strategy_change)
        
        ttk.Button(preset_frame, text="应用预设", 
                  command=self.apply_preset_strategy).pack(side='left', padx=5)
        
        # 参数调整区域
        params_frame = ttk.LabelFrame(snn_frame, text="高级参数", padding=5)
        params_frame.pack(fill='x', pady=5)
        
        # 动态范围增强因子
        self.create_parameter_control(params_frame, "动态范围增强:", 
                                    'dynamic_range_factor', 0.5, 5.0, 0.1)
        
        # 噪声鲁棒性
        self.create_parameter_control(params_frame, "噪声鲁棒性:", 
                                    'noise_robustness', 0.0, 0.5, 0.01)
        
        # 正则化强度
        self.create_parameter_control(params_frame, "正则化强度:", 
                                    'regularization_strength', 0.0, 0.1, 0.001)
        
        # 特征增强和自适应缩放
        feature_frame = ttk.Frame(params_frame)
        feature_frame.pack(fill='x', pady=2)
        ttk.Checkbutton(feature_frame, text="特征增强", 
                       variable=self.snn_optimization_vars['feature_enhancement']).pack(side='left', padx=10)
        ttk.Checkbutton(feature_frame, text="自适应缩放", 
                       variable=self.snn_optimization_vars['adaptive_scaling']).pack(side='left', padx=10)
        
        # 操作按钮
        button_frame = ttk.Frame(snn_frame)
        button_frame.pack(fill='x', pady=5)
        ttk.Button(button_frame, text="应用优化设置", 
                  command=self.apply_snn_optimization).pack(side='left', padx=5)
        ttk.Button(button_frame, text="重置默认值", 
                  command=self.reset_snn_optimization).pack(side='left', padx=5)
        ttk.Button(button_frame, text="查看优化效果", 
                  command=self.show_optimization_effect).pack(side='left', padx=5)
    
    def create_parameter_control(self, parent, label_text, var_name, min_val, max_val, step):
        """创建参数控制组件"""
        frame = ttk.Frame(parent)
        frame.pack(fill='x', pady=2)
        
        ttk.Label(frame, text=label_text, width=15).pack(side='left')
        
        scale = ttk.Scale(frame, from_=min_val, to=max_val, 
                         variable=self.snn_optimization_vars[var_name],
                         orient='horizontal', length=150)
        scale.pack(side='left', padx=5)
        
        value_label = ttk.Label(frame, width=8)
        value_label.pack(side='left', padx=5)
        
        def update_label(*args):
            value_label.config(text=f"{self.snn_optimization_vars[var_name].get():.3f}")
        
        self.snn_optimization_vars[var_name].trace('w', update_label)
        update_label()  # 初始化显示
    
    def on_snn_optimization_toggle(self):
        """处理SNN优化开关切换"""
        enabled = self.snn_optimization_vars['enable_optimization'].get()
        status = "已启用" if enabled else "已禁用"
        self.append_log(f"SNN优化{status}")
    
    def on_preset_strategy_change(self, event=None):
        """处理预设策略变化"""
        strategy = self.snn_optimization_vars['preset_strategy'].get()
        strategy_names = {
            'high_accuracy': '高精度模式',
            'fast_training': '快速训练模式',
            'robust_generalization': '鲁棒泛化模式'
        }
        self.append_log(f"选择优化策略: {strategy_names.get(strategy, strategy)}")
    
    def apply_preset_strategy(self):
        """应用预设策略"""
        try:
            strategy = self.snn_optimization_vars['preset_strategy'].get()
            info = self.data_processor.get_snn_optimization_info()
            
            if strategy in info['recommendations']:
                params = info['recommendations'][strategy]
                
                # 更新GUI参数
                self.snn_optimization_vars['dynamic_range_factor'].set(params.get('dynamic_range_factor', 2.0))
                self.snn_optimization_vars['noise_robustness'].set(params.get('noise_robustness', 0.1))
                self.snn_optimization_vars['regularization_strength'].set(params.get('regularization_strength', 0.01))
                
                # 应用到数据处理器
                self.data_processor.configure_snn_optimization(**params)
                
                strategy_names = {
                    'high_accuracy': '高精度模式',
                    'fast_training': '快速训练模式',
                    'robust_generalization': '鲁棒泛化模式'
                }
                self.append_log(f"已应用{strategy_names.get(strategy, strategy)}预设参数")
            else:
                self.append_log(f"未知的预设策略: {strategy}")
                
        except Exception as e:
            self.append_log(f"应用预设策略失败: {str(e)}")
            messagebox.showerror("错误", f"应用预设策略失败: {str(e)}")
    
    def apply_snn_optimization(self):
        """应用SNN优化设置"""
        try:
            enabled = self.snn_optimization_vars['enable_optimization'].get()
            
            if not enabled:
                # 当优化关闭时，将所有参数设为不生效状态
                params = {
                    'dynamic_range_factor': 1.0,
                    'noise_robustness': 0.0,
                    'regularization_strength': 0.0,
                    'feature_enhancement': False,
                    'adaptive_scaling': False,
                    'temporal_diversity': False
                }
                self.data_processor.configure_snn_optimization(**params)
                self.append_log("SNN优化已禁用，所有优化参数已设为默认值")
            else:
                # 获取当前参数
                params = {
                    'dynamic_range_factor': self.snn_optimization_vars['dynamic_range_factor'].get(),
                    'noise_robustness': self.snn_optimization_vars['noise_robustness'].get(),
                    'regularization_strength': self.snn_optimization_vars['regularization_strength'].get(),
                    'feature_enhancement': self.snn_optimization_vars['feature_enhancement'].get(),
                    'adaptive_scaling': self.snn_optimization_vars['adaptive_scaling'].get(),
                    'temporal_diversity': False  # 注意：GUI中未直接提供此参数控制
                }
                
                # 应用到数据处理器
                self.data_processor.configure_snn_optimization(**params)
                
                self.append_log("SNN优化参数已更新:")
                for key, value in params.items():
                    self.append_log(f"  {key}: {value}")
                
        except Exception as e:
            self.append_log(f"应用SNN优化设置失败: {str(e)}")
            messagebox.showerror("错误", f"应用SNN优化设置失败: {str(e)}")
    
    def reset_snn_optimization(self):
        """重置为默认值"""
        self.snn_optimization_vars['enable_optimization'].set(False)
        self.snn_optimization_vars['dynamic_range_factor'].set(1.0)
        self.snn_optimization_vars['noise_robustness'].set(0.0)
        self.snn_optimization_vars['regularization_strength'].set(0.0)
        self.snn_optimization_vars['feature_enhancement'].set(False)
        self.snn_optimization_vars['adaptive_scaling'].set(False)
        self.snn_optimization_vars['preset_strategy'].set('high_accuracy')
        
        self.apply_snn_optimization()
        self.append_log("已重置为SNN优化默认参数（优化已关闭）")
    
    def show_optimization_effect(self):
        """显示SNN优化效果"""
        try:
            # 运行优化效果演示
            import subprocess
            import sys
            
            script_path = os.path.join(os.path.dirname(__file__), 'snn_optimization_example.py')
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
                self.append_log("已启动SNN优化效果演示")
            else:
                self.append_log("优化效果演示脚本不存在")
                
        except Exception as e:
            self.append_log(f"启动优化效果演示失败: {str(e)}")
            messagebox.showerror("错误", f"启动优化效果演示失败: {str(e)}")

    def create_peak_detection_tab(self, parent):
        """创建峰值检测标签页内容"""
        # 创建滚动框架
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 标题和说明
        title_frame = ttk.Frame(scrollable_frame)
        title_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Label(title_frame, text="数据可视化", 
                 font=('Arial', 12, 'bold')).pack(anchor='w')
        ttk.Label(title_frame, text="查看和分析突触数据的峰值检测结果",
                 foreground='gray').pack(anchor='w')
        
        # 峰值检测可视化按钮
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        ttk.Button(button_frame, text="打开峰值检测窗口",
                  command=self.show_peak_detection_window,
                  style='Action.TButton').pack(fill='x', pady=5)
        
        # 说明文本
        info_frame = ttk.LabelFrame(scrollable_frame, text="功能说明", padding=10)
        info_frame.pack(fill='x', padx=10, pady=10)
        
        info_text = (
            "• 可视化原始时间-电流数据\n"
            "• 显示峰值检测结果\n"
            "• 查看LTP/LTD曲线生成过程\n"
            "• 验证数据处理的正确性"
        )
        
        ttk.Label(info_frame, text=info_text, justify='left').pack(anchor='w')
        
        # 布局滚动组件
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def append_log(self, message):
        if self.log_text:
            self.log_text.configure(state='normal')
            self.log_text.insert(tk.END, str(message) + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state='disabled')

    def update_progress(self, value):
        if self.progress:
            self.progress['value'] = value
            self.master.update_idletasks()

    def enable_start_button(self):
        for widget in self.master.winfo_children():
            if isinstance(widget, ttk.Button) and widget['text'] == "开始训练":
                widget.configure(state='normal')

    def update_recognition_result(self, result):
        self.recognition_results.append(result)
        if self.recognition_display:
            self.recognition_display.update_result(**result)

    def update_accuracy(self, accuracy):
        if self.accuracy_var:
            self.accuracy_var.set(f"准确度: {accuracy:.2f}%")

    def choose_and_process_image(self):
        img_path = filedialog.askopenfilename(
            title="选择一张图像进行处理",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if img_path and os.path.exists(img_path):
            try:
                # 读取并预处理图像
                img_pil = Image.open(img_path).convert('L')
                img_pil = img_pil.resize((28, 28))
                img_tensor = transforms.ToTensor()(img_pil)

                # 创建新窗口显示处理结果
                top = tk.Toplevel(self.master)
                top.title("单张图像处理预览")

                # 显示原图和处理后的图像
                self.show_image_processing_results(top, img_tensor)

            except Exception as e:
                messagebox.showerror("错误", f"图像处理失败: {str(e)}")

    def show_image_processing_results(self, window, img_tensor):
        # 创建显示框架
        display_frame = ttk.Frame(window)
        display_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 显示原图
        orig_frame = ttk.LabelFrame(display_frame, text="原图")
        orig_frame.pack(side='left', padx=5)
        self.display_tensor_image(img_tensor, orig_frame)

        # 显示处理后的图像
        processed_tensor = fuzzy_process_image(img_tensor)
        proc_frame = ttk.LabelFrame(display_frame, text="处理后")
        proc_frame.pack(side='left', padx=5)
        self.display_tensor_image(processed_tensor, proc_frame)

    @staticmethod
    def display_tensor_image(tensor, frame, size=(200, 200)):
        img_pil = transforms.ToPILImage()(tensor)
        img_pil = img_pil.resize(size, Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_pil)
        label = ttk.Label(frame, image=img_tk)
        label.image = img_tk
        label.pack(padx=5, pady=5)

    def start_training(self):
        """开始训练过程"""
        self.append_log("[训练流程][关键步骤] start_training方法被调用")
        
        if self.is_training:
            self.append_log("[训练流程] 当前已经在训练中，返回")
            return
            
        try:
            # 立即禁用按钮并更新状态
            self.append_log("[训练流程] 更新训练UI状态")
            self.update_training_ui(True)
            self.append_log("[训练流程] 开始初始化训练环境...")
            
            # 从参数区域获取所有网络参数
            self.append_log("[训练流程] 获取网络参数")
            hidden_layers = self.network_section.parameters["隐藏层数"].get()
            hidden_neurons = self.network_section.parameters["神经元数"].get()
            time_steps = self.network_section.parameters["时间步长"].get()
            tau = self.network_section.parameters["时间常数"].get()
            v_threshold = self.network_section.parameters["阈值电压"].get()
            v_reset = self.network_section.parameters["重置电压"].get()
            
            # 从参数区域获取所有训练参数
            self.append_log("[训练流程] 获取训练参数")
            epochs = self.training_section.parameters["训练轮次"].get()
            batch_size = self.training_section.parameters["批次大小"].get()
            learning_rate = self.training_section.parameters["学习率"].get()
            
            # 记录参数详情
            self.append_log(f"[训练流程] 网络参数: 隐藏层数={hidden_layers}, 神经元数={hidden_neurons}, 时间步长={time_steps}, 时间常数={tau}, 阈值电压={v_threshold}, 重置电压={v_reset}")
            self.append_log(f"[训练流程] 训练参数: 轮次={epochs}, 批次大小={batch_size}, 学习率={learning_rate}")
            
            # 更新App类变量以保持一致性
            self.hidden_layers_var.set(str(hidden_layers))
            self.hidden_neurons_var.set(str(hidden_neurons))
            self.time_steps_var.set(str(time_steps))
            self.epochs_var.set(str(epochs))
            self.batch_size_var.set(str(batch_size))
            self.learning_rate_var.set(str(learning_rate))
            
            # 初始化模型和优化器
            input_dim = 28 * 28  # MNIST图像大小
            output_dim = 10      # 10个类别
            
            # 创建模型（SNN类不接受tau、v_threshold、v_reset等参数）
            self.model = ScientificSNN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=hidden_layers,
                hidden_neurons=hidden_neurons,
                time_steps=time_steps
            ).to(self.device)
            
            # 手动设置各层的神经元参数
            for layer in self.model.layers:
                if hasattr(layer, 'neurons'):
                    layer.neurons.tau = tau
                    layer.neurons.v_threshold = v_threshold
                    layer.neurons.v_reset = v_reset
            self.append_log(f"已创建SNN模型: {hidden_layers}层, 每层{hidden_neurons}神经元")
            self.append_log(f"网络参数: 时间常数={tau}, 阈值电压={v_threshold}, 重置电压={v_reset}, 时间步长={time_steps}")
            
            # STDP训练不使用传统优化器
            self.append_log(f"训练参数: 轮次={epochs}, 批次大小={batch_size}, STDP学习率={learning_rate}")
            
            # 初始化训练管理器
            self.training_manager = TrainingManager(
                dataset_manager=self.dataset_manager,
                model=self.model,
                device=self.device
            )
            
            # 设置训练状态
            self.is_training = True
            
            # 初始化训练历史记录
            self.loss_history = []
            self.accuracy_history = []
            
            # 初始化训练指标历史数据结构
            self.training_history = {
                'train_loss': [],
                'train_acc': [],
                'test_acc': [],
                'learning_rates': []
            }
            
            # 启动训练线程
            self.training_thread = threading.Thread(
                target=self._run_training,
                daemon=True
            )
            self.training_thread.start()
            
            # 启动进度更新定时器
            self.update_progress_timer()
            
        except Exception as e:
            self.is_training = False
            self.update_training_ui(False)
            import traceback
            error_msg = f"训练初始化失败: {str(e)}\n{traceback.format_exc()}"
            self.append_log(error_msg)
            messagebox.showerror("初始化错误", error_msg)
            traceback.print_exc()

    def _run_training(self):
        try:
            # 获取训练参数（使用STDPTrainingConfig）
            # 从参数区域直接获取用户设置的所有值，并添加错误处理
            try:
                epochs = int(self.training_section.parameters["训练轮次"].get() or 10)
                batch_size = int(self.training_section.parameters["批次大小"].get() or 64)
                learning_rate = float(self.training_section.parameters["学习率"].get() or 0.001)
                train_batches = int(self.training_section.parameters["训练批次"].get() or 500)
                test_batches = int(self.training_section.parameters["测试批次"].get() or 100)
                
                # 网络参数
                hidden_layers = int(self.network_section.parameters["隐藏层数"].get() or 3)
                hidden_neurons = int(self.network_section.parameters["神经元数"].get() or 256)
                time_steps = int(self.network_section.parameters["时间步长"].get() or 64)
                tau = float(self.network_section.parameters["时间常数"].get() or 10.0)
                v_threshold = float(self.network_section.parameters["阈值电压"].get() or 0.5)
                v_reset = float(self.network_section.parameters["重置电压"].get() or 0.0)
            except Exception as e:
                # 使用默认参数
                self.master.after(0, lambda: self.append_log(f"警告: 参数获取失败，使用默认值: {str(e)}"))
                epochs = 10
                batch_size = 64
                learning_rate = 0.001
                train_batches = 500
                test_batches = 100
                hidden_layers = 3
                hidden_neurons = 256
                time_steps = 64
                tau = 10.0
                v_threshold = 0.5
                v_reset = 0.0
            
            # 更新所有App类变量以保持一致性
            self.epochs_var.set(str(epochs))
            self.batch_size_var.set(str(batch_size))
            self.learning_rate_var.set(str(learning_rate))
            self.hidden_layers_var.set(str(hidden_layers))
            self.hidden_neurons_var.set(str(hidden_neurons))
            self.time_steps_var.set(str(time_steps))
            
            config = STDPTrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                hidden_layers=hidden_layers,
                hidden_neurons=hidden_neurons,
                time_steps=time_steps,
                learning_rate=learning_rate,
                train_batches=train_batches,
                test_batches=test_batches,
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                # STDP特定参数
                a_plus=0.01,
                a_minus=0.01,
                tau_plus=20.0,
                tau_minus=20.0,
                weight_min=0.0,
                weight_max=1.0
            )
            
            # 记录配置信息
            self.master.after(0, lambda: self.append_log(f"训练配置: 轮次={epochs}, 批次大小={batch_size}, 训练批次={train_batches}"))
            self.master.after(0, lambda: self.append_log("使用STDP学习规则进行训练"))
            
            # 立即更新UI状态
            self.master.after(0, lambda: self.status_var.set('训练初始化中...'))
            
            # 初始化训练历史记录
            if not hasattr(self, 'loss_history'):
                self.loss_history = []
            if not hasattr(self, 'accuracy_history'):
                self.accuracy_history = []
            if not hasattr(self, 'test_accuracy_history'):
                self.test_accuracy_history = []
            
            # 创建数据加载器
            train_loader, test_loader = self.dataset_manager.get_data_loaders(
                batch_size=config.batch_size
            )
            
            # 初始化模型（确保使用修改后的SNN类）
            input_dim = 28 * 28  # MNIST图像大小（根据你的数据集调整）
            output_dim = 10      # 10个类别（根据你的数据集调整）
            self.model = ScientificSNN(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_layers=config.hidden_layers,
                hidden_neurons=config.hidden_neurons,
                time_steps=config.time_steps
            ).to(self.device)
            
            # 训练模型 - 仅使用STDP
            self.model.train()
            
            for epoch in range(config.epochs):
                # 更新UI状态
                self.master.after(0, lambda e=epoch, c=config: 
                    self.status_var.set(f'训练中: 第 {e+1}/{c.epochs} 轮'))
                
                # 检查是否需要停止训练
                if not self.is_training:
                    break
                
                total_correct = 0
                total_samples = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    # 数据移到设备（CPU/GPU）
                    data, target = data.to(self.device), target.to(self.device)
                    batch_size = data.size(0)
                    
                    # 重置模型状态（每个批次前必须重置）
                    self.model.reset_state()
                    
                    # 记录最后一个时间步的输出（用于计算准确率）
                    final_output = None
                    
                    # 时间步迭代（模拟脉冲时序）
                    for t in range(config.time_steps):
                        # 检查是否需要停止训练
                        if not self.is_training:
                            break
                        # 将输入数据编码为脉冲（关键步骤：静态图像→时序脉冲）
                        input_spikes = self.encode_input(data, t)
                        # 前向传播（STDP在forward中自动更新权重）
                        output = self.model(input_spikes, t=t, training=True)
                        final_output = output  # 保存最后一个时间步的输出
                    
                    # 计算准确率（根据最后一个时间步的输出）
                    # 检查是否在停止训练时final_output为None
                    if final_output is not None:
                        _, predicted = torch.max(final_output.float(), dim=1)
                        total_correct += (predicted == target).sum().item()
                        total_samples += batch_size
                        
                        # 计算并更新当前进度（轮次内的进度 + 轮次进度）
                        epoch_progress = epoch / config.epochs
                        batch_progress = (batch_idx + 1) / len(train_loader) / config.epochs
                        total_progress = epoch_progress + batch_progress
                        
                        # 更新进度条（通过progress_var）
                        self.progress_var.set(total_progress * 100)
                        # 直接调用update_progress方法确保进度条更新
                        self.update_progress(total_progress * 100)
                        # 立即刷新UI，确保进度条实时更新
                        self.master.update_idletasks()
                        
                        # 每100个批次更新一次日志（减少日志输出频率）
                        if batch_idx % 100 == 0:
                            # 使用当前批次累计的样本数计算准确率
                            current_acc = total_correct / total_samples if total_samples > 0 else 0
                            # 计算已处理的样本数和百分比
                            processed_samples = batch_idx * batch_size
                            dataset_total_samples = len(train_loader.dataset)
                            percentage = 100. * batch_idx / len(train_loader)
                            # 使用更清晰的日志格式
                            log_msg = f"轮次 {epoch+1}/{config.epochs} [{processed_samples}/{dataset_total_samples} 样本 ({percentage:.0f}%)] - 训练准确率: {current_acc:.2%}"
                            self.master.after(0, lambda msg=log_msg, prog=total_progress: 
                                self._log_training_progress(msg, prog))
                    else:
                        # 如果final_output为None（训练被停止），则不更新任何统计和进度
                        pass
                
                # 计算本轮训练的准确率
                train_acc = total_correct / total_samples if total_samples > 0 else 0
                self.accuracy_history.append(train_acc)
                
                # 评估模型在测试集上的表现
                test_correct = 0
                test_total = 0
                self.model.eval()
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(self.device), target.to(self.device)
                        self.model.reset_state()
                        final_output = None
                        for t in range(config.time_steps):
                            input_spikes = self.encode_input(data, t)
                            output = self.model(input_spikes, t=t, training=False)
                            final_output = output
                        _, predicted = torch.max(final_output.float(), dim=1)
                        test_correct += (predicted == target).sum().item()
                        test_total += data.size(0)
                test_acc = test_correct / test_total if test_total > 0 else 0
                self.model.train()  # 切换回训练模式
                
                # 更新UI（损失设为0，因为STDP没有传统损失）
                self.master.after(0, self._update_training_results, epoch, 0.0, train_acc, test_acc)
                
                # 记录轮次完成信息，显示训练和测试准确率
                self.master.after(0, lambda e=epoch, c=config.epochs, ta=train_acc, tea=test_acc: 
                    self._log_training_message(f"✓ 轮次 {e+1}/{c} 完成 | 训练准确率: {ta:.2%} | 测试准确率: {tea:.2%}"))
                
                # 更新进度条
                progress = (epoch + 1) / config.epochs
                self.master.after(0, lambda p=progress: self.progress_var.set(p * 100))
                
                # 更新训练进度页面数据
                self.training_history['train_loss'].append(0.0)  # STDP无传统损失
                self.training_history['train_acc'].append(train_acc)
                self.training_history['test_acc'].append(test_acc)
                # 记录学习率用于可视化，STDP配置没有learning_rate属性
                if hasattr(config, 'learning_rate'):
                    self.training_history['learning_rates'].append(config.learning_rate)
                else:
                    # 对于STDP配置，使用一个默认值或相关参数
                    self.training_history['learning_rates'].append(0.0)
                
                # 获取训练批次总数
                total_batches = len(train_loader)
                
                # 更新训练进度UI，正确处理STDP配置
                # 安全获取学习率，避免STDP配置下的属性访问错误
                lr_value = config.learning_rate if hasattr(config, 'learning_rate') else 0.0
                self.master.after(0, lambda e=epoch, te=config.epochs, b=total_batches-1, tb=total_batches, 
                                  a=train_acc, lr=lr_value: 
                    self.update_training_progress(e, te, b, tb, 0.0, a, lr))
                    
        except Exception as e:
            import traceback
            error_msg = f"训练错误: {str(e)}\n{traceback.format_exc()}"
            self.master.after(0, self._handle_training_error, error_msg)
        finally:
            self.master.after(0, self._training_finished)

    def encode_input(self, data, t):
        """优化的脉冲编码方法，结合泊松编码和时间编码优势"""
        try:
            # 确保数据类型正确
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, device=self.device, dtype=torch.float32)
            else:
                # 确保数据在正确设备上且为float32类型
                data = data.to(device=self.device, dtype=torch.float32)
            
            # 处理不同维度的输入
            original_dims = data.dim()
            if original_dims == 3:  # 单张图像 [channels, height, width]
                data = data.view(-1)  # 展平为一维
            elif original_dims == 4:  # 批次图像 [batch, channels, height, width]
                batch_size = data.size(0)
                data = data.view(batch_size, -1)  # 展平每个图像
            
            # 确保输入是2D张量 [batch, features] 或 1D张量 [features]
            if data.dim() == 1:
                data = data.unsqueeze(0)
            
            # 数据归一化到[0,1]范围
            data_min = data.min(dim=1, keepdim=True)[0]
            data_max = data.max(dim=1, keepdim=True)[0]
            data_range = data_max - data_min
            # 避免除零错误
            data_range[data_range == 0] = 1.0
            normalized_data = (data - data_min) / data_range
            
            # 应用非线性变换增强对比度
            normalized_data = normalized_data.pow(0.8)  # gamma压缩
            
            # 从网络参数区域获取时间步长，添加错误处理
            try:
                time_steps = self.network_section.parameters["时间步长"].get()
                # 确保time_steps是整数
                time_steps = int(time_steps) if time_steps is not None else 20
            except (ValueError, TypeError, KeyError):
                time_steps = 20  # 默认值
            
            # 时间编码组件
            # 计算训练进度百分比
            progress = min(1.0, t / time_steps)
            
            # 将progress转换为PyTorch张量，避免exp()函数类型错误
            progress_tensor = torch.tensor(progress, device=self.device, dtype=torch.float32)
            
            # 计算时间衰减因子
            # 使用指数衰减模型，在开始时有较高的激活率
            time_factor = torch.exp(-3.0 * progress_tensor)
            
            # 生成基于泊松分布的脉冲概率
            # 基础概率与归一化数据成正比，但添加了最小激活阈值
            min_activation = 0.02  # 最小激活概率，防止神经元完全沉默
            max_activation = 0.90   # 最大激活概率，防止过拟合
            
            # 使用with torch.no_grad()上下文管理器避免不必要的梯度计算
            with torch.no_grad():
                # 计算动态激活概率
                # 使用自适应缩放，根据当前训练进度调整
                if progress < 0.3:  # 训练初期，使用较高的基础激活率
                    probabilities = min_activation + (max_activation - min_activation) * normalized_data * time_factor * 1.5
                elif progress < 0.7:  # 训练中期，适中的基础激活率
                    probabilities = min_activation + (max_activation - min_activation) * normalized_data
                else:  # 训练后期，降低基础激活率以提高选择性
                    probabilities = min_activation + 0.7 * (max_activation - min_activation) * normalized_data
                
                # 确保概率在有效范围内
                probabilities = torch.clamp(probabilities, 0.0, 1.0)
                
                # 生成二进制脉冲
                spikes = torch.bernoulli(probabilities)
                
                # 仅在训练初期添加少量噪声，后期减少以提高稳定性
                if hasattr(self, 'model') and hasattr(self, 'is_training') and self.is_training and progress < 0.5:
                    noise_level = 0.005 * (1.0 - 2.0 * progress)  # 随训练进行减少噪声
                    noise = torch.bernoulli(torch.full_like(data, noise_level))
                    # 使用逻辑或代替按位或，避免Float类型不支持按位操作的错误
                    spikes = spikes + (noise > 0) > 0.5  # 转换为布尔操作，然后再转回浮点数
            
            # 确保返回浮点张量以兼容PyTorch操作
            return spikes.float()
            
        except Exception as e:
            # 添加错误处理机制
            print(f"encode_input错误: {str(e)}")
            # 返回默认的零张量作为安全机制
            return torch.zeros_like(data, device=self.device, dtype=torch.float32) if isinstance(data, torch.Tensor) else torch.tensor(False, device=self.device, dtype=torch.float32)
    
    def _log_training_message(self, message):
        """记录训练消息"""
        import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"
        
        # 添加到日志
        self.master.after(0, self.append_log, full_message)
        
        # 如果有专门的训练日志文本框，也添加到那里
        if hasattr(self, 'training_log_text') and self.training_log_text:
            def update_log():
                self.training_log_text.configure(state='normal')
                # 对于关键步骤的日志，使用不同的颜色显示
                if "[关键步骤]" in message:
                    self.training_log_text.insert(tk.END, full_message + "\n", "critical")
                else:
                    self.training_log_text.insert(tk.END, full_message + "\n")
                self.training_log_text.see(tk.END)  # 滚动到底部
                self.training_log_text.configure(state='disabled')
            self.master.after(0, update_log)

    def _log_training_progress(self, message, progress=None):
        """记录训练进度，确保在UI线程中执行"""
        if progress is not None:
            self.progress_var.set(progress * 100)
        self.master.after(0, self.append_log, message)

    def _update_training_results(self, epoch, loss, accuracy, test_accuracy=None):
        """更新训练结果"""
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)
        
        # 如果有测试准确率，也记录下来
        if test_accuracy is not None:
            if not hasattr(self, 'test_accuracy_history'):
                self.test_accuracy_history = []
            self.test_accuracy_history.append(test_accuracy)
            
        # 更新状态栏
        if test_accuracy is not None:
            self.status_var.set(f'训练完成: 第 {epoch+1} 轮, 损失: {loss:.4f}, 训练准确率: {accuracy:.2f}%, 测试准确率: {test_accuracy:.2f}%')
        else:
            self.status_var.set(f'训练完成: 第 {epoch+1} 轮, 损失: {loss:.4f}, 准确率: {accuracy:.2f}%')
        
        # 更新图表
        self._update_plots()
        
        # 保存模型检查点
        if hasattr(self, 'model_save_path'):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'accuracy': accuracy,
                'loss_history': self.loss_history,
                'accuracy_history': self.accuracy_history
            }
            torch.save(checkpoint, self.model_save_path)
            
        # 更新UI
        self.master.update_idletasks()

    def _update_plots(self):
        """更新训练图表"""
        try:
            # 检查图表对象是否存在
            if not hasattr(self, 'loss_line') or not hasattr(self, 'acc_line'):
                return
            
            # 更新损失曲线
            x_data = list(range(len(self.loss_history)))
            self.loss_line.set_data(x_data, self.loss_history)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

            # 更新准确率曲线
            self.acc_line.set_data(x_data, self.accuracy_history)
            
            # 如果有测试准确率历史，也绘制测试准确率曲线
            if hasattr(self, 'test_accuracy_history') and len(self.test_accuracy_history) > 0:
                if not hasattr(self, 'test_acc_line'):
                    # 创建测试准确率曲线
                    self.test_acc_line, = self.ax_acc.plot(x_data[:len(self.test_accuracy_history)], 
                                                          self.test_accuracy_history, 'r-', label='测试准确率')
                    # 添加图例
                    self.ax_acc.legend()
                else:
                    # 更新测试准确率曲线
                    self.test_acc_line.set_data(x_data[:len(self.test_accuracy_history)], 
                                              self.test_accuracy_history)
            
            self.ax_acc.relim()
            self.ax_acc.autoscale_view()

            # 重绘图表
            if hasattr(self, 'canvas'):
                self.canvas.draw()
            self.master.update_idletasks()
        except Exception as e:
            # 忽略图表更新错误，不影响训练
            pass

    def _handle_training_error(self, error_msg):
        """处理训练错误"""
        self.append_log(error_msg)
        messagebox.showerror("错误", error_msg)

    def _evaluate_model(self, test_loader):
        """评估模型性能"""
        try:
            self.model.eval()
            total_correct = 0
            total_samples = 0
            
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    batch_size = data.size(0)
                    
                    # 重置模型状态
                    self.model.reset_state()
                    final_output = None
                    
                    # 前向传播
                    for t in range(self.model.time_steps):
                        input_spikes = self.encode_input(data, t)
                        output = self.model(input_spikes, t=t, training=False)
                        final_output = output
                    
                    # 计算准确率
                    _, predicted = torch.max(final_output.float(), dim=1)
                    total_correct += (predicted == target).sum().item()
                    total_samples += batch_size
            
            self.model.train()  # 恢复训练模式
            return total_correct / total_samples if total_samples > 0 else 0
        except Exception as e:
            self.append_log(f"评估模型时出错: {str(e)}")
            return 0
    
    def _training_finished(self):
        """训练完成"""
        self.is_training = False
        self.update_training_ui(False)
        
        # 输出训练总结，使用更清晰的格式
        if hasattr(self, 'accuracy_history') and len(self.accuracy_history) > 0:
            final_train_acc = self.accuracy_history[-1] if len(self.accuracy_history) > 0 else 0
            final_test_acc = self.test_accuracy_history[-1] if hasattr(self, 'test_accuracy_history') and len(self.test_accuracy_history) > 0 else 0
            self.append_log("\n===== 训练总结 =====")
            self.append_log(f"训练轮次: {len(self.accuracy_history)}")
            self.append_log(f"  - 最终训练准确率: {final_train_acc:.4%}")
            self.append_log(f"  - 最终测试准确率: {final_test_acc:.4%}")
            self.append_log("==================\n")

    def update_training_ui(self, is_training):
        """更新训练相关UI"""
        if is_training:
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
        else:
            self.start_button.config(state='normal')
            self.stop_button.config(state='disabled')

    def update_progress_timer(self):
        """定期更新进度条和状态"""
        if self.is_training:
            # 更新进度条
            if hasattr(self, 'progress') and hasattr(self, 'progress_var'):
                self.progress['value'] = self.progress_var.get()
                self.progress.update()
                # 确保窗口重绘
                self.master.update_idletasks()
            # 继续更新
            self.master.after(100, self.update_progress_timer)

    def load_sample_images(self):
        """加载并显示样例图像"""
        for child in self.images_display_frame.winfo_children():
            child.destroy()

        dataset_name = self.preview_dataset_var.get()
        if dataset_name not in self.dataset_manager.datasets:
            self.append_log(f"数据集'{dataset_name}'不存在")
            return

        test_dataset = self.dataset_manager.datasets[dataset_name].get('test', None)
        if test_dataset is None:
            test_dataset = self.dataset_manager.datasets[dataset_name]['train']

        if len(test_dataset) < 5:
            self.append_log("数据集中样本数不足5个，无法加载样例图像。")
            return

        indices = random.sample(range(len(test_dataset)), 5)
        transform_to_pil = transforms.ToPILImage()

        for i, idx in enumerate(indices):
            img, label = test_dataset[idx]
            processed_img = fuzzy_process_image(img)

            # 原始图像处理
            img_display = (img * 0.5 + 0.5).clamp(0, 1)
            img_pil = transform_to_pil(img_display)
            processed_pil = transform_to_pil(processed_img)

            # 调整图像大小
            img_pil = img_pil.resize((100, 100), Image.LANCZOS)
            processed_pil = processed_pil.resize((100, 100), Image.LANCZOS)

            # 转换为Tkinter图像对象
            img_tk = ImageTk.PhotoImage(img_pil)
            processed_tk = ImageTk.PhotoImage(processed_pil)

            # 创建显示框架
            frame = ttk.Frame(self.images_display_frame, borderwidth=2,
                            relief='groove', padding=5)
            frame.grid(row=i // 2, column=i % 2, padx=10, pady=10, sticky='nsew')

            # 显示原始图像
            ttk.Label(frame, text="原图").pack()
            orig_img_label = ttk.Label(frame, image=img_tk)
            orig_img_label.image = img_tk
            orig_img_label.pack()

            # 显示处理后图像
            ttk.Label(frame, text="处理后图像").pack()
            processed_img_label = ttk.Label(frame, image=processed_tk)
            processed_img_label.image = processed_tk
            processed_img_label.pack()

            # 显示标签信息
            class_names = getattr(test_dataset, 'classes', None)
            if class_names:
                class_name = class_names[label]
                ttk.Label(frame, text=f"类别: {class_name}").pack()

        self.append_log("已加载并显示5张样例图像的处理结果。")

    def stop_training(self):
        """停止训练"""
        if self.is_training:
            self.append_log("正在停止训练...")
            self.is_training = False
            
            # 等待训练线程结束，但设置超时避免永久阻塞
            if hasattr(self, 'training_thread') and self.training_thread.is_alive():
                try:
                    # 不使用join，而是让线程自然退出
                    # 每100ms检查一次线程状态
                    def check_thread_status():
                        if self.training_thread.is_alive():
                            self.master.after(100, check_thread_status)
                        else:
                            self.append_log("训练已安全停止")
                            self.update_training_ui(False)
                    
                    # 设置一个超时处理
                    def force_stop():
                        if self.is_training:
                            self.append_log("⚠️  训练可能未完全停止，请检查资源占用")
                            self.update_training_ui(False)
                    
                    check_thread_status()
                    self.master.after(10000, force_stop)  # 10秒后强制更新UI状态
                except Exception as e:
                    self.append_log(f"停止训练时出错: {str(e)}")
                    self.update_training_ui(False)

    def run_training(self, current_data_path, dataset_name, network_params, training_params):
        """运行训练过程"""
        try:
            # 设置训练标志
            self.is_training = True
            self.append_log(f"开始训练: {training_params['训练轮次']}轮，{training_params['批次大小']}批次大小")
            
            # 更新网络参数区域的输入值
            for param_name, value in network_params.items():
                if param_name in self.network_section.parameters:
                    self.network_section.parameters[param_name].set(value)
            
            # 更新训练参数区域的输入值
            for param_name, value in training_params.items():
                if param_name in self.training_section.parameters:
                    self.training_section.parameters[param_name].set(value)

            # 同步更新App类变量
            self.hidden_layers_var.set(str(network_params['隐藏层数']))
            self.hidden_neurons_var.set(str(network_params['神经元数']))
            self.time_steps_var.set(str(network_params['时间步长']))
            self.epochs_var.set(str(training_params['训练轮次']))
            self.batch_size_var.set(str(training_params['批次大小']))
            self.learning_rate_var.set(str(training_params['学习率']))

            # 直接使用原始参数进行映射
            mapped_network_params = {
                'hidden_layers': network_params['隐藏层数'],
                'hidden_neurons': network_params['神经元数'],
                'tau': network_params['时间常数'],
                'v_threshold': network_params['阈值电压'],
                'v_reset': network_params['重置电压'],
                'time_steps': network_params['时间步长']
            }

            mapped_training_params = {
                'epochs': training_params['训练轮次'],
                'batch_size': training_params['批次大小'],
                'learning_rate': training_params['学习率'],
                'train_batches': training_params['训练批次'],
                'test_batches': training_params['测试批次']
            }

            # 运行单次训练，不再使用while循环避免潜在的无限循环
            if self.is_training:  # 再次检查训练标志
                accuracy = run_simulation(
                    current_data_path=current_data_path,
                    dataset_name=dataset_name,
                    dataset_manager=self.dataset_manager,
                    network_params=mapped_network_params,
                    training_params=mapped_training_params,
                    log_callback=self.append_log,
                    progress_callback=self.update_progress
                )

                if accuracy is not None:
                    self.master.after(0, lambda: self.update_accuracy(accuracy))

        except Exception as e:
            self.append_log(f"训练出错: {str(e)}")
            raise
        finally:
            self.is_training = False
            self.master.after(0, self.enable_start_button)
            if current_data_path is not None:
                try:
                    time_data, current_data, _ = load_current_time_data(current_data_path)
                    plot_input_data(time_data, current_data, filename="input_data_plot.png")
                    self.append_log("已保存输入数据图像到: input_data_plot.png")
                except Exception as e:
                    self.append_log(f"保存图像时出错: {str(e)}")

    def update_visualization(self):
        """更新可视化数据"""
        if hasattr(self, 'training_manager'):
            data = self.training_manager.get_visualization_data()
            self.recognition_display.confusion_matrix = data['confusion_matrix']
            if data['weight_matrices'] is not None:
                self.recognition_display.weight_matrices = data['weight_matrices']

    def train_model(self):
        """训练模型"""
        try:
            # 获取当前选择的数据集
            dataset_name = 'mnist'  # 默认使用MNIST数据集

            # 创建训练配置
            config = STDPTrainingConfig(
                epochs=20,
                batch_size=64,
                learning_rate=0.001,
                hidden_layers=3,
                hidden_neurons=512,
                time_steps=128
            )
            
            # 应用SNN优化设置
            if self.snn_optimization_vars['enable_optimization'].get():
                self.apply_snn_optimization()  # 确保参数已应用
                
                # 准备突触数据（如果有文件路径）
                file_path = self.file_var.get() if self.file_var.get() else None
                if file_path:
                    self.append_log("正在加载和优化突触数据...")
                    success = self.data_processor.load_data(file_path)
                    if success:
                        # 应用SNN优化的归一化
                        snn_params = {
                            'tau': config.tau,
                            'v_threshold': config.v_threshold,
                            'time_steps': config.time_steps
                        }
                        self.data_processor.normalize_data(num_points=100, snn_params=snn_params)
                        self.append_log("SNN优化的突触数据已准备就绪")
                    else:
                        self.append_log("突触数据加载失败，使用默认数据")
                else:
                    # 使用手动数据
                    self.append_log("使用默认突触数据")
                    self.data_processor.load_manual_data()
                    snn_params = {
                        'tau': config.tau,
                        'v_threshold': config.v_threshold,
                        'time_steps': config.time_steps
                    }
                    self.data_processor.normalize_data(num_points=100, snn_params=snn_params)
                    
                self.append_log("SNN优化已启用，预期提高识别精度")
            else:
                # 确保优化参数已设为不生效状态
                if hasattr(self.data_processor, 'configure_snn_optimization'):
                    self.data_processor.configure_snn_optimization(
                        dynamic_range_factor=1.0,
                        noise_robustness=0.0,
                        regularization_strength=0.0,
                        feature_enhancement=False,
                        adaptive_scaling=False,
                        temporal_diversity=False
                    )
                self.append_log("SNN优化已禁用，使用标准训练流程")

            # 开始训练
            self.append_log("开始训练...")
            self.append_log(f"使用数据集: {dataset_name}")
            self.append_log(f"网络参数: {{'隐藏层数': {config.hidden_layers}, '神经元数': {config.hidden_neurons}, "
                          f"'时间常数': {config.tau}, '阈值电压': {config.v_threshold}, "
                          f"'重置电压': {config.v_reset}, '时间步长': {config.time_steps}}}")
            self.append_log(f"训练参数: {{'训练轮次': {config.epochs}, '批次大小': {config.batch_size}, "
                          f"'学习率': {config.learning_rate}, '训练批次': {config.train_batches}, "
                          f"'测试批次': {config.test_batches}}}")

            # 执行训练
            best_acc = self.training_manager.train(
                config,
                dataset_name,
                progress_callback=self.update_progress,
                log_callback=self.append_log
            )

            # 更新可视化数据
            self.update_visualization()

            self.append_log(f"训练完成！最佳准确率: {best_acc:.2f}%")
            
            # 显示SNN优化效果统计
            if self.snn_optimization_vars['enable_optimization'].get():
                self.append_log("✨ SNN优化已应用，如果精度提高说明优化有效")

        except Exception as e:
            self.append_log(f"训练出错: {str(e)}")
            raise

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()