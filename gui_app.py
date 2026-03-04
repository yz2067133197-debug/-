import tkinter as tk
from tkinter import ttk, filedialog, StringVar, IntVar, DoubleVar, messagebox
import threading
from PIL import Image, ImageTk
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib
# 设置全局字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import os

from gui_config import PresetConfig
from gui_components import ParameterSection, SynapticDataSection
from gui_visualization import WeightVisualization
from gui_recognition import RecognitionDisplay
from dataset_manager import DatasetManager
from utils import plot_input_data, fuzzy_process_image
from data_processing import load_current_time_data
from run_simulation import run_simulation, get_output_dir
import random
# 在gui_app.py的顶部添加导入
from handwriting_recognition import add_handwriting_recognition
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import torch
from snn import SNN
from training_manager import TrainingManager, TrainingConfig
from tkinter import messagebox
class App:
    def __init__(self, master):
        self.master = master
        self.setup_window()
        self.create_styles()
        self.dataset_manager = DatasetManager()
        # 添加训练状态标志
        self.is_training = False
        self.stop_event = threading.Event()

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
            'enable_optimization': tk.BooleanVar(value=True),
            'dynamic_range_factor': tk.DoubleVar(value=2.0),
            'noise_robustness': tk.DoubleVar(value=0.1),
            'regularization_strength': tk.DoubleVar(value=0.01),
            'feature_enhancement': tk.BooleanVar(value=True),
            'adaptive_scaling': tk.BooleanVar(value=True),
            'preset_strategy': tk.StringVar(value='high_accuracy')
        }
        
        # 初始化数据处理器
        from data_processing import SynapticDataProcessor
        self.data_processor = SynapticDataProcessor()

        # 创建布局
        self.create_layout()

        self.append_log("系统已启动，使用优化预设参数配置")
        self.append_log("MNIST数据集：训练集60,000样本，测试集10,000样本")
        self.append_log("预设批次大小：32，对应完整训练需要1875批次，完整测试需要313批次")
        self.append_log("SNN优化功能已启用，可在数据配置中调整参数")

    # 在 gui_app.py 的 App 类中添加和修改以下方法

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

        # 实时指标显示
        metrics_frame = ttk.LabelFrame(train_frame, text="训练指标", padding=5)
        metrics_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # 创建图表
        self.figure = plt.Figure(figsize=(6, 4), dpi=100)
        self.figure.subplots_adjust(bottom=0.15)

        # 损失曲线
        self.ax_loss = self.figure.add_subplot(221)
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

        # 更新损失曲线
        x_data = list(range(len(self.training_history['train_loss'])))
        self.loss_line.set_data(x_data, self.training_history['train_loss'])
        self.ax_loss.relim()
        self.ax_loss.autoscale_view()

        # 更新准确率曲线
        self.acc_line.set_data(x_data, self.training_history['train_acc'])
        self.ax_acc.relim()
        self.ax_acc.autoscale_view()

        # 更新学习率曲线
        x_data = list(range(len(self.training_history['learning_rates'])))
        self.lr_line.set_data(x_data, self.training_history['learning_rates'])
        self.ax_lr.relim()
        self.ax_lr.autoscale_view()

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
        # 设置全局ttk样式字体
        style.configure('.', font=('SimHei', 10))
        style.configure('Title.TLabel', font=('Arial', 16, 'bold'))
        style.configure('Subtitle.TLabel', font=('Arial', 10))
        style.configure('Section.TLabelframe', padding=10)
        style.configure('Action.TButton', padding=5, font=('Arial', 10, 'bold'))
        # 确保表头字体显示
        style.configure('Treeview.Heading', font=('SimHei', 10, 'bold'))

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

        # SNN优化设置标签页（第一个位置）
        snn_frame = ttk.Frame(self.notebook)
        self.notebook.add(snn_frame, text='SNN优化设置')
        self.create_snn_optimization_panel(snn_frame)

        # 训练进度标签页
        train_frame = ttk.Frame(self.notebook)
        self.notebook.add(train_frame, text='训练进度')
        self.train_canvas = tk.Canvas(train_frame, bg='white')
        self.train_canvas.pack(fill='both', expand=True, padx=5, pady=5)

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

        # 突触数据配置
        self.synaptic_section = SynapticDataSection(parent, log_callback=self.append_log)
        self.synaptic_section.pack(fill='x', pady=10)

        # 控制按钮
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=10)

        ttk.Button(control_frame, text="开始训练", style='Action.TButton',
                command=self.start_training).pack(side='left', padx=5)
        ttk.Button(control_frame, text="停止",
                command=self.stop_training).pack(side='left')

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
        # Add Fingerprint option if available
        if "fingerprint" in self.dataset_manager.datasets:
            ttk.Radiobutton(dataset_frame, text="Fingerprint", variable=self.dataset_var,
                         value="fingerprint").pack(side='left', padx=10)

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
    
    def show_peak_detection_window(self):
        """显示峰值检测数据可视化窗口"""
        try:
            show_peak_detection_window()
            self.append_log("已打开峰值检测数据可视化窗口")
        except Exception as e:
            self.append_log(f"打开峰值检测窗口失败: {str(e)}")
            messagebox.showerror("错误", f"无法打开峰值检测窗口: {str(e)}")
    
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
            if not self.snn_optimization_vars['enable_optimization'].get():
                self.append_log("SNN优化已禁用，跳过应用")
                return
            
            # 获取当前参数
            params = {
                'dynamic_range_factor': self.snn_optimization_vars['dynamic_range_factor'].get(),
                'noise_robustness': self.snn_optimization_vars['noise_robustness'].get(),
                'regularization_strength': self.snn_optimization_vars['regularization_strength'].get(),
                'feature_enhancement': self.snn_optimization_vars['feature_enhancement'].get(),
                'adaptive_scaling': self.snn_optimization_vars['adaptive_scaling'].get()
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
        self.snn_optimization_vars['dynamic_range_factor'].set(2.0)
        self.snn_optimization_vars['noise_robustness'].set(0.1)
        self.snn_optimization_vars['regularization_strength'].set(0.01)
        self.snn_optimization_vars['feature_enhancement'].set(True)
        self.snn_optimization_vars['adaptive_scaling'].set(True)
        self.snn_optimization_vars['preset_strategy'].set('high_accuracy')
        
        self.apply_snn_optimization()
        self.append_log("已重置为SNN优化默认参数")
    
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
        def _update():
            if self.log_text:
                self.log_text.configure(state='normal')
                self.log_text.insert(tk.END, str(message) + "\n")
                self.log_text.see(tk.END)
                self.log_text.configure(state='disabled')
        self.master.after(0, _update)

    def update_progress(self, value):
        def _update():
            if self.progress:
                self.progress['value'] = value
                # self.master.update_idletasks() # 不要在回调中强制刷新，让mainloop处理
        self.master.after(0, _update)

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
        # 检查是否使用手动数据
        use_manual_data = self.synaptic_section.use_manual_data
        if not use_manual_data:
            # 非手动模式才需要选择文件
            if not self.browse_file():
                return
            if not self.file_var.get():
                messagebox.showerror("错误", "请选择电流数据文件")
                return
        
        network_params = self.network_section.get_values()
        # network_params['隐藏层数'] = 1  # 恢复用户设置，不再强制
        training_params = self.training_section.get_values()
        self.stop_event.clear()

        self.append_log("\n开始训练...")
        self.append_log(f"使用数据集: {self.dataset_var.get()}")
        self.append_log(f"网络参数: {network_params}")
        self.append_log(f"训练参数: {training_params}")

        # 禁用训练按钮
        for widget in self.master.winfo_children():
            if isinstance(widget, ttk.Button) and widget['text'] == "开始训练":
                widget.configure(state='disabled')

        # 启动训练线程
        current_data_path = self.file_var.get() if not use_manual_data else None
        self.training_thread = threading.Thread(
            target=self.run_training,
            args=(current_data_path, self.dataset_var.get(), network_params, training_params)
        )
        self.training_thread.start()

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
        if hasattr(self, 'training_thread') and self.training_thread.is_alive():
            self.is_training = False
            self.stop_event.set()
            self.append_log("正在停止训练...")
            self.training_thread.join(timeout=5.0)  # 等待线程结束
            self.append_log("训练已停止")
            self.enable_start_button()

    def run_training(self, current_data_path, dataset_name, network_params, training_params):
        """运行训练过程"""
        try:
            # 设置训练标志
            self.is_training = True

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

            # 获取 synaptic_processor (如果存在且已配置)
            synaptic_processor = None
            if hasattr(self, 'synaptic_section'):
                synaptic_processor = self.synaptic_section.synaptic_processor

            # 运行训练循环
            while self.is_training:
                accuracy = run_simulation(
                    current_data_path=current_data_path,
                    dataset_name=dataset_name,
                    dataset_manager=self.dataset_manager,
                    network_params=mapped_network_params,
                    training_params=mapped_training_params,
                    log_callback=self.append_log,
                    progress_callback=self.update_progress,
                    synaptic_processor=synaptic_processor,
                    stop_event=self.stop_event
                )

                if accuracy is not None:
                    self.master.after(0, lambda: self.update_accuracy(accuracy))
                break  # 训练完成后退出循环

        except Exception as e:
            self.append_log(f"训练出错: {str(e)}")
            raise
        finally:
            self.is_training = False
            self.master.after(0, self.enable_start_button)
            if current_data_path is not None:
                try:
                    time_data, current_data, _ = load_current_time_data(current_data_path)
                    output_dir = get_output_dir()
                    input_data_path = os.path.join(output_dir, 'input_data_plot.png')
                    plot_input_data(time_data, current_data, filename=input_data_path)
                    self.append_log(f"已保存输入数据图像到: {input_data_path}")
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
            dataset_name = self.dataset_var.get()  # 使用选择的数据集

            # 创建训练配置
            config = TrainingConfig(
                epochs=20,
                batch_size=64,
                learning_rate=0.001,
                train_batches=1000,
                test_batches=200,
                hidden_layers=3,
                hidden_neurons=512,
                tau=10.0,
                v_threshold=0.5,
                v_reset=0.0,
                time_steps=128
            )
            
            # 应用SNN优化设置
            if self.snn_optimization_vars['enable_optimization'].get():
                self.apply_snn_optimization()  # 确保参数已应用
                
                # 准备突触数据
                # 优先检查 SynapticDataSection 是否已有处理好的数据
                if hasattr(self, 'synaptic_section') and \
                   self.synaptic_section.synaptic_processor.normalized_data is not None:
                    self.append_log("使用 SynapticDataSection 中已配置的突触数据")
                    # 直接使用引用，或者深拷贝以防修改
                    # 这里我们需要将处理好的 processor 赋值给 data_processor 或者直接使用它
                    # 为了保持一致性，我们更新 self.data_processor 的状态
                    self.data_processor = self.synaptic_section.synaptic_processor
                    
                    # 检查点数
                    current_points = self.data_processor.normalized_data['num_points']
                    self.append_log(f"当前突触数据采样点数: {current_points}")
                    
                else:
                    # 如果没有预处理的数据，尝试从文件加载
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
                            # 尝试获取用户设置的点数，如果没有则默认为100
                            num_points = 100
                            if hasattr(self, 'synaptic_section'):
                                try:
                                    num_points = int(self.synaptic_section.points_var.get())
                                except:
                                    pass

                            self.data_processor.normalize_data(num_points=num_points, snn_params=snn_params)
                            self.append_log(f"SNN优化的突触数据已准备就绪 (点数: {num_points})")
                        else:
                            self.append_log("突触数据加载失败，使用默认数据")
                    else:
                        # 使用手动数据
                        self.append_log("使用默认突触数据并应用SNN优化")
                        if hasattr(self.data_processor, 'load_manual_data'):
                             self.data_processor.load_manual_data()
                        snn_params = {
                            'tau': config.tau,
                            'v_threshold': config.v_threshold,
                            'time_steps': config.time_steps
                        }
                        self.data_processor.normalize_data(num_points=100, snn_params=snn_params)
                    
                self.append_log("SNN优化已启用，预期提高识别精度")
            else:
                self.append_log("SNN优化已禁用，使用标准训练流程")

            # 确保在 TrainingManager 中使用处理后的数据
            if self.data_processor.normalized_data is not None and self.snn_optimization_vars['enable_optimization'].get():
                # 获取LTP/LTD数据
                ltp_data = self.data_processor.normalized_data['ltp']
                ltd_data = self.data_processor.normalized_data['ltd']
                
                # 转换为张量
                if not isinstance(ltp_data, torch.Tensor):
                    ltp_data = torch.tensor(ltp_data, dtype=torch.float32)
                if not isinstance(ltd_data, torch.Tensor):
                    ltd_data = torch.tensor(ltd_data, dtype=torch.float32)
                
                # 设置到 training_manager
                self.training_manager.set_synaptic_data(ltp_data, ltd_data)
                self.append_log("已将优化后的LTP/LTD数据加载到训练管理器")

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
