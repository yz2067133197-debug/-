import tkinter as tk
from tkinter import ttk, filedialog
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch.nn.functional as F
from snn import SNN
import tkinter as tk
from tkinter import ttk, filedialog, StringVar, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class HandwritingCanvas:
    def __init__(self, parent, size=280):
        self.parent = parent
        self.size = size
        self.setup_ui()
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prediction_history = []  # 添加预测历史记录
        self.preview_window = None  # 预览窗口

    def setup_ui(self):
        """设置UI界面"""
        # 主框架
        self.frame = ttk.LabelFrame(self.parent, text="手写数字识别", padding=10)
        self.frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 创建左右分隔的框架
        self.paned_window = ttk.PanedWindow(self.frame, orient='horizontal')
        self.paned_window.pack(fill='both', expand=True)

        # 左侧绘图区域
        self.left_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.left_frame)

        # 创建画布
        self.canvas = tk.Canvas(self.left_frame,
                                width=self.size,
                                height=self.size,
                                bg='black')
        self.canvas.pack(pady=5)

        # 绘制网格线
        self.draw_grid()

        # 右侧控制和显示区域
        self.right_frame = ttk.Frame(self.paned_window)
        self.paned_window.add(self.right_frame)

        # 控制按钮区域
        control_frame = ttk.LabelFrame(self.right_frame, text="控制面板", padding=5)
        control_frame.pack(fill='x', padx=5, pady=5)

        # 按钮组
        ttk.Button(control_frame, text="加载模型",
                   command=self.load_model).pack(fill='x', pady=2)

        self.predict_btn = ttk.Button(control_frame, text="识别",
                                      command=self.predict,
                                      state='disabled')
        self.predict_btn.pack(fill='x', pady=2)

        ttk.Button(control_frame, text="清除",
                   command=self.clear_canvas).pack(fill='x', pady=2)

        ttk.Button(control_frame, text="保存图像",
                   command=self.save_canvas).pack(fill='x', pady=2)

        # 添加预处理预览按钮
        ttk.Button(control_frame, text="预览预处理",
                  command=self.show_preprocessing_preview).pack(fill='x', pady=2)

        # 预测结果显示区域
        result_frame = ttk.LabelFrame(self.right_frame, text="识别结果", padding=5)
        result_frame.pack(fill='x', padx=5, pady=5)

        self.result_var = tk.StringVar(value="-")
        ttk.Label(result_frame,
                  textvariable=self.result_var,
                  font=('Arial', 24, 'bold')).pack()

        # 预测历史区域
        history_frame = ttk.LabelFrame(self.right_frame, text="识别历史", padding=5)
        history_frame.pack(fill='both', expand=True, padx=5, pady=5)

        # 添加历史记录表格
        self.history_tree = ttk.Treeview(history_frame,
                                         columns=('time', 'prediction', 'confidence'),
                                         show='headings',
                                         height=10)

        self.history_tree.heading('time', text='时间')
        self.history_tree.heading('prediction', text='预测结果')
        self.history_tree.heading('confidence', text='置信度')

        self.history_tree.column('time', width=100)
        self.history_tree.column('prediction', width=70)
        self.history_tree.column('confidence', width=70)

        scrollbar = ttk.Scrollbar(history_frame,
                                  orient='vertical',
                                  command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)

        self.history_tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # 绑定鼠标事件
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.reset_point)
        self.last_x = None
        self.last_y = None

        # 创建图像缓冲
        self.image = Image.new('L', (self.size, self.size), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def draw_grid(self):
        """绘制网格线"""
        for i in range(0, self.size + 1, 28):
            self.canvas.create_line([(i, 0), (i, self.size)],
                                    fill='grey20', dash=(2, 2))
            self.canvas.create_line([(0, i), (self.size, i)],
                                    fill='grey20', dash=(2, 2))

    def paint(self, event):
        """绘画处理"""
        if self.last_x and self.last_y:
            x = event.x
            y = event.y
            # 在画布上绘制
            self.canvas.create_line((self.last_x, self.last_y, x, y),
                                    fill='white', width=20, capstyle=tk.ROUND,
                                    smooth=True)
            # 在图像缓冲中绘制
            self.draw.line((self.last_x, self.last_y, x, y),
                           fill='white', width=20)
        self.last_x = event.x
        self.last_y = event.y

    def reset_point(self, event):
        """重置绘画点"""
        self.last_x = None
        self.last_y = None

    def clear_canvas(self):
        """清除画布"""
        self.canvas.delete('all')
        self.image = Image.new('L', (self.size, self.size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.draw_grid()
        self.result_var.set("-")

    def save_canvas(self):
        """保存画布内容"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"),
                       ("All files", "*.*")],
            title="保存手写数字图像"
        )
        if file_path:
            # 保存原始大小的图像
            self.image.save(file_path)
            # 同时保存缩放后的28x28版本
            scaled_path = file_path.replace('.png', '_28x28.png')
            scaled_image = self.image.resize((28, 28), Image.LANCZOS)
            scaled_image.save(scaled_path)
            messagebox.showinfo("成功",
                                f"图像已保存:\n原始图像: {file_path}\n缩放图像: {scaled_path}")

    def load_model(self):
        """加载模型"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择模型文件",
                filetypes=[("PyTorch模型", "*.pth"),
                           ("所有文件", "*.*")]
            )
            if file_path:
                checkpoint = torch.load(file_path, map_location=self.device)
                
                # 检查模型状态字典
                if 'model_state_dict' in checkpoint:
                    model_state_dict = checkpoint['model_state_dict']
                elif 'model' in checkpoint:
                    model_state_dict = checkpoint['model']
                else:
                    raise ValueError("模型文件中找不到模型状态字典")
                
                # 检测模型是否使用了突触数据功能
                use_synaptic_data = any('synaptic_projection' in key for key in model_state_dict.keys())
                
                # 获取突触数据维度
                synaptic_data_dim = 100  # 默认值
                if use_synaptic_data:
                    for key in model_state_dict.keys():
                        if 'synaptic_projection.weight' in key:
                            synaptic_data_dim = model_state_dict[key].shape[1]
                            break
                    print(f"检测到突触数据功能，维度: {synaptic_data_dim}")
                
                # 尝试从checkpoint中获取network_params
                if 'network_params' in checkpoint:
                    network_params = checkpoint['network_params']
                    
                    # 从保存的参数创建模型
                    self.model = SNN(
                        input_dim=28 * 28,
                        output_dim=10,
                        hidden_layers=network_params['hidden_layers'],
                        hidden_neurons=network_params['hidden_neurons'],
                        tau=network_params['tau'],
                        v_threshold=network_params['v_threshold'],
                        v_reset=network_params['v_reset'],
                        time_steps=network_params['time_steps'],
                        use_synaptic_data=use_synaptic_data,
                        synaptic_data_dim=synaptic_data_dim if use_synaptic_data else 100
                    ).to(self.device)
                else:
                    # 如果没有network_params，尝试从模型状态字典中推断架构
                    # 查找layers的数量（排除输出层）
                    layers_count = 0
                    hidden_neurons = 100  # 默认值
                    
                    for key in model_state_dict.keys():
                        if key.startswith('layers.') and '.fc.weight' in key:
                            parts = key.split('.')
                            layer_idx = int(parts[1])
                            layers_count = max(layers_count, layer_idx + 1)
                            
                            # 获取隐藏层的神经元数量
                            weight_shape = model_state_dict[key].shape
                            if len(weight_shape) == 2:
                                hidden_neurons = weight_shape[0]
                    
                    # 减去输出层，得到隐藏层数量
                    hidden_layers = max(1, layers_count - 1)
                    
                    # 创建推断出的模型
                    self.model = SNN(
                        input_dim=28 * 28,
                        output_dim=10,
                        hidden_layers=hidden_layers,
                        hidden_neurons=hidden_neurons,
                        use_synaptic_data=use_synaptic_data,
                        synaptic_data_dim=synaptic_data_dim if use_synaptic_data else 100
                    ).to(self.device)
                    
                    print(f"推断出的网络架构: 隐藏层数={hidden_layers}, 神经元数={hidden_neurons}")
                    if use_synaptic_data:
                        print(f"启用突触数据功能，维度: {synaptic_data_dim}")
                
                # 加载模型状态字典
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'model' in checkpoint:
                    self.model.load_state_dict(checkpoint['model'])
                
                self.model.eval()
                self.predict_btn['state'] = 'normal'
                
                # 获取准确率
                accuracy = checkpoint.get('test_acc', checkpoint.get('accuracy', 0))
                
                messagebox.showinfo("成功",
                                    f"模型加载成功！\n准确率: {accuracy:.2f}%")
        except Exception as e:
            messagebox.showerror("错误", f"加载模型失败: {str(e)}")

    def center_image(self, img_array):
        """将图像中的数字居中处理"""
        # 计算非零像素的重心
        y_indices, x_indices = np.where(img_array > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            return img_array  # 如果没有找到非零像素，直接返回原图

        x_center = int(np.mean(x_indices))
        y_center = int(np.mean(y_indices))

        # 计算需要移动的距离（目标中心是(14,14)）
        shift_x = 14 - x_center
        shift_y = 14 - y_center

        # 创建新图像
        new_img = np.zeros_like(img_array)

        # 应用平移
        for y in range(28):
            for x in range(28):
                new_x = x + shift_x
                new_y = y + shift_y
                if 0 <= new_x < 28 and 0 <= new_y < 28:
                    new_img[new_y, new_x] = img_array[y, x]

        return new_img

    def preprocess_image(self):
        """预处理图像并返回各步骤结果
        
        处理步骤:
        1. 获取原始图像
        2. 缩放至28x28
        3. 中心化处理
        4. 转换为张量并归一化到[-1, 1]范围 (与训练时一致)
        """
        if not hasattr(self, 'image'):
            messagebox.showwarning("提示", "请先绘制数字")
            return None

        # 1. 获取原始图像
        original = self.image.copy()

        # 2. 缩放图像
        resized = self.image.resize((28, 28), Image.LANCZOS)

        # 3. 转换为numpy数组
        img_array = np.array(resized, dtype=np.float32)

        # 4. 中心化处理
        centered = self.center_image(img_array)
        
        # 5. 转换为张量并归一化到[0,1]
        tensor = torch.FloatTensor(centered).unsqueeze(0) / 255.0
        
        # 6. 归一化到[-1, 1]范围 (与训练时一致)
        tensor = (tensor - 0.5) / 0.5
        
        # 7. 调整维度 [1, 1, 28, 28]
        tensor = tensor.reshape(1, 1, 28, 28).to(self.device)
        
        # 用于可视化的归一化版本 [0,1]范围
        normalized_for_display = centered / 255.0

        return {
            'original': original,
            'resized': resized,
            'centered': centered,
            'normalized': normalized_for_display,  # 用于显示
            'tensor': tensor  # 用于预测
        }

    def show_preprocessing_preview(self):
        """显示预处理步骤预览"""
        # 如果预览窗口已存在，则销毁重建
        if hasattr(self, 'preview_window') and self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except:
                pass
            self.preview_window = None

        # 预处理图像
        processed = self.preprocess_image()
        if processed is None:
            return

        # 创建预览窗口
        self.preview_window = tk.Toplevel(self.parent)
        self.preview_window.title("预处理步骤预览")
        self.preview_window.geometry("1000x800")

        # 创建主框架
        main_frame = ttk.Frame(self.preview_window)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # 创建步骤框架
        steps_frame = ttk.LabelFrame(main_frame, text="预处理步骤", padding=10)
        steps_frame.pack(fill='both', expand=True, pady=5)

        # 创建图形
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('图像预处理步骤', fontsize=16)

        # 显示原始图像
        axs[0, 0].imshow(processed['original'], cmap='gray')
        axs[0, 0].set_title('1. 原始图像')
        axs[0, 0].axis('off')

        # 显示缩放后图像
        axs[0, 1].imshow(processed['resized'], cmap='gray')
        axs[0, 1].set_title('2. 缩放至28x28')
        axs[0, 1].axis('off')

        # 显示中心化后图像
        axs[1, 0].imshow(processed['centered'], cmap='gray')
        axs[1, 0].set_title('3. 中心化处理')
        axs[1, 0].axis('off')

        # 显示归一化后图像
        # 注意：实际用于预测的张量是[-1,1]范围的，这里显示的是[0,1]范围用于可视化
        axs[1, 1].imshow(processed['normalized'], cmap='gray')
        axs[1, 1].set_title('4. 归一化[0,1] (显示用)')
        axs[1, 1].axis('off')
        
        # 添加张量范围信息
        tensor_min = processed['tensor'].min().item()
        tensor_max = processed['tensor'].max().item()
        tensor_mean = processed['tensor'].mean().item()
        tensor_std = processed['tensor'].std().item()
        
        info_text = (f'张量信息:\n'
                   f'范围: [{tensor_min:.2f}, {tensor_max:.2f}]\n'
                   f'均值: {tensor_mean:.4f}, 标准差: {tensor_std:.4f}')
        
        fig.text(0.5, 0.02, info_text, 
                ha='center', fontsize=10, 
                bbox=dict(facecolor='lightgray', alpha=0.5))

        # 调整子图间距
        plt.tight_layout()

        # 在Tkinter中显示matplotlib图形
        canvas = FigureCanvasTkAgg(fig, master=steps_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # 添加关闭按钮
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill='x', pady=10)

        ttk.Button(btn_frame, text="关闭预览",
                  command=self.preview_window.destroy).pack(side='right', padx=5)

        # 添加预测按钮
        if self.model is not None:
            ttk.Button(btn_frame, text="使用此预处理结果进行预测",
                      command=lambda: self.predict(processed['tensor'])).pack(side='right', padx=5)

    def predict(self, preprocessed_tensor=None):
        """进行预测

        Args:
            preprocessed_tensor: 可选的预处理的张量，如果为None则重新处理
        """
        if self.model is None:
            messagebox.showerror("错误", "请先加载模型！")
            return

        try:
            # 如果没有提供预处理的张量，则进行完整处理
            if preprocessed_tensor is None:
                processed = self.preprocess_image()
                if processed is None:
                    return
                tensor = processed['tensor']
            else:
                tensor = preprocessed_tensor

            # 进行预测
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = F.softmax(output, dim=1)
                predicted = output.argmax(dim=1).item()
                confidence = probabilities[0][predicted].item() * 100

            # 更新显示结果
            self.result_var.set(f"{predicted}\n{confidence:.1f}%")

            # 添加到历史记录
            timestamp = datetime.now().strftime('%H:%M:%S')
            self.history_tree.insert('', 0, values=(timestamp,
                                                    str(predicted),
                                                    f"{confidence:.1f}%"))

            # 保存预测记录
            self.prediction_history.append({
                'time': timestamp,
                'prediction': predicted,
                'confidence': confidence,
                'image_tensor': tensor.cpu()
            })

        except Exception as e:
            messagebox.showerror("错误", f"预测失败: {str(e)}")



def add_handwriting_recognition(notebook):
    """添加手写识别标签页到notebook"""
    recognition_frame = ttk.Frame(notebook)
    notebook.add(recognition_frame, text='手写识别')

    # 创建手写识别画布
    canvas = HandwritingCanvas(recognition_frame)
    return canvas