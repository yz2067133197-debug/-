import tkinter as tk
from tkinter import ttk
import numpy as np

class WeightVisualization:
    """权重可视化类，用于显示神经网络权重的电导值矩阵"""

    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("权重矩阵可视化")
        self.setup_ui()

    def setup_ui(self):
        self.window.geometry("800x800")

        # 控制区域
        control_frame = ttk.Frame(self.window)
        control_frame.pack(fill='x', padx=10, pady=5)

        # 图层选择
        ttk.Label(control_frame, text="选择图层:").pack(side='left', padx=5)
        self.layer_var = tk.StringVar(value="1")
        self.layer_combo = ttk.Combobox(control_frame, textvariable=self.layer_var,
                                      values=["1", "2"], state='readonly', width=5)
        self.layer_combo.pack(side='left', padx=5)
        self.layer_combo.bind('<<ComboboxSelected>>', self.on_layer_change)

        # 缩放控制
        ttk.Label(control_frame, text="缩放:").pack(side='left', padx=5)
        self.scale_var = tk.DoubleVar(value=1.0)
        scale = ttk.Scale(control_frame, from_=0.5, to=2.0,
                         variable=self.scale_var, orient='horizontal',
                         command=self.on_scale_change)
        scale.pack(side='left', padx=5, fill='x', expand=True)

        # 绘图区域
        canvas_frame = ttk.Frame(self.window)
        canvas_frame.pack(fill='both', expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(side='left', fill='both', expand=True)

        # 滚动条
        scrollbar_y = ttk.Scrollbar(canvas_frame, orient='vertical',
                                  command=self.canvas.yview)
        scrollbar_y.pack(side='right', fill='y')

        scrollbar_x = ttk.Scrollbar(self.window, orient='horizontal',
                                  command=self.canvas.xview)
        scrollbar_x.pack(fill='x', padx=10)

        self.canvas.configure(xscrollcommand=scrollbar_x.set,
                            yscrollcommand=scrollbar_y.set)

        # 色标
        self.colorbar_canvas = tk.Canvas(self.window, height=50, bg='white')
        self.colorbar_canvas.pack(fill='x', padx=10, pady=5)

        # 状态栏
        self.status_var = tk.StringVar()
        status_label = ttk.Label(self.window, textvariable=self.status_var)
        status_label.pack(pady=5)

    def visualize_weights(self, weights, layer_index=0):
        """可视化权重矩阵"""
        if weights is None:
            return

        # 清空画布
        self.canvas.delete('all')
        self.colorbar_canvas.delete('all')

        # 获取权重范围
        min_val = np.min(weights)
        max_val = np.max(weights)

        # 更新状态信息
        self.status_var.set(f"Layer {layer_index + 1} - Shape: {weights.shape}, "
                          f"Range: [{min_val:.3f}, {max_val:.3f}] Siemens")

        # 设置可视化参数
        cell_size = 40 * self.scale_var.get()
        rows, cols = weights.shape
        canvas_width = cols * cell_size
        canvas_height = rows * cell_size

        # 配置画布滚动区域
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

        # 绘制权重矩阵
        for i in range(rows):
            for j in range(cols):
                conductance = weights[i, j]
                normalized = (conductance - min_val) / (max_val - min_val)
                color = self.get_conductance_color(normalized)

                x1 = j * cell_size
                y1 = i * cell_size
                x2 = x1 + cell_size
                y2 = y1 + cell_size

                self.canvas.create_rectangle(x1, y1, x2, y2,
                                          fill=color, outline='gray')

                if cell_size >= 30:
                    self.canvas.create_text((x1 + x2) / 2, (y1 + y2) / 2,
                                          text=f"{conductance:.2e}",
                                          font=('Arial', int(cell_size / 5)),
                                          fill='white' if normalized < 0.5 else 'black')

        self.draw_colorbar(min_val, max_val)

    def draw_colorbar(self, min_val, max_val):
        """绘制色标"""
        width = self.colorbar_canvas.winfo_width()
        height = 30
        steps = 100

        for i in range(steps):
            normalized = i / (steps - 1)
            color = self.get_conductance_color(normalized)
            x1 = i * (width / steps)
            x2 = (i + 1) * (width / steps)
            self.colorbar_canvas.create_rectangle(x1, 0, x2, height,
                                              fill=color, outline='')

        self.colorbar_canvas.create_text(10, height + 10,
                                      text=f"{min_val:.2e}",
                                      anchor='w')
        self.colorbar_canvas.create_text(width - 10, height + 10,
                                      text=f"{max_val:.2e}",
                                      anchor='e')
        self.colorbar_canvas.create_text(width / 2, height + 10,
                                      text="电导值 (S)",
                                      anchor='n')

    @staticmethod
    def get_conductance_color(value):
        """获取电导值对应的颜色（蓝-白-红色方案）"""
        if value < 0.5:
            # 蓝到白
            intensity = int(255 * (value * 2))
            return f"#{intensity:02x}{intensity:02x}ff"
        else:
            # 白到红
            intensity = int(255 * (2 - value * 2))
            return f"#ff{intensity:02x}{intensity:02x}"

    def on_layer_change(self, event=None):
        """切换显示层时的回调"""
        pass

    def on_scale_change(self, event=None):
        """缩放比例改变时的回调"""
        pass

    def update(self, weights, layer_index=0):
        """更新显示的权重"""
        self.visualize_weights(weights, layer_index)

    def close(self):
        """关闭可视化窗口"""
        self.window.destroy()