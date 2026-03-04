import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from scipy.signal import find_peaks
from data_processing import load_current_time_data

class PeakDetectionWindow(tk.Toplevel):
    def __init__(self, parent, data_processor, file_path=None):
        super().__init__(parent)
        self.parent_section = parent
        self.title("峰值检测与参数调整")
        self.geometry("1000x800")
        
        self.data_processor = data_processor
        self.file_path = file_path
        self.peak_data = None
        
        # 如果没有提供文件路径，尝试从processor获取
        if not self.file_path and hasattr(data_processor, 'file_path'):
            self.file_path = data_processor.file_path
            
        self.setup_ui()
        
        # 如果有文件路径，自动加载
        if self.file_path:
            self.load_and_plot_data()
            
    def setup_ui(self):
        # 顶部控制面板
        control_frame = ttk.LabelFrame(self, text="检测参数", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        # 参数输入
        ttk.Label(control_frame, text="最小高度 (Height):").pack(side='left', padx=5)
        self.min_height_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.min_height_var, width=8).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="最小间距 (Distance):").pack(side='left', padx=5)
        self.min_distance_var = tk.IntVar(value=10)
        ttk.Entry(control_frame, textvariable=self.min_distance_var, width=8).pack(side='left', padx=5)
        
        ttk.Label(control_frame, text="相对高度 (Prominence):").pack(side='left', padx=5)
        self.prominence_var = tk.DoubleVar(value=0.0)
        ttk.Entry(control_frame, textvariable=self.prominence_var, width=8).pack(side='left', padx=5)
        
        # 按钮
        ttk.Button(control_frame, text="更新检测", command=self.update_detection).pack(side='left', padx=15)
        ttk.Button(control_frame, text="应用并关闭", command=self.apply_and_close).pack(side='left', padx=5)
        
        # 信息显示
        self.info_label = ttk.Label(control_frame, text="就绪")
        self.info_label.pack(side='right', padx=10)
        
        # 绘图区域
        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.figure = plt.Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
    def load_and_plot_data(self):
        try:
            if not self.file_path:
                return
                
            time_final, ai_final, _ = load_current_time_data(self.file_path)
            
            self.peak_data = {
                'time': time_final,
                'current': ai_final
            }
            
            # 初始检测
            self.update_detection()
            
        except Exception as e:
            messagebox.showerror("加载错误", str(e))
            
    def update_detection(self):
        if self.peak_data is None:
            return
            
        try:
            current = self.peak_data['current']
            time = self.peak_data['time']
            
            height = self.min_height_var.get()
            distance = self.min_distance_var.get()
            prominence = self.prominence_var.get()
            
            # 只有当参数大于0时才使用
            kwargs = {}
            if height > 0: kwargs['height'] = height
            if distance > 0: kwargs['distance'] = distance
            if prominence > 0: kwargs['prominence'] = prominence
            
            peaks, properties = find_peaks(current, **kwargs)
            
            # 绘图
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            
            ax.plot(time, current, 'b-', label='原始电流', alpha=0.7)
            ax.plot(time[peaks], current[peaks], 'rx', label=f'检测峰值 ({len(peaks)})', markersize=10)
            
            # 标记阈值线
            if height > 0:
                ax.axhline(y=height, color='g', linestyle='--', alpha=0.5, label=f'最小高度={height}')
                
            ax.set_title(f"峰值检测结果 (共 {len(peaks)} 个峰值)")
            ax.set_xlabel("时间")
            ax.set_ylabel("电流")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.canvas.draw()
            
            self.info_label.config(text=f"检测到 {len(peaks)} 个峰值")
            
            # 临时保存结果以便应用
            self.current_peaks = peaks
            
        except Exception as e:
            messagebox.showerror("检测错误", str(e))
            
    def apply_and_close(self):
        """将结果应用到主处理器并关闭"""
        if hasattr(self, 'current_peaks'):
            # 更新data_processor中的峰值计数
            self.data_processor.peak_count = len(self.current_peaks)
            
            # 使用保存的 parent_section 引用更新参数
            target = self.parent_section
            
            # 更新参数标志
            updated = False
            
            if hasattr(target, 'min_height_var'):
                target.min_height_var.set(self.min_height_var.get())
                updated = True
            if hasattr(target, 'min_distance_var'):
                target.min_distance_var.set(self.min_distance_var.get())
                updated = True
            if hasattr(target, 'prominence_var'):
                target.prominence_var.set(self.prominence_var.get())
                updated = True
            
            if updated:
                msg = f"已应用参数！\n检测到 {len(self.current_peaks)} 个峰值。\n"
                msg += f"设置参数: H={self.min_height_var.get()}, D={self.min_distance_var.get()}, P={self.prominence_var.get()}\n"
                msg += "请在主界面点击'处理数据'以应用新结果。"
                messagebox.showinfo("成功", msg)
            else:
                messagebox.showwarning("警告", "无法更新主界面参数，请检查连接。")
                
            self.destroy()
        else:
            self.destroy()

def show_peak_detection_window(parent, data_processor, file_path=None):
    window = PeakDetectionWindow(parent, data_processor, file_path)
    # window.grab_set() # 模态对话框
    return window
