# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline, interp1d
import torch


def load_current_time_data(excel_path):
    try:
        # 读取Excel文件，设置第一行为表头
        data = pd.read_excel(excel_path)

        # 检查数据列数
        if data.shape[1] < 2:
            raise ValueError("数据列数不足，无法提取前两列数据。")

        # 获取列名
        column_names = data.columns.tolist()
        print(f"列名: {column_names}")  # 打印列名以便调试

        # 获取数值数据
        time_raw = data[column_names[0]].values  # 使用第一列的列名
        ai_raw = data[column_names[1]].values  # 使用第二列的列名

        # 清理nan
        mask = ~np.isnan(time_raw) & ~np.isnan(ai_raw)
        time_clean = time_raw[mask]
        ai_clean = ai_raw[mask]

        # 排序
        sort_idx = np.argsort(time_clean)
        time_sorted = time_clean[sort_idx]
        ai_sorted = ai_clean[sort_idx]

        # 去重
        unique_time, unique_ind = np.unique(time_sorted, return_index=True)
        if len(unique_time) < len(time_sorted):
            from collections import defaultdict
            ai_dict = defaultdict(list)
            for t, a in zip(time_sorted, ai_sorted):
                ai_dict[t].append(a)
            time_final = []
            ai_final = []
            for t_key in sorted(ai_dict.keys()):
                time_final.append(t_key)
                ai_final.append(np.mean(ai_dict[t_key]))
            time_final = np.array(time_final)
            ai_final = np.array(ai_final)
        else:
            time_final = unique_time
            ai_final = ai_sorted[unique_ind]

        # 找脉冲峰值个数
        peaks, _ = find_peaks(ai_final, height=0)
        peak_count = len(peaks)

        print(f"数据处理完成：")
        print(f"- 数据点数：{len(time_final)}")
        print(f"- 峰值数：{peak_count}")

        return time_final, ai_final, peak_count

    except Exception as e:
        print(f"数据处理错误: {str(e)}")
        raise ValueError(f"数据加载失败: {str(e)}")

# -*- coding: utf-8 -*-

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os


class SynapticDataProcessor:
    # 默认手动输入数据
    DEFAULT_LTP_DATA = [
        1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    ]
    
    DEFAULT_LTD_DATA = [
    15,14,13,12,11,10,9,8,7,6,5,4,3,2,1
    ] 

    def __init__(self):
        self.ltp_data = None
        self.ltd_data = None
        self.normalized_data = None
        self.raw_data = None
        self.peak_count = None
        self.use_manual_data = False
        
        # SNN优化参数
        self.snn_optimization_params = {
            'dynamic_range_factor': 2.0,      # 动态范围增强因子
            'temporal_diversity': True,        # 时间多样性
            'noise_robustness': 0.1,          # 噪声鲁棒性
            'feature_enhancement': True,       # 特征增强
            'adaptive_scaling': True,          # 自适应缩放
            'regularization_strength': 0.01   # 正则化强度
        }

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
    def generate_ltp_curve(self, pulse_norm, ltp_values):
        """生成平滑的LTP曲线，使用sigmoid-like形状"""
        x = pulse_norm
        base_curve = 1 / (1 + np.exp(-10 * (x - 0.5)))
        min_val, max_val = np.min(ltp_values), np.max(ltp_values)
        return min_val + base_curve * (max_val - min_val)
    
    def generate_ltd_curve(self, pulse_norm, ltp_values):
        """生成平滑的LTD曲线，使用指数衰减形状"""
        x = pulse_norm
        decay_curve = np.exp(-2 * x)
        noise = np.random.normal(0, 0.01, len(x))
        decay_curve = np.clip(decay_curve + noise, 0, 1)
        min_val, max_val = np.min(ltp_values) * 0.8, np.max(ltp_values) * 0.8
        return min_val + decay_curve * (max_val - min_val)
        
    def smooth_data(self, data, window_size=5):
        """对数据进行平滑处理，带边缘修正"""
        if len(data) < window_size:
            return data
        
        # 构造平均窗口
        window = np.ones(window_size) / window_size
        
        # 使用 same 模式卷积
        smoothed = np.convolve(data, window, mode='same')
        
        # 边缘修正：用原始数据填充受边界效应影响的区域
        # 卷积会导致两端数值因“零填充”或“部分重叠”而失真
        # 我们直接恢复两端的原始值
        half_window = window_size // 2
        smoothed[:half_window] = data[:half_window]
        smoothed[-half_window:] = data[-half_window:]
        
        return smoothed
    
    def enhance_dynamic_range(self, data, factor=2.0):
        """增强数据的动态范围，提高SNN的区分能力"""
        # 使用非线性变换增强对比度
        data_norm = (data - data.min()) / (data.max() - data.min())
        # 幂函数变换增强对比度
        enhanced = np.power(data_norm, 1.0 / factor)
        # 重新缩放到原范围
        return data.min() + enhanced * (data.max() - data.min())
    
    def add_temporal_diversity(self, time_data, current_data):
        """添加时间多样性，生成多尺度特征"""
        # 生成不同时间尺度的特征
        scales = [0.5, 1.0, 2.0]  # 不同的时间尺度
        diverse_features = []
        
        for scale in scales:
            # 时间缩放
            scaled_time = time_data * scale
            # 使用不同的平滑窗口
            window_size = max(3, int(len(current_data) * scale / 10))
            smoothed = self.smooth_data(current_data, window_size)
            diverse_features.append((scaled_time, smoothed))
        
        return diverse_features
    
    def apply_noise_robustness(self, data, noise_level=0.1):
        """增强噪声鲁棒性，防止过拟合"""
        # 添加适度的高斯噪声
        noise = np.random.normal(0, noise_level * np.std(data), len(data))
        robust_data = data + noise
        
        # 使用中值滤波器去除异常值
        from scipy.signal import medfilt
        if len(robust_data) > 5:
            robust_data = medfilt(robust_data, kernel_size=5)
        
        return robust_data
    
    def adaptive_feature_scaling(self, ltp_data, ltd_data):
        """自适应特征缩放，根据数据特性动态调整"""
        # 计算数据的统计特征
        ltp_std = np.std(ltp_data)
        ltd_std = np.std(ltd_data)
        ltp_range = np.ptp(ltp_data)  # peak-to-peak
        ltd_range = np.ptp(ltd_data)
        
        # 自适应缩放因子
        ltp_scale = 1.0 + (ltp_std / ltp_range) if ltp_range > 0 else 1.0
        ltd_scale = 1.0 + (ltd_std / ltd_range) if ltd_range > 0 else 1.0
        
        # 应用缩放
        ltp_scaled = ltp_data * ltp_scale
        ltd_scaled = ltd_data * ltd_scale
        
        return ltp_scaled, ltd_scaled
    
    def apply_regularization(self, data, strength=0.01):
        """应用正则化，防止过拟合"""
        # L2正则化：对数据进行微小的平滑处理
        regularized = data.copy()
        for i in range(1, len(data) - 1):
            regularized[i] = (1 - strength) * data[i] + \
                           strength * 0.5 * (data[i-1] + data[i+1])
        return regularized
    
    def optimize_for_snn(self, ltp_data, ltd_data, tau=20.0, v_threshold=1.0):
        """为SNN优化LTP/LTD数据，提高识别精度"""
        params = self.snn_optimization_params
        
        # 1. 动态范围增强
        if params['feature_enhancement']:
            ltp_enhanced = self.enhance_dynamic_range(ltp_data, params['dynamic_range_factor'])
            ltd_enhanced = self.enhance_dynamic_range(ltd_data, params['dynamic_range_factor'])
        else:
            ltp_enhanced, ltd_enhanced = ltp_data, ltd_data
        
        # 2. 噪声鲁棒性
        if params['noise_robustness'] > 0:
            ltp_robust = self.apply_noise_robustness(ltp_enhanced, params['noise_robustness'])
            ltd_robust = self.apply_noise_robustness(ltd_enhanced, params['noise_robustness'])
        else:
            ltp_robust, ltd_robust = ltp_enhanced, ltd_enhanced
        
        # 3. 自适应缩放
        if params['adaptive_scaling']:
            ltp_scaled, ltd_scaled = self.adaptive_feature_scaling(ltp_robust, ltd_robust)
        else:
            ltp_scaled, ltd_scaled = ltp_robust, ltd_robust
        
        # 4. 正则化
        if params['regularization_strength'] > 0:
            ltp_final = self.apply_regularization(ltp_scaled, params['regularization_strength'])
            ltd_final = self.apply_regularization(ltd_scaled, params['regularization_strength'])
        else:
            ltp_final, ltd_final = ltp_scaled, ltd_scaled
        
        # 5. 与SNN参数匹配
        # 根据SNN的tau和v_threshold调整数据范围
        tau_factor = tau / 20.0  # 基准tau=20
        threshold_factor = v_threshold / 1.0  # 基准threshold=1.0
        
        # 调整数据以匹配SNN参数
        ltp_matched = ltp_final * tau_factor * threshold_factor
        ltd_matched = ltd_final * tau_factor * threshold_factor
        
        # 6. 确保数据范围合理
        ltp_matched = np.clip(ltp_matched, 0, v_threshold * 2)
        ltd_matched = np.clip(ltd_matched, 0, v_threshold * 2)
        
        return ltp_matched, ltd_matched
    
    def configure_snn_optimization(self, **kwargs):
        """配置SNN优化参数"""
        for key, value in kwargs.items():
            if key in self.snn_optimization_params:
                self.snn_optimization_params[key] = value
                print(f"SNN优化参数已更新: {key} = {value}")
            else:
                print(f"警告: 未知的SNN优化参数: {key}")
    
    def get_snn_optimization_info(self):
        """获取SNN优化信息"""
        info = {
            'current_params': self.snn_optimization_params.copy(),
            'description': {
                'dynamic_range_factor': '动态范围增强因子，越大对比度越强',
                'temporal_diversity': '是否启用时间多样性特征',
                'noise_robustness': '噪声鲁棒性级别，0-1之间',
                'feature_enhancement': '是否启用特征增强',
                'adaptive_scaling': '是否启用自适应缩放',
                'regularization_strength': '正则化强度，防止过拟合'
            },
            'recommendations': {
                'high_accuracy': {
                    'dynamic_range_factor': 2.5,
                    'noise_robustness': 0.05,
                    'regularization_strength': 0.02
                },
                'fast_training': {
                    'dynamic_range_factor': 1.5,
                    'noise_robustness': 0.15,
                    'regularization_strength': 0.005
                },
                'robust_generalization': {
                    'dynamic_range_factor': 2.0,
                    'noise_robustness': 0.2,
                    'regularization_strength': 0.03
                }
            }
        }
        return info

    def extract_peak_points(self, time_data, current_data, height=None, distance=None, prominence=None):
        """提取所有的局部最大值点作为峰值点"""
        try:
            # 使用 scipy.signal.find_peaks 进行更强大的峰值检测
            from scipy.signal import find_peaks
            
            # 设置默认参数
            kwargs = {}
            if height is not None and height > 0: kwargs['height'] = height
            if distance is not None and distance > 0: kwargs['distance'] = distance
            if prominence is not None and prominence > 0: kwargs['prominence'] = prominence
            
            # 如果没有设置任何参数，使用默认distance=10
            if not kwargs:
                kwargs['distance'] = 10
            
            peaks, _ = find_peaks(current_data, **kwargs)
            
            # 如果没有找到任何峰值，回退到简单的局部最大值方法或返回前几个点
            if len(peaks) == 0:
                print("Warning: find_peaks found no peaks, falling back to simple local maxima.")
                peaks = []
                for i in range(1, len(current_data)-1):
                    if current_data[i] > current_data[i-1] and current_data[i] > current_data[i+1]:
                        peaks.append(i)
                
                if len(peaks) == 0:
                    peaks = list(range(min(10, len(current_data))))
            
            # 提取峰值点的时间和电流值
            peak_times = time_data[peaks]
            peak_currents = current_data[peaks]
            
            return np.array(peaks), peak_times, peak_currents
            
        except Exception as e:
            print(f"Error in peak extraction: {e}")
            # Fallback implementation
            peaks = list(range(min(10, len(current_data))))
            return np.array(peaks), time_data[peaks], current_data[peaks]
    
    def split_peaks_into_ltd_ltp(self, peak_times, peak_currents):
        """将峰值点按趋势分为LTD和LTP两组
        
        策略：
        1. 找到全局最大值点作为转折点 (Turning Point)
        2. 0 到 最大值点 -> LTP (上升阶段)
        3. 最大值点 到 结束 -> LTD (下降阶段)
        4. 即使局部有波动，也严格按照时间轴分割，保留器件真实行为
        """
        if len(peak_times) < 2:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 按时间顺序排序 (关键)
        sort_idx = np.argsort(peak_times)
        sorted_times = peak_times[sort_idx]
        sorted_currents = peak_currents[sort_idx]
        
        # 找到全局最大值的索引
        # 注意：如果有多个相同的最大值，我们通常取第一个或中间的
        # 这里为了稳健，取平滑后的最大值位置可能更好，但直接取最大值也行
        max_idx = np.argmax(sorted_currents)
        
        # 为了防止噪声导致的最大值误判（例如开始就有个尖峰），
        # 我们可以先做一个简单的移动平均来找趋势的最高点
        window_size = min(5, len(sorted_currents))
        if window_size > 1:
            smoothed_currents = np.convolve(sorted_currents, np.ones(window_size)/window_size, mode='same')
            max_idx_smooth = np.argmax(smoothed_currents)
            # 如果平滑后的最大值位置和原始最大值位置相差不大，优先用原始的
            # 如果相差很大，说明原始最大值可能是个离群点
            if abs(max_idx - max_idx_smooth) > len(sorted_currents) * 0.1:
                print(f"警告：原始最大值位置({max_idx})与趋势最大值位置({max_idx_smooth})偏离较大，使用趋势最大值")
                max_idx = max_idx_smooth

        print(f"分割点索引: {max_idx}, 时间: {sorted_times[max_idx]:.2f}, 电流: {sorted_currents[max_idx]:.2e}")
        
        # 分割数据
        # LTP: 包含最大值点
        ltp_times = sorted_times[:max_idx+1]
        ltp_currents = sorted_currents[:max_idx+1]
        
        # LTD: 从最大值点开始（也可以不包含，看具体需求，这里为了连续性包含）
        # 或者从最大值点后一个开始
        if max_idx + 1 < len(sorted_times):
            ltd_times = sorted_times[max_idx:]
            ltd_currents = sorted_currents[max_idx:]
        else:
            ltd_times = np.array([])
            ltd_currents = np.array([])
            
        return ltd_times, ltd_currents, ltp_times, ltp_currents

        
        
    
    def process_curve_scientifically(self, times, currents, label="Curve"):
        """
        科学地处理曲线数据：
        1. 排序
        2. 强力去噪（Savitzky-Golay 滤波）
        3. 保留真实趋势，仅去除高频噪声，不强制单调性
        """
        if len(times) < 2:
            return times, currents
            
        # 1. 排序
        sort_idx = np.argsort(times)
        t_sorted = times[sort_idx]
        c_sorted = currents[sort_idx]
        
        # 2. 强力平滑 (Savitzky-Golay Filter)
        # 窗口长度必须是奇数且小于数据长度
        window_length = min(11, len(c_sorted))
        if window_length % 2 == 0:
            window_length -= 1
            
        if window_length > 3:
            try:
                from scipy.signal import savgol_filter
                # polyorder=2 (保留二次曲线特征), mode='interp' (插值填充边缘)
                c_smooth = savgol_filter(c_sorted, window_length, polyorder=2, mode='interp')
            except Exception as e:
                print(f"平滑失败，回退到原始数据: {e}")
                c_smooth = c_sorted
        else:
            c_smooth = c_sorted
            
        # 3. 移除强制单调性约束
        # 用户希望保留器件的真实趋势，即使存在非理想的波动
        # 我们只返回去噪后的数据
        c_constrained = c_smooth
                    
        return t_sorted, c_constrained

    def load_data(self, file_path, height=None, distance=None, prominence=None):
        """加载突触可塑性数据，并计算峰值数"""
        try:
            # 读取数据，明确指定表头行
            data = pd.read_excel(file_path, header=0)
            print(f"数据形状: {data.shape}")
            print(f"列名: {data.columns.tolist()}")

            # 验证数据列数
            if data.shape[1] < 2:
                raise ValueError(f"数据列数不足: 需要至少2列, 当前有{data.shape[1]}列")

            # 提取数据
            time_data = data.iloc[:, 0].values
            current_data = data.iloc[:, 1].values

            print(f"原始数据范围: 时间 {time_data.min():.2f} - {time_data.max():.2f}, 电流 {current_data.min():.2e} - {current_data.max():.2e}")

            # 提取峰值点
            peaks, peak_times, peak_currents = self.extract_peak_points(time_data, current_data, height=height, distance=distance, prominence=prominence)
            self.peak_count = len(peaks)
            
            print(f"检测到 {self.peak_count} 个峰值点")
            
            if self.peak_count < 2:
                raise ValueError("检测到的峰值点太少，无法进行LTP/LTD分析")

            # 将峰值点分为LTD和LTP两组
            ltd_times, ltd_currents, ltp_times, ltp_currents = self.split_peaks_into_ltd_ltp(peak_times, peak_currents)
            
            print(f"LTD组: {len(ltd_times)} 个点, LTP组: {len(ltp_times)} 个点")

            # 科学处理数据：仅平滑和排序，不强制拟合
            ltd_t_clean, ltd_c_clean = self.process_curve_scientifically(ltd_times, ltd_currents, "LTD")
            ltp_t_clean, ltp_c_clean = self.process_curve_scientifically(ltp_times, ltp_currents, "LTP")
            
            # 归一化时间到 [0, 1] 以便后续处理
            # 注意：这里我们分别保存 LTP 和 LTD 曲线，而不是混合它们
            
            if len(ltp_t_clean) > 0:
                ltp_t_norm = (ltp_t_clean - ltp_t_clean.min()) / (ltp_t_clean.max() - ltp_t_clean.min() + 1e-9)
                self.ltp_data = np.column_stack((ltp_t_norm, ltp_c_clean))
            else:
                self.ltp_data = np.zeros((0, 2))

            if len(ltd_t_clean) > 0:
                ltd_t_norm = (ltd_t_clean - ltd_t_clean.min()) / (ltd_t_clean.max() - ltd_t_clean.min() + 1e-9)
                self.ltd_data = np.column_stack((ltd_t_norm, ltd_c_clean))
            else:
                # 如果没有LTD数据，使用翻转的LTP作为近似（如果用户同意）或全零
                if len(ltp_c_clean) > 0:
                     print("警告: 未检测到LTD数据，使用LTP逆序作为LTD近似")
                     self.ltd_data = np.column_stack((ltp_t_norm, ltp_c_clean[::-1]))
                else:
                    self.ltd_data = np.zeros((0, 2))
            
            # 保存原始数据供可视化使用
            self.raw_data = {
                'pulses': peak_times,
                'ltp': peak_currents, # 这里的命名保留兼容性，实际是所有peaks
                'ltd': peak_currents, 
                'peaks': peaks,
                'original_time': time_data,
                'original_current': current_data,
                'ltp_clean': ltp_c_clean, # 新增清洁数据
                'ltd_clean': ltd_c_clean
            }

            print(f"数据加载完成 (科学模式)：")
            print(f"- 峰值数：{self.peak_count}")
            if len(ltp_c_clean) > 0:
                print(f"- LTP范围：{ltp_c_clean.min():.2e} - {ltp_c_clean.max():.2e}")
            if len(ltd_c_clean) > 0:
                print(f"- LTD范围：{ltd_c_clean.min():.2e} - {ltd_c_clean.max():.2e}")

            return True

        except Exception as e:
            print(f"数据加载错误: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def load_manual_data(self):
        """加载手动输入的默认数据"""
        try:
            self.use_manual_data = True
            
            # 创建时间序列
            time_ltp = np.linspace(0, 1, len(self.DEFAULT_LTP_DATA))
            time_ltd = np.linspace(0, 1, len(self.DEFAULT_LTD_DATA))
            
            # 转换为numpy数组
            ltp_values = np.array(self.DEFAULT_LTP_DATA)
            ltd_values = np.array(self.DEFAULT_LTD_DATA)
            
            # 保存原始数据
            self.raw_data = {
                'pulses': time_ltp,
                'ltp': ltp_values,
                'ltd': ltd_values
            }
            
            # 归一化处理（只做一次）
            ltp_norm = (ltp_values - ltp_values.min()) / (ltp_values.max() - ltp_values.min())
            ltd_norm = (ltd_values - ltd_values.min()) / (ltd_values.max() - ltd_values.min())
            
            # 创建数据对
            self.ltp_data = np.column_stack((time_ltp, ltp_norm))
            self.ltd_data = np.column_stack((time_ltd, ltd_norm))
            
            # 设置峰值数量
            self.peak_count = len(ltp_values)
            
            print("手动数据加载完成")
            return True
            
        except Exception as e:
            print(f"加载手动数据错误: {str(e)}")
            return False

    def normalize_data(self, num_points=100, snn_params=None):
        """归一化数据并进行采样点数验证 - LTP和LTD分别归一化，集成SNN优化"""
        try:
            if self.ltp_data is None or self.ltd_data is None:
                raise ValueError("请先加载数据")

            # 移除强制限制：允许 num_points > peak_count (通过插值实现)
            # if self.peak_count is not None:
            #    num_points = min(num_points, self.peak_count)
            
            # 创建目标均匀时间轴 [0, 1]
            x_new = np.linspace(0, 1, num_points)
            
            # 使用 numpy.interp 进行基于时间的线性插值 (保留原始时间特征)
            # LTP插值
            if len(self.ltp_data) > 1:
                ltp_t = self.ltp_data[:, 0]
                ltp_v = self.ltp_data[:, 1]
                # 确保时间范围覆盖 [0, 1]，如果不覆盖则外推或截断
                ltp_resampled = np.interp(x_new, ltp_t, ltp_v)
            else:
                ltp_resampled = np.zeros(num_points)
                
            # LTD插值
            if len(self.ltd_data) > 1:
                ltd_t = self.ltd_data[:, 0]
                ltd_v = self.ltd_data[:, 1]
                ltd_resampled = np.interp(x_new, ltd_t, ltd_v)
            else:
                ltd_resampled = np.zeros(num_points)

            # 转换为Tensor进行后续处理 (保持兼容性)
            ltp_tensor = torch.from_numpy(ltp_resampled).float()
            ltd_tensor = torch.from_numpy(ltd_resampled).float()

            # 分别对LTP和LTD进行归一化到[0,1]范围
            # LTP归一化
            ltp_min, ltp_max = torch.min(ltp_tensor), torch.max(ltp_tensor)
            if ltp_max > ltp_min:
                ltp_normalized = (ltp_tensor - ltp_min) / (ltp_max - ltp_min)
            else:
                ltp_normalized = torch.zeros_like(ltp_tensor)
            
            # LTD归一化
            ltd_min, ltd_max = torch.min(ltd_tensor), torch.max(ltd_tensor)
            if ltd_max > ltd_min:
                ltd_normalized = (ltd_tensor - ltd_min) / (ltd_max - ltd_min)
            else:
                ltd_normalized = torch.zeros_like(ltd_tensor)
            
            # 转换为numpy进行SNN优化
            ltp_np = ltp_normalized.numpy()
            ltd_np = ltd_normalized.numpy()
            
            # 应用SNN优化
            if snn_params is not None:
                tau = snn_params.get('tau', 20.0)
                v_threshold = snn_params.get('v_threshold', 1.0)
                ltp_optimized, ltd_optimized = self.optimize_for_snn(ltp_np, ltd_np, tau, v_threshold)
                print(f"SNN优化已应用: tau={tau}, v_threshold={v_threshold}")
            else:
                ltp_optimized, ltd_optimized = ltp_np, ltd_np
            
            # 对LTP进行适度平滑 (仅用于消除采样噪声，不改变形状)
            if len(ltp_optimized) > 5:
                window_size = 3
                ltp_optimized = self.smooth_data(ltp_optimized, window_size)
            
            # 对LTD进行适度平滑
            if len(ltd_optimized) > 5:
                window_size = 3
                ltd_optimized = self.smooth_data(ltd_optimized, window_size)
                
            # 移除强制递减/递增的逻辑，保留科学性

            # 存储优化后的结果
            self.normalized_data = {
                'x': x_new,
                'ltp': ltp_optimized,
                'ltd': ltd_optimized,
                'num_points': num_points,
                'peak_count': self.peak_count,
                'original_points': num_points,
                'ltp_original_range': (float(ltp_min), float(ltp_max)),
                'ltd_original_range': (float(ltd_min), float(ltd_max))
            }

            # 添加统计信息
            self.normalized_data.update({
                'stats': {
                    'ltp_mean': float(torch.mean(ltp_normalized)),
                    'ltp_std': float(torch.std(ltp_normalized)),
                    'ltd_mean': float(torch.mean(ltd_normalized)),
                    'ltd_std': float(torch.std(ltd_normalized)),
                    'diff_mean': float(torch.mean(ltp_normalized - ltd_normalized)),
                    'max_diff': float(torch.max(torch.abs(ltp_normalized - ltd_normalized))),
                    'ltp_range_original': f"{float(ltp_min):.2e} - {float(ltp_max):.2e}",
                    'ltd_range_original': f"{float(ltd_min):.2e} - {float(ltd_max):.2e}"
                }
            })

            return self.normalized_data

        except Exception as e:
            print(f"归一化错误: {str(e)}")
            raise ValueError(f"数据归一化失败: {str(e)}")

    def get_peak_count(self):
        """获取峰值数量"""
        return self.peak_count

    def validate_sampling_points(self, num_points):
        """验证采样点数是否有效"""
        if self.peak_count is None:
            raise ValueError("请先加载数据")
        
        # 移除严格限制，改为警告（或者完全允许，因为插值可以处理）
        if num_points > self.peak_count:
            print(f"提示: 采样点数({num_points})大于实际峰值数量({self.peak_count})，将通过插值生成数据")
            # raise ValueError(f"采样点数({num_points})不能大于实际峰值数量({self.peak_count})")
            
        return True

    def plot_data(self, save_path=None):
        """绘制突触可塑性数据"""
        if self.normalized_data is None:
            return False

        plt.figure(figsize=(10, 6))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        if self.raw_data is not None:
            raw_x = (self.raw_data['pulses'] - self.raw_data['pulses'].min()) / \
                    (self.raw_data['pulses'].max() - self.raw_data['pulses'].min())
            plt.scatter(raw_x, self.raw_data['ltp'], c='blue', alpha=0.3, label='原始LTP数据')
            plt.scatter(raw_x, self.raw_data['ltd'], c='red', alpha=0.3, label='原始LTD数据')

        plt.plot(self.normalized_data['x'], self.normalized_data['ltp'], 'b-',
                 label='LTP (归一化)', linewidth=2)
        plt.plot(self.normalized_data['x'], self.normalized_data['ltd'], 'r-',
                 label='LTD (归一化)', linewidth=2)

        plt.xlabel('归一化脉冲数')
        plt.ylabel('归一化权重变化')
        plt.title('突触权重变化的归一化表示')
        plt.grid(True)
        plt.legend()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return True
        else:
            plt.show()
            return True

    def save_data(self, filename):
        """保存归一化数据到CSV文件"""
        if self.normalized_data is None:
            return False

        try:
            data = pd.DataFrame({
                'normalized_pulse': self.normalized_data['x'],
                'normalized_ltp': self.normalized_data['ltp'],
                'normalized_ltd': self.normalized_data['ltd']
            })
            data.to_csv(filename, index=False)
            return True
        except Exception as e:
            print(f"Error saving data: {str(e)}")
            return False
