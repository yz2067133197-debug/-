import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.interpolate import make_interp_spline, interp1d
import torch
import matplotlib.pyplot as plt
import os
import scipy.optimize as opt


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
        self.synaptic_parameters = None
        
        # SNN优化参数
        self.snn_optimization_params = {
            'dynamic_range_factor': 1.0,      # 动态范围增强因子（1.0表示不增强）
            'temporal_diversity': False,       # 时间多样性
            'noise_robustness': 0.0,          # 噪声鲁棒性
            'feature_enhancement': False,      # 特征增强
            'adaptive_scaling': False,         # 自适应缩放
            'regularization_strength': 0.0    # 正则化强度
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
        """对数据进行平滑处理"""
        if len(data) < window_size:
            return data
        window = np.ones(window_size) / window_size
        return np.convolve(data, window, mode='same')
    
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
        # 定义默认关闭状态的参数值
        default_off_values = {
            'dynamic_range_factor': 1.0,
            'temporal_diversity': False,
            'noise_robustness': 0.0,
            'feature_enhancement': False,
            'adaptive_scaling': False,
            'regularization_strength': 0.0
        }
        
        for key, value in kwargs.items():
            if key in self.snn_optimization_params:
                self.snn_optimization_params[key] = value
                # 只有当参数不是默认关闭值时才打印日志
                if key not in default_off_values or value != default_off_values[key]:
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

    def extract_peak_points(self, time_data, current_data):
        """提取所有的局部最大值点作为峰值点"""
        # 使用简单的差分法找到所有局部最大值
        peaks = []
        
        # 遍历数据点，寻找局部最大值
        for i in range(1, len(current_data)-1):
            # 如果当前点比左右两侧的点都大，则认为是峰值点
            if current_data[i] > current_data[i-1] and current_data[i] > current_data[i+1]:
                peaks.append(i)
        
        # 如果没有找到任何峰值，返回前几个点
        if len(peaks) == 0:
            peaks = list(range(min(10, len(current_data))))
        
        # 提取峰值点的时间和电流值
        peak_times = time_data[peaks]
        peak_currents = current_data[peaks]
        
        return np.array(peaks), peak_times, peak_currents
    
    def split_peaks_into_ltd_ltp(self, peak_times, peak_currents):
        """将峰值点按电流变化趋势分为LTD和LTP两组，避免简单的时间分割"""
        if len(peak_times) < 2:  # 至少需要2个点
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # 按时间顺序排序
        sort_idx = np.argsort(peak_times)
        sorted_times = peak_times[sort_idx]
        sorted_currents = peak_currents[sort_idx]
        
        # 计算电流变化趋势
        current_diffs = np.diff(sorted_currents)
        avg_diff = np.mean(current_diffs)
        
        # 根据电流变化趋势分组
        # 正变化表示LTP（增强），负变化表示LTD（抑制）
        if avg_diff > 0:
            # 整体呈增强趋势，将较大的电流变化作为LTP
            ltp_mask = current_diffs > 0
            ltp_indices = np.where(ltp_mask)[0] + 1  # +1因为diff减少了一个元素
            ltd_indices = np.where(~ltp_mask)[0] + 1
            
            # 如果LTP组太少，确保至少有2个点
            if len(ltp_indices) < 2:
                ltp_indices = np.argsort(-sorted_currents)[-min(3, len(sorted_currents)//2):]
                ltd_indices = np.argsort(sorted_currents)[:min(3, len(sorted_currents)//2)]
        else:
            # 整体呈抑制趋势，将较大的电流变化作为LTD
            ltd_mask = current_diffs < 0
            ltd_indices = np.where(ltd_mask)[0] + 1
            ltp_indices = np.where(~ltd_mask)[0] + 1
            
            # 如果LTD组太少，确保至少有2个点
            if len(ltd_indices) < 2:
                ltd_indices = np.argsort(sorted_currents)[:min(3, len(sorted_currents)//2)]
                ltp_indices = np.argsort(-sorted_currents)[-min(3, len(sorted_currents)//2):]
        
        # 确保索引不重复且有效
        all_indices = np.unique(np.concatenate([ltp_indices, ltd_indices]))
        valid_indices = all_indices[all_indices < len(sorted_times)]
        
        if len(valid_indices) < 2:
            # 如果仍然无法有效分组，回退到简单的时间分割
            mid = len(sorted_times) // 2
            ltd_indices = np.arange(mid)
            ltp_indices = np.arange(mid, len(sorted_times))
        
        # 获取最终分组
        ltd_times = sorted_times[ltd_indices]
        ltd_currents = sorted_currents[ltd_indices]
        ltp_times = sorted_times[ltp_indices]
        ltp_currents = sorted_currents[ltp_indices]
        
        return ltd_times, ltd_currents, ltp_times, ltp_currents

    def create_preserving_ltd_curve(self, ltd_times, ltd_currents):
        """创建保留原始数据特征的LTD曲线，避免过度平滑"""
        if len(ltd_times) < 2:
            return ltd_times, ltd_currents
        
        # 归一化时间
        time_norm = (ltd_times - ltd_times.min()) / (ltd_times.max() - ltd_times.min())
        
        # 不强制使用预设的指数衰减曲线，而是保留原始数据的整体趋势
        # 仅进行轻微平滑以去除噪声
        smooth_currents = ltd_currents.copy()
        
        # 使用移动平均进行轻微平滑
        window_size = min(3, len(ltd_currents) // 2)
        if window_size > 1:
            for i in range(1, len(smooth_currents) - 1):
                # 权重逐渐增加，中心数据权重最大
                weights = np.exp(-np.abs(np.arange(-window_size, window_size+1)) / 2)
                weights = weights / np.sum(weights)
                start = max(0, i - window_size)
                end = min(len(smooth_currents), i + window_size + 1)
                # 修复：简化actual_weights的计算逻辑
                window_length = end - start
                if window_length > 0:
                    # 生成与窗口长度相匹配的权重
                    actual_weights = np.exp(-np.abs(np.arange(-(i-start), window_length-(i-start))) / 2)
                    actual_weights = actual_weights / np.sum(actual_weights)
                    smooth_currents[i] = np.sum(smooth_currents[start:end] * actual_weights)
        
        # 确保整体保持递减趋势（对于LTD数据），但允许局部波动
        trend_preserved = smooth_currents.copy()
        for i in range(1, len(trend_preserved)):
            # 只在当前值明显高于前一个值时才调整
            if trend_preserved[i] > trend_preserved[i-1] * 1.05:
                trend_preserved[i] = trend_preserved[i-1] * 1.02  # 允许小幅递增
        
        return time_norm, trend_preserved

    def create_preserving_ltp_curve(self, ltp_times, ltp_currents):
        """创建保留原始数据特征的LTP曲线，避免过度平滑"""
        if len(ltp_times) < 2:
            return ltp_times, ltp_currents
        
        # 归一化时间
        time_norm = (ltp_times - ltp_times.min()) / (ltp_times.max() - ltp_times.min())
        
        # 不强制使用预设的sigmoid曲线，而是保留原始数据的整体趋势
        # 仅进行轻微平滑以去除噪声
        smooth_currents = ltp_currents.copy()
        
        # 使用移动平均进行轻微平滑
        window_size = min(3, len(ltp_currents) // 2)
        if window_size > 1:
            for i in range(1, len(smooth_currents) - 1):
                # 权重逐渐增加，中心数据权重最大
                weights = np.exp(-np.abs(np.arange(-window_size, window_size+1)) / 2)
                weights = weights / np.sum(weights)
                start = max(0, i - window_size)
                end = min(len(smooth_currents), i + window_size + 1)
                # 修复：简化actual_weights的计算逻辑
                window_length = end - start
                if window_length > 0:
                    # 生成与窗口长度相匹配的权重
                    actual_weights = np.exp(-np.abs(np.arange(-(i-start), window_length-(i-start))) / 2)
                    actual_weights = actual_weights / np.sum(actual_weights)
                    smooth_currents[i] = np.sum(smooth_currents[start:end] * actual_weights)
        
        # 确保整体保持递增趋势（对于LTP数据），但允许局部波动
        trend_preserved = smooth_currents.copy()
        for i in range(1, len(trend_preserved)):
            # 只在当前值明显低于前一个值时才调整
            if trend_preserved[i] < trend_preserved[i-1] * 0.95:
                trend_preserved[i] = trend_preserved[i-1] * 0.98  # 允许小幅递减
        
        return time_norm, trend_preserved

    def load_data(self, file_path):
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
            peaks, peak_times, peak_currents = self.extract_peak_points(time_data, current_data)
            self.peak_count = len(peaks)
            
            print(f"检测到 {self.peak_count} 个峰值点")
            
            if self.peak_count < 2:
                raise ValueError("检测到的峰值点太少，无法进行LTP/LTD分析")
        
            # 将峰值点分为LTD和LTP两组
            ltd_times, ltd_currents, ltp_times, ltp_currents = self.split_peaks_into_ltd_ltp(peak_times, peak_currents)
            
            print(f"LTD组: {len(ltd_times)} 个点, LTP组: {len(ltp_times)} 个点")
        
            # 创建平滑的LTD和LTP曲线
            # 创建平滑的LTD和LTP曲线 - 使用新的保留原始特征的方法
            ltd_time_norm, ltd_smooth = self.create_preserving_ltd_curve(ltd_times, ltd_currents)
            ltp_time_norm, ltp_smooth = self.create_preserving_ltp_curve(ltp_times, ltp_currents)
            
            # 不再混合LTD和LTP数据，保持它们的独立性
            # 而是分别进行处理和存储
            
            # 为LTD和LTP创建独立的时间序列，范围都是[0,1]
            ltd_time_full = np.linspace(0, 1, len(ltd_smooth))
            ltp_time_full = np.linspace(0, 1, len(ltp_smooth))
            
            # 保存处理后的数据 - 保持LTD和LTP的独立性
            self.ltp_data = np.column_stack((ltp_time_full, ltp_smooth))
            self.ltd_data = np.column_stack((ltd_time_full, ltd_smooth))
            
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

    # 将normalize_data方法替换为以下实现
    def normalize_data(self, num_points=100, snn_params=None):
        """归一化数据并提取LTP/LTD参数"""
        try:
            if self.ltp_data is None or self.ltd_data is None:
                raise ValueError("LTP或LTD数据未初始化，请先加载数据")

            # 验证采样点数
            if num_points < 10 or num_points > 1000:
                raise ValueError("采样点数应在10-1000之间")

            # 生成归一化脉冲序列（0-1范围）- 与峰值检测窗口相同
            pulse_norm = np.linspace(0, 1, num_points)
            
            # 处理不同类型的ltp_data和ltd_data
            # 检查是否为二维数组（时间和值）或一维数组（只有值）
            if isinstance(self.ltp_data, np.ndarray) and len(self.ltp_data.shape) == 2:
                ltp_values = self.ltp_data[:, 1]  # 提取值部分
            elif self.ltp_data is not None:
                ltp_values = self.ltp_data
            else:
                raise ValueError("LTP数据为空或格式错误")
                
            if isinstance(self.ltd_data, np.ndarray) and len(self.ltd_data.shape) == 2:
                ltd_values = self.ltd_data[:, 1]  # 提取值部分
            elif self.ltd_data is not None:
                ltd_values = self.ltd_data
            else:
                raise ValueError("LTD数据为空或格式错误")
            
            # 确保数据类型正确
            if isinstance(ltp_values, list):
                ltp_values = np.array(ltp_values)
            if isinstance(ltd_values, list):
                ltd_values = np.array(ltd_values)
            
            # 移除NaN和无穷大值
            ltp_values = ltp_values[~np.isnan(ltp_values) & ~np.isinf(ltp_values)]
            ltd_values = ltd_values[~np.isnan(ltd_values) & ~np.isinf(ltd_values)]
            
            # 确保数据不为空 - 不使用默认数据，直接抛出异常
            if len(ltp_values) == 0:
                raise ValueError("LTP数据为空，请确保数据已正确加载")
            if len(ltd_values) == 0:
                raise ValueError("LTD数据为空，请确保数据已正确加载")
            
            # 生成平滑的拟合曲线 - 与峰值检测窗口使用相同的方法
            ltp_fit = self.generate_ltp_curve(pulse_norm, ltp_values)
            ltd_fit = self.generate_ltd_curve(pulse_norm, ltd_values)
            
            # 应用SNN优化 - 与峰值检测窗口使用相同的方法
            tau = snn_params.get('tau', 20.0) if snn_params else 20.0
            v_threshold = snn_params.get('v_threshold', 1.0) if snn_params else 1.0
            ltp_optimized, ltd_optimized = self.optimize_for_snn(ltp_fit, ltd_fit, tau, v_threshold)
            
            # 确保数据在[0, 1]范围内，并处理异常情况
            ltp_min, ltp_max = np.min(ltp_optimized), np.max(ltp_optimized)
            ltd_min, ltd_max = np.min(ltd_optimized), np.max(ltd_optimized)
            
            # 处理LTP数据归一化
            if ltp_max - ltp_min < 1e-6:
                # 添加小值防止范围过小
                ltp_np = np.clip(ltp_optimized / (ltp_max + 1e-6), 0, 1)
            else:
                # 扩展范围以获得更好的学习效果
                extended_ltp_min = ltp_min - 0.1 * (ltp_max - ltp_min)
                extended_ltp_max = ltp_max + 0.1 * (ltp_max - ltp_min)
                ltp_np = np.clip((ltp_optimized - extended_ltp_min) / (extended_ltp_max - extended_ltp_min), 0, 1)
                
            # 处理LTD数据归一化
            if ltd_max - ltd_min < 1e-6:
                # 添加小值防止范围过小
                ltd_np = np.clip(ltd_optimized / (ltd_max + 1e-6), 0, 1)
            else:
                # 扩展范围以获得更好的学习效果
                extended_ltd_min = ltd_min - 0.1 * (ltd_max - ltd_min)
                extended_ltd_max = ltd_max + 0.1 * (ltd_max - ltd_min)
                ltd_np = np.clip((ltd_optimized - extended_ltd_min) / (extended_ltd_max - extended_ltd_min), 0, 1)
            
            # 存储优化后的结果
            self.normalized_data = {
                'x': pulse_norm,
                'ltp': ltp_np,
                'ltd': ltd_np,
                'num_points': len(pulse_norm),
                'peak_count': self.peak_count,
                'original_points': num_points
            }
            
            # 提取LTP/LTD参数
            if self.ltp_data is not None and self.ltd_data is not None:
                self.synaptic_parameters = self.extract_ltp_ltd_parameters()
                print(f"提取的突触参数: {self.synaptic_parameters}")
            
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
        if num_points > self.peak_count:
            raise ValueError(f"采样点数({num_points})不能大于实际峰值数量({self.peak_count})")
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

    def extract_ltp_ltd_parameters(self):
        """从LTP/LTD曲线中提取关键参数(G_max/G_min、τ、非线性系数)"""
        if self.ltp_data is None or self.ltd_data is None:
            raise ValueError("请先加载数据")
        
        parameters = {}
        
        # 提取LTP参数
        ltp_time = self.ltp_data[:, 0]
        ltp_current = self.ltp_data[:, 1]
        
        # 使用论文公式拟合LTP曲线：I_LTP(P) = B*(1-e^(-P/A)) + I_min
        def ltp_function(P, A, B, I_min):
            return B * (1 - np.exp(-P / A)) + I_min
        
        try:
            # 初始参数猜测
            I_min_guess = np.min(ltp_current)
            B_guess = np.max(ltp_current) - I_min_guess
            A_guess = 10.0
            
            # 拟合曲线
            popt, _ = opt.curve_fit(ltp_function, ltp_time, ltp_current,
                                  p0=[A_guess, B_guess, I_min_guess],
                                  bounds=([0.01, 0.01, -np.inf], [np.inf, np.inf, np.inf]))
            
            A_ltp, B_ltp, I_min_ltp = popt
            G_max = B_ltp + I_min_ltp  # 最大电导
            tau_ltp = A_ltp  # 时间常数
            
            parameters['ltp'] = {
                'A': A_ltp,        # 非线性系数
                'B': B_ltp,        # 峰值增量系数
                'I_min': I_min_ltp,# 最小电流
                'G_max': G_max,    # 最大电导
                'tau': tau_ltp     # 时间常数
            }
            
        except Exception as e:
            print(f"LTP参数拟合失败: {str(e)}")
            # 回退到直接从数据计算参数
            parameters['ltp'] = {
                'G_max': np.max(ltp_current),
                'G_min': np.min(ltp_current),
                'tau': 20.0  # 默认值
            }
        
        # 提取LTD参数
        ltd_time = self.ltd_data[:, 0]
        ltd_current = self.ltd_data[:, 1]
        
        # 使用论文公式拟合LTD曲线：I_LTD(P) = -B*(1-e^((P-P_max)/A)) + I_max
        def ltd_function(P, A, B, I_max):
            P_max = max(ltp_time) if len(ltp_time) > 0 else 1.0
            return -B * (1 - np.exp((P - P_max) / A)) + I_max
        
        try:
            # 初始参数猜测
            I_max_guess = np.max(ltd_current)
            B_guess = I_max_guess - np.min(ltd_current)
            A_guess = -10.0
            
            # 拟合曲线
            popt, _ = opt.curve_fit(ltd_function, ltd_time, ltd_current,
                                  p0=[A_guess, B_guess, I_max_guess],
                                  bounds=([-np.inf, 0.01, -np.inf], [-0.01, np.inf, np.inf]))
            
            A_ltd, B_ltd, I_max_ltd = popt
            G_min = np.min(ltd_current)  # 最小电导
            tau_ltd = abs(A_ltd)  # 时间常数（取绝对值）
            
            parameters['ltd'] = {
                'A': A_ltd,        # 非线性系数
                'B': B_ltd,        # 峰值减量系数
                'I_max': I_max_ltd,# 最大电流
                'G_min': G_min,    # 最小电导
                'tau': tau_ltd     # 时间常数
            }
            
        except Exception as e:
            print(f"LTD参数拟合失败: {str(e)}")
            # 回退到直接从数据计算参数
            parameters['ltd'] = {
                'G_max': np.max(ltd_current),
                'G_min': np.min(ltd_current),
                'tau': 20.0  # 默认值
            }
        
        # 计算LTP/LTD比率和范围等衍生参数
        if 'ltp' in parameters and 'ltd' in parameters:
            ltp_params = parameters['ltp']
            ltd_params = parameters['ltd']
            
            parameters['derived'] = {
                'ltp_ltd_ratio': ltp_params.get('G_max', 1.0) / max(1e-10, ltd_params.get('G_min', 1.0)),
                'dynamic_range': ltp_params.get('G_max', 1.0) - ltd_params.get('G_min', 0.0),
                'tau_ratio': ltp_params.get('tau', 20.0) / max(1e-10, ltd_params.get('tau', 20.0))
            }
        
        return parameters

    def get_synaptic_parameters(self):
        """获取提取的LTP/LTD参数"""
        if not hasattr(self, 'synaptic_parameters') or self.synaptic_parameters is None:
            self.synaptic_parameters = self.extract_ltp_ltd_parameters()
        return self.synaptic_parameters