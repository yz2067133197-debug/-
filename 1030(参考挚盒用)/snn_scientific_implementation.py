import torch
import torch.nn as nn
import numpy as np
import time
import types
from tqdm import tqdm

class ScientificSTDPLearning:
    """科学实现的STDP学习规则，基于精确脉冲时间差"""
    def __init__(self, a_plus=0.01, a_minus=0.01, tau_plus=20.0, tau_minus=20.0,
                 weight_min=0.0, weight_max=1.0, weight_decay=0.0001):
        # 标准STDP参数
        self.a_plus = a_plus         # LTP学习率
        self.a_minus = a_minus       # LTD学习率
        self.tau_plus = tau_plus     # LTP时间常数
        self.tau_minus = tau_minus   # LTD时间常数
        self.weight_min = weight_min # 权重最小值
        self.weight_max = weight_max # 权重最大值
        self.weight_decay = weight_decay  # 权重衰减系数
        
        # 存储初始参数用于动态调度
        self.initial_a_plus = a_plus
        self.initial_a_minus = a_minus
        self.initial_tau_plus = tau_plus
        self.initial_tau_minus = tau_minus
        
        # 控制标志
        self.enable = True
        self.use_dynamic_params = True
        
        # 训练进度跟踪
        self.epoch = 0
        self.total_epochs = 10
    
    def update_parameters(self, epoch, total_epochs):
        """科学的参数动态调整方法"""
        self.epoch = epoch
        self.total_epochs = total_epochs
        
        # 计算训练进度
        progress = epoch / max(1, total_epochs)
        
        # 指数衰减学习率，比线性衰减更科学
        self.a_plus = self.initial_a_plus * np.exp(-progress * 3)
        self.a_minus = self.initial_a_minus * np.exp(-progress * 3)
        
        # 时间常数随训练进行适当增加，提高长期依赖性
        self.tau_plus = min(30.0, self.initial_tau_plus + progress * 20)
        self.tau_minus = min(30.0, self.initial_tau_minus + progress * 20)
        
        # 权重衰减逐渐增加，增强正则化效果
        self.weight_decay = 0.0001 + progress * 0.0009

class ScientificLIFNeuron(nn.Module):
    """科学实现的漏电整合发放(LIF)神经元模型"""
    def __init__(self, num_neurons, tau=10.0, dt=1.0, v_threshold=0.5, v_reset=0.0, v_rest=0.0, 
                 ref_period=2.0, adaptation_rate=0.0):
        super().__init__()
        self.num_neurons = num_neurons
        self.tau = tau            # 膜时间常数
        self.dt = dt              # 时间步长
        self.v_threshold = v_threshold  # 发放阈值
        self.v_reset = v_reset    # 重置电压
        self.v_rest = v_rest      # 静息电位
        self.ref_period = ref_period    # 不应期
        self.adaptation_rate = adaptation_rate  # 自适应阈值调整率
        
        # 初始化神经元状态
        self.membrane_potential = torch.full((num_neurons,), v_rest, dtype=torch.float32)
        self.adaptive_threshold = torch.full((num_neurons,), v_threshold, dtype=torch.float32)
        self.last_spike_time = torch.full((num_neurons,), -float('inf'), dtype=torch.float32)
        self.refractory = torch.zeros(num_neurons, dtype=torch.bool)
    
    def reset_state(self):
        """重置神经元状态"""
        device = self.membrane_potential.device
        self.membrane_potential = torch.full((self.num_neurons,), self.v_rest, device=device, dtype=torch.float32)
        self.adaptive_threshold = torch.full((self.num_neurons,), self.v_threshold, device=device, dtype=torch.float32)
        self.last_spike_time = torch.full((self.num_neurons,), -float('inf'), device=device, dtype=torch.float32)
        self.refractory = torch.zeros(self.num_neurons, device=device, dtype=torch.bool)
    
    def to(self, device):
        """将神经元状态移动到指定设备"""
        self.membrane_potential = self.membrane_potential.to(device)
        self.adaptive_threshold = self.adaptive_threshold.to(device)
        self.last_spike_time = self.last_spike_time.to(device)
        self.refractory = self.refractory.to(device)
        return self
    
    def forward(self, input_current, t):
        """前向传播：根据输入电流更新膜电位并判断是否发放脉冲"""
        # 计算不应期
        self.refractory = (t - self.last_spike_time) < self.ref_period
        
        # 更新膜电位
        dt_over_tau = self.dt / self.tau
        self.membrane_potential[~self.refractory] += (
            (self.v_rest - self.membrane_potential[~self.refractory]) * dt_over_tau +
            input_current[~self.refractory] * dt_over_tau
        )
        
        # 判断发放脉冲的神经元
        spike = self.membrane_potential >= self.adaptive_threshold
        
        # 更新神经元状态
        self.last_spike_time[spike] = t
        self.membrane_potential[spike] = self.v_reset  # 重置膜电位
        
        # 自适应阈值调整
        if self.adaptation_rate > 0:
            self.adaptive_threshold[spike] += self.adaptation_rate
            self.adaptive_threshold[~spike] -= self.adaptation_rate * 0.5
            self.adaptive_threshold = torch.clamp(self.adaptive_threshold, 
                                                 self.v_threshold * 0.8, 
                                                 self.v_threshold * 1.5)
        
        return spike.float()

class ScientificSNNSynapticLayer(nn.Module):
    """科学实现的SNN突触层，支持精确STDP学习规则"""
    def __init__(self, in_features, out_features, stdplearning=None, **kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 初始化权重 - 使用Xavier初始化以获得更好的初始值分布
        self.weights = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.xavier_normal_(self.weights)
        
        # STDP学习规则对象
        self.stdplearning = stdplearning if stdplearning is not None else ScientificSTDPLearning()
        
        # 确保STDP有enable标志
        if not hasattr(self.stdplearning, 'enable'):
            self.stdplearning.enable = True
        
        # 记录最近一次脉冲时间
        self.pre_spike_times = torch.full((in_features,), -float('inf'))
        self.post_spike_times = torch.full((out_features,), -float('inf'))
        
        # 使用真实的LIF神经元模型
        self.neurons = ScientificLIFNeuron(
            num_neurons=out_features,
            tau=kwargs.get('tau', 10.0),
            v_threshold=kwargs.get('v_threshold', 0.5),
            v_reset=kwargs.get('v_reset', 0.0),
            adaptation_rate=kwargs.get('adaptation_rate', 0.0)
        )
    
    def reset(self):
        """重置层状态"""
        device = self.weights.device
        
        # 重置脉冲时间
        self.pre_spike_times = torch.full((self.in_features,), -float('inf'), device=device, dtype=torch.float32)
        self.post_spike_times = torch.full((self.out_features,), -float('inf'), device=device, dtype=torch.float32)
        
        # 重置神经元状态
        if hasattr(self, 'neurons') and hasattr(self.neurons, 'reset_state'):
            self.neurons.reset_state()
    
    def reset_state(self):
        """兼容方法：重置层状态"""
        self.reset()
    
    def to(self, device):
        """将层移动到指定设备"""
        super().to(device)
        self.pre_spike_times = self.pre_spike_times.to(device)
        self.post_spike_times = self.post_spike_times.to(device)
        if hasattr(self, 'neurons') and hasattr(self.neurons, 'to'):
            self.neurons.to(device)
        return self
    
    def forward(self, x, t=None, training=None, targets=None):
        """前向传播，支持监督STDP"""
        try:
            # 使用外部提供的training标志或默认值
            is_training = training if training is not None else self.training
            current_time = t if t is not None else 0.0
            
            # 确保x是tensor并在正确设备上
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.weights.device)
            else:
                x = x.to(device=self.weights.device, dtype=torch.float32)
            
            # 防止0维张量，确保至少是1维
            if x.dim() == 0:
                x = x.unsqueeze(0)
            
            # 处理批次或单个样本
            is_batch = x.dim() > 1
            
            if is_batch:
                # 批次处理
                batch_size = x.size(0)
                outputs = []
                
                for b in range(batch_size):
                    # 确保x[b]是1维张量用于矩阵乘法
                    sample = x[b].view(-1)  # 确保至少1维
                    
                    # 计算当前样本的输入电流
                    input_current = torch.mv(self.weights, sample)
                    
                    # 使用LIF神经元模型生成脉冲
                    post_spikes = self.neurons(input_current, current_time)
                    
                    # 更新权重（如果在训练模式）
                    if is_training and self.stdplearning.enable:
                        sample_target = targets[b] if targets is not None else None
                        self._update_single_sample_weights(sample, post_spikes, current_time, target=sample_target)
                    
                    outputs.append(post_spikes)
                
                # 堆叠批次结果
                post_spikes = torch.stack(outputs)
            else:
                # 单个样本处理 - 确保是1维张量
                x_1d = x.view(-1)
                input_current = torch.mv(self.weights, x_1d)
                post_spikes = self.neurons(input_current, current_time)
                
                # 更新权重（如果在训练模式）
                if is_training and self.stdplearning.enable:
                    self._update_single_sample_weights(x_1d, post_spikes, current_time, target=targets)
            
            return post_spikes
        except Exception as e:
            print(f"SNNSynapticLayer forward error: {e}")
            # 错误处理并返回适当形状的零张量
            device = self.weights.device
            try:
                if x.dim() > 1:
                    return torch.zeros((x.size(0), self.out_features), device=device, dtype=torch.float32)
                else:
                    return torch.zeros((self.out_features,), device=device, dtype=torch.float32)
            except:
                # 极端情况下返回基本形状的零张量
                return torch.zeros((self.out_features,), device=device, dtype=torch.float32)
    
    def _update_single_sample_weights(self, pre_spikes, post_spikes, t, target=None):
        """实现监督STDP权重更新"""
        device = self.weights.device
        
        # 找出活跃的前突触神经元和后突触神经元
        active_pre = (pre_spikes > 0).nonzero().squeeze()
        active_post = (post_spikes > 0).nonzero().squeeze()
        
        # 更新前突触脉冲时间
        if active_pre.numel() > 0:
            if active_pre.dim() == 0:  # 单个神经元活跃
                active_pre = active_pre.unsqueeze(0)
            self.pre_spike_times[active_pre] = t
        
        # 更新后突触脉冲时间
        if active_post.numel() > 0:
            if active_post.dim() == 0:  # 单个神经元活跃
                active_post = active_post.unsqueeze(0)
            self.post_spike_times[active_post] = t
        
        # 获取STDP参数
        a_plus = self.stdplearning.a_plus
        a_minus = self.stdplearning.a_minus
        tau_plus = self.stdplearning.tau_plus
        tau_minus = self.stdplearning.tau_minus
        weight_min = self.stdplearning.weight_min
        weight_max = self.stdplearning.weight_max
        weight_decay = self.stdplearning.weight_decay
        
        # 计算所有前突触和后突触之间的时间差
        valid_pre = self.pre_spike_times > -float('inf')
        valid_post = self.post_spike_times > -float('inf')
        
        # 判断是否是输出层且有标签
        is_output_layer = hasattr(self, 'is_output_layer') and self.is_output_layer
        has_supervision = is_output_layer and target is not None
        
        # 监督学习系数 - 控制监督信号的强度
        supervision_strength = 1.5 if has_supervision else 1.0
        
        # 实现真正的STDP学习规则
        # 1. 对于每个活跃的前突触神经元
        if active_pre.numel() > 0:
            for pre_idx in active_pre:
                # 计算与所有后突触神经元的时间差
                # LTP: 前突触先发放，后突触后发放 (pre_spike_time < post_spike_time)
                for post_idx in valid_post.nonzero().squeeze():
                    if self.pre_spike_times[pre_idx] < self.post_spike_times[post_idx]:
                        delta_t = self.post_spike_times[post_idx] - self.pre_spike_times[pre_idx]
                        # LTP: 长期增强
                        LTP = a_plus * torch.exp(-delta_t / tau_plus)
                        
                        # 应用监督信号（仅对输出层）
                        if has_supervision:
                            # 如果是目标类神经元，增强LTP
                            if post_idx == target:
                                LTP *= supervision_strength
                            # 如果不是目标类神经元，减弱LTP
                            else:
                                LTP *= 0.5
                        
                        self.weights.data[post_idx, pre_idx] += LTP
        
        # 2. 对于每个活跃的后突触神经元
        if active_post.numel() > 0:
            for post_idx in active_post:
                # 计算与所有前突触神经元的时间差
                # LTD: 后突触先发放，前突触后发放 (post_spike_time < pre_spike_time)
                for pre_idx in valid_pre.nonzero().squeeze():
                    if self.post_spike_times[post_idx] < self.pre_spike_times[pre_idx]:
                        delta_t = self.pre_spike_times[pre_idx] - self.post_spike_times[post_idx]
                        # LTD: 长期抑制
                        LTD = -a_minus * torch.exp(-delta_t / tau_minus)
                        
                        # 应用监督信号（仅对输出层）
                        if has_supervision:
                            # 如果是目标类神经元，减弱LTD
                            if post_idx == target:
                                LTD *= 0.5
                            # 如果不是目标类神经元，增强LTD
                            else:
                                LTD *= supervision_strength
                        
                        self.weights.data[post_idx, pre_idx] += LTD
        
        # 应用权重衰减（正则化）
        self.weights.data *= (1 - weight_decay)
        
        # 权重裁剪，确保在有效范围内
        self.weights.data = torch.clamp(self.weights.data, weight_min, weight_max)
    
    def update_stdp_parameters(self, ltp_params, ltd_params):
        """更新STDP参数"""
        if hasattr(self, 'stdplearning'):
            # 如果提供了参数字典，使用其中的值
            if isinstance(ltp_params, dict):
                for key, value in ltp_params.items():
                    if hasattr(self.stdplearning, key):
                        setattr(self.stdplearning, key, value)
            # 如果提供了单独的值，直接设置
            elif isinstance(ltp_params, (int, float)):
                self.stdplearning.a_plus = ltp_params
                if isinstance(ltd_params, (int, float)):
                    self.stdplearning.a_minus = ltd_params

class ScientificSNN(nn.Module):
    """科学实现的脉冲神经网络模型"""
    def __init__(self, input_dim, output_dim, hidden_layers=1, hidden_neurons=100, time_steps=100,
                 tau=10.0, v_threshold=0.5, v_reset=0.0, adaptation_rate=0.0):
        super().__init__()
        
        # 初始化网络参数
        self.time_steps = time_steps
        
        # 构建层大小列表
        hidden_sizes = [hidden_neurons] * hidden_layers
        layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        
        # 为每一层创建独立的STDP学习器和神经元参数
        self.layers = nn.ModuleList()
        
        for i in range(len(layer_sizes) - 1):
            # 对不同层使用不同的STDP参数以提高学习效果
            if i == 0:  # 输入层
                # 第一层使用稍大的学习率以快速捕获输入特征
                stdp = ScientificSTDPLearning(a_plus=0.015, a_minus=0.012)
            elif i == len(layer_sizes) - 2:  # 输出层
                # 输出层使用较小的学习率以稳定预测
                stdp = ScientificSTDPLearning(a_plus=0.008, a_minus=0.010)
            else:  # 隐藏层
                # 隐藏层使用标准学习率
                stdp = ScientificSTDPLearning()
            
            # 创建突触层，传递神经元参数
            layer = ScientificSNNSynapticLayer(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i+1],
                stdplearning=stdp,
                tau=tau,
                v_threshold=v_threshold,
                v_reset=v_reset,
                adaptation_rate=adaptation_rate
            )
            
            # 标记输出层以支持监督学习
            if i == len(layer_sizes) - 2:
                layer.is_output_layer = True
            
            self.layers.append(layer)
    
    def reset(self):
        """重置所有层的状态"""
        for layer in self.layers:
            if hasattr(layer, 'reset'):
                layer.reset()
    
    def reset_state(self):
        """重置所有层的状态（兼容方法）"""
        self.reset()
    
    def forward(self, input_spikes, t=0, training=None, targets=None):
        """前向传播，支持监督STDP"""
        try:
            # 使用外部提供的training标志或默认值
            training_mode = training if training is not None else self.training
            
            # 确保输入是tensor并在正确设备上
            if not isinstance(input_spikes, torch.Tensor):
                input_spikes = torch.tensor(input_spikes, dtype=torch.float32)
            
            x = input_spikes
            
            # 逐层处理
            for i, layer in enumerate(self.layers):
                # 对于输出层，传递targets信息
                if i == len(self.layers) - 1 and targets is not None and training_mode:
                    x = layer(x, t, training=training_mode, targets=targets)
                else:
                    x = layer(x, t, training=training_mode)
            
            return x
        except Exception as e:
            print(f"SNN forward error: {e}")
            # 返回合适形状的零张量
            device = input_spikes.device if isinstance(input_spikes, torch.Tensor) else torch.device('cpu')
            if input_spikes.dim() > 1:
                return torch.zeros((input_spikes.size(0), self.layers[-1].out_features), 
                                  device=device, dtype=torch.float32)
            else:
                return torch.zeros((self.layers[-1].out_features,), device=device, dtype=torch.float32)
    
    def update_stdp_parameters(self, ltp_params, ltd_params):
        """更新所有层的STDP学习参数"""
        for layer in self.layers:
            if hasattr(layer, 'update_stdp_parameters'):
                layer.update_stdp_parameters(ltp_params, ltd_params)
    
    def get_stdp_parameters(self, layer_index=None):
        """获取模型中指定层的STDP参数"""
        params = {}
        
        if layer_index is not None:
            # 获取特定层的参数
            if 0 <= layer_index < len(self.layers) and hasattr(self.layers[layer_index], 'stdplearning'):
                stdp = self.layers[layer_index].stdplearning
                params[f'layer_{layer_index}'] = {
                    'a_plus': stdp.a_plus,
                    'a_minus': stdp.a_minus,
                    'tau_plus': stdp.tau_plus,
                    'tau_minus': stdp.tau_minus,
                    'weight_min': stdp.weight_min,
                    'weight_max': stdp.weight_max
                }
        else:
            # 获取所有层的参数
            for i, layer in enumerate(self.layers):
                if hasattr(layer, 'stdplearning'):
                    stdp = layer.stdplearning
                    params[f'layer_{i}'] = {
                        'a_plus': stdp.a_plus,
                        'a_minus': stdp.a_minus,
                        'tau_plus': stdp.tau_plus,
                        'tau_minus': stdp.tau_minus,
                        'weight_min': stdp.weight_min,
                        'weight_max': stdp.weight_max
                    }
        
        return params
    
    def to(self, device):
        """将模型移动到指定设备"""
        super().to(device)
        # 确保各层正确迁移到设备
        for layer in self.layers:
            layer.to(device)
        return self

class ScientificEncodeInput:
    """科学实现的脉冲编码方法"""
    @staticmethod
    def encode(data, t, training=True, total_time_steps=100, device='cpu'):
        """优化的脉冲编码方法"""
        try:
            # 确保数据类型正确
            if not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32, device=device)
            
            # 转换为浮点数
            data = data.float()
            
            # 处理不同维度的输入
            if data.dim() == 3:  # 单张图像 [channels, height, width]
                data = data.view(-1)  # 展平为一维
            elif data.dim() == 4:  # 批次图像 [batch, channels, height, width]
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
            
            # 计算训练进度
            progress = min(1.0, t / total_time_steps)
            
            # 计算时间衰减因子 - 将progress转换为张量
            progress_tensor = torch.tensor(progress, device=device, dtype=torch.float32)
            time_factor = torch.exp(-3.0 * progress_tensor)
            
            # 生成基于泊松分布的脉冲概率
            min_activation = 0.02  # 最小激活概率
            max_activation = 0.90   # 最大激活概率
            
            # 计算动态激活概率
            if progress < 0.3:  # 训练初期
                # 确保所有乘数都是张量
                factor_tensor = torch.tensor(1.5, device=device, dtype=torch.float32)
                probabilities = min_activation + (max_activation - min_activation) * normalized_data * time_factor * factor_tensor
            elif progress < 0.7:  # 训练中期
                probabilities = min_activation + (max_activation - min_activation) * normalized_data
            else:  # 训练后期
                # 确保乘数是张量
                factor_tensor = torch.tensor(0.7, device=device, dtype=torch.float32)
                probabilities = min_activation + factor_tensor * (max_activation - min_activation) * normalized_data
            
            # 确保概率在有效范围内
            probabilities = torch.clamp(probabilities, 0.0, 1.0)
            
            # 生成二进制脉冲
            with torch.no_grad():
                spikes = torch.bernoulli(probabilities)
            
            # 仅在训练初期添加少量噪声
            if training and progress < 0.5:
                # 将噪声参数转换为张量
                noise_level_tensor = torch.tensor(0.005 * (1.0 - 2.0 * progress), device=device, dtype=torch.float32)
                noise = torch.bernoulli(torch.full_like(data, noise_level_tensor))
                spikes = spikes | (noise > 0)
            
            return spikes.float()
            
        except Exception as e:
            print(f"编码错误: {e}")
            return torch.zeros_like(data, dtype=torch.float32, device=device)

class ScientificSTDPTrainingManager:
    """科学实现的STDP训练管理器"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.best_acc = 0
        self.best_model_state = None
        self.training_history = {
            'train_acc': [],
            'test_acc': [],
            'epoch_times': []
        }
    
    def set_params_from_curves(self, ltp_params, ltd_params):
        """从拟合的LTP/LTD曲线设置STDP参数
        
        Args:
            ltp_params (dict): 从LTP曲线提取的参数，必须包含'a_plus'和'tau_plus'
            ltd_params (dict): 从LTD曲线提取的参数，必须包含'a_minus'和'tau_minus'
        """
        # 验证参数完整性
        required_ltp = ['a_plus', 'tau_plus']
        required_ltd = ['a_minus', 'tau_minus']
        
        for param in required_ltp:
            if param not in ltp_params:
                raise ValueError(f"LTP参数缺少必要项: {param}")
        for param in required_ltd:
            if param not in ltd_params:
                raise ValueError(f"LTD参数缺少必要项: {param}")
        
        # 应用参数到所有层
        if hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                if hasattr(layer, 'stdplearning'):
                    # 设置基本参数
                    layer.stdplearning.a_plus = ltp_params['a_plus']
                    layer.stdplearning.tau_plus = ltp_params['tau_plus']
                    layer.stdplearning.a_minus = ltd_params['a_minus']
                    layer.stdplearning.tau_minus = ltd_params['tau_minus']
                    
                    # 设置可选参数
                    if 'weight_min' in ltp_params:
                        layer.stdplearning.weight_min = ltp_params['weight_min']
                    if 'weight_max' in ltp_params:
                        layer.stdplearning.weight_max = ltp_params['weight_max']
                    if 'weight_decay' in ltp_params:
                        layer.stdplearning.weight_decay = ltp_params['weight_decay']
        
        print(f"已从LTP/LTD曲线设置STDP参数: LTP(a_plus={ltp_params['a_plus']}, tau_plus={ltp_params['tau_plus']}), "
              f"LTD(a_minus={ltd_params['a_minus']}, tau_minus={ltd_params['tau_minus']})"),
    
    def extract_params_from_curve_data(self, ltp_data, ltd_data):
        """从曲线数据中提取STDP参数
        
        Args:
            ltp_data (dict): 包含'delta_t'和'weight_change'的LTP曲线数据
            ltd_data (dict): 包含'delta_t'和'weight_change'的LTD曲线数据
            
        Returns:
            tuple: (ltp_params, ltd_params) - 提取的参数字典
        """
        import numpy as np
        from scipy.optimize import curve_fit
        
        # 指数拟合函数
        def exponential_func(x, a, tau):
            return a * np.exp(-x / tau)
        
        # 提取LTP参数
        ltp_delta_t = np.array(ltp_data['delta_t'])
        ltp_weight_change = np.array(ltp_data['weight_change'])
        
        # 确保只使用正的时间差和正的权重变化
        valid_ltp = (ltp_delta_t > 0) & (ltp_weight_change > 0)
        if not np.any(valid_ltp):
            raise ValueError("LTP曲线数据无效，需要正的时间差和正的权重变化")
        
        # 拟合LTP曲线
        ltp_popt, _ = curve_fit(exponential_func, ltp_delta_t[valid_ltp], 
                               ltp_weight_change[valid_ltp], 
                               bounds=([0.0, 0.1], [1.0, 100.0]))
        
        # 提取LTD参数
        ltd_delta_t = np.array(ltd_data['delta_t'])
        ltd_weight_change = np.array(ltd_data['weight_change'])
        
        # 确保只使用正的时间差和负的权重变化，并取绝对值进行拟合
        valid_ltd = (ltd_delta_t > 0) & (ltd_weight_change < 0)
        if not np.any(valid_ltd):
            raise ValueError("LTD曲线数据无效，需要正的时间差和负的权重变化")
        
        # 拟合LTD曲线（使用权重变化的绝对值）
        ltd_popt, _ = curve_fit(exponential_func, ltd_delta_t[valid_ltd], 
                               np.abs(ltd_weight_change[valid_ltd]),
                               bounds=([0.0, 0.1], [1.0, 100.0]))
        
        # 构建参数字典
        ltp_params = {
            'a_plus': float(ltp_popt[0]),
            'tau_plus': float(ltp_popt[1])
        }
        ltd_params = {
            'a_minus': float(ltd_popt[0]),
            'tau_minus': float(ltd_popt[1])
        }
        
        return ltp_params, ltd_params
    
    def train_epoch_stdp(self, train_loader, epoch, num_epochs, log_callback=None):
        """使用科学的STDP进行训练的epoch方法"""
        self.model.train()
        
        # 跟踪准确率和训练指标
        correct = 0
        total = 0
        running_loss = 0.0
        spike_rates = {}
        
        # 基于训练进度动态更新STDP参数
        progress = epoch / max(1, num_epochs)
        
        # 科学的参数调度策略 - 增加初始学习率以促进早期学习
        if progress < 0.2:  # 早期训练：快速探索
            base_a_plus = 0.05  # 大幅增加初始LTP学习率
            base_a_minus = 0.04  # 大幅增加初始LTD学习率
            weight_decay = 0.00001  # 降低早期权重衰减
        elif progress < 0.7:  # 中期训练：稳定学习
            base_a_plus = 0.02
            base_a_minus = 0.018
            weight_decay = 0.0001
        else:  # 后期训练：精细调整
            base_a_plus = 0.01
            base_a_minus = 0.009
            weight_decay = 0.0005
        
        # 时间步数动态调整
        default_time_steps = getattr(self.model, 'time_steps', 100)
        if progress < 0.3:
            effective_time_steps = max(50, int(default_time_steps * 0.7))
        elif progress < 0.7:
            effective_time_steps = max(80, int(default_time_steps * 0.9))
        else:
            effective_time_steps = default_time_steps
        
        # 更新每层的STDP参数
        if hasattr(self.model, 'layers'):
            updated_layers = 0
            for layer_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'stdplearning'):
                    # 为不同层设置不同的参数
                    if layer_idx == 0:  # 输入层
                        a_plus = base_a_plus * 1.2
                        a_minus = base_a_minus * 0.9
                    elif layer_idx == len(self.model.layers) - 1:  # 输出层
                        a_plus = base_a_plus * 0.9
                        a_minus = base_a_minus * 1.2
                    else:  # 隐藏层
                        a_plus = base_a_plus
                        a_minus = base_a_minus
                    
                    # 确保update_parameters方法存在
                    if not hasattr(layer.stdplearning, 'update_parameters'):
                        def default_update_parameters(self, epoch, total_epochs):
                            # 科学的参数衰减策略
                            progress = epoch / total_epochs
                            self.a_plus = self.initial_a_plus * np.exp(-progress * 3)
                            self.a_minus = self.initial_a_minus * np.exp(-progress * 3)
                            self.tau_plus = min(30.0, self.initial_tau_plus + progress * 20)
                            self.tau_minus = min(30.0, self.initial_tau_minus + progress * 20)
                            if hasattr(self, 'weight_decay'):
                                self.weight_decay = 0.0001 + progress * 0.0009
                        
                        # 动态添加方法
                        setattr(layer.stdplearning, 'update_parameters', types.MethodType(default_update_parameters, layer.stdplearning))
                        
                        # 存储初始参数供衰减使用
                        setattr(layer.stdplearning, 'initial_a_plus', a_plus)
                        setattr(layer.stdplearning, 'initial_a_minus', a_minus)
                        setattr(layer.stdplearning, 'initial_tau_plus', 20.0)
                        setattr(layer.stdplearning, 'initial_tau_minus', 20.0)
                        setattr(layer.stdplearning, 'weight_decay', weight_decay)
                    
                    # 初始化或更新参数
                    layer.stdplearning.a_plus = a_plus
                    layer.stdplearning.a_minus = a_minus
                    layer.stdplearning.weight_decay = weight_decay
                    
                    # 调用动态参数更新方法
                    try:
                        layer.stdplearning.update_parameters(epoch, num_epochs)
                        updated_layers += 1
                    except Exception as e:
                        if log_callback:
                            log_callback(f"警告: 层{layer_idx}参数更新失败: {str(e)}")
            
            if log_callback and updated_layers > 0:
                log_callback(f"成功更新了{updated_layers}/{len(self.model.layers)}层的STDP参数")
        
        # 主训练循环
        start_time = time.time()
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                # 重置网络状态
                self.model.reset_state()
                
                # 确保将数据移动到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 累积输出
                outputs_accumulated = None
                
                # 处理每个时间步
                for t in range(effective_time_steps):
                    # 编码输入
                    encoded_inputs = ScientificEncodeInput.encode(
                        inputs, 
                        t, 
                        training=True, 
                        total_time_steps=effective_time_steps,
                        device=self.device
                    )
                    
                    # 前向传播
                    outputs = self.model(encoded_inputs, t, training=True)
                    
                    # 累加输出
                    if outputs_accumulated is None:
                        outputs_accumulated = outputs.float()
                    else:
                        outputs_accumulated += outputs.float()
                
                # 计算预测和准确率
                if outputs_accumulated is not None:
                    _, predicted = outputs_accumulated.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    # 计算简单损失用于监控
                    try:
                        loss = torch.nn.functional.cross_entropy(
                            outputs_accumulated / effective_time_steps, 
                            targets
                        )
                        running_loss += loss.item()
                    except:
                        pass
                
                # 计算当前准确率
                current_acc = 100. * correct / total if total > 0 else 0.0
                
                # 打印训练进度
                if batch_idx % 100 == 0 and log_callback:
                    avg_loss = running_loss / (batch_idx + 1) if batch_idx > 0 else 0
                    log_callback(f'Epoch {epoch+1}/{num_epochs} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ' \
                              f'({100. * batch_idx / len(train_loader):.0f}%)]\t' \
                              f'准确率: {current_acc:.2f}%\t' \
                              f'损失: {avg_loss:.4f}\t' \
                              f'时间步数: {effective_time_steps}')
                
            except Exception as e:
                if log_callback:
                    log_callback(f"批次{batch_idx}训练错误: {str(e)}")
                continue
        
        # 计算最终准确率
        final_accuracy = 100. * correct / total if total > 0 else 0.0
        final_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_time = time.time() - start_time
        
        # 更新历史记录
        self.training_history['train_acc'].append(final_accuracy)
        self.training_history['epoch_times'].append(epoch_time)
        
        # 保存最佳模型
        if final_accuracy > self.best_acc:
            self.best_acc = final_accuracy
            self.best_model_state = self.model.state_dict()
        
        if log_callback:
            log_callback(f'Epoch {epoch+1}/{num_epochs} 完成: 准确率 = {final_accuracy:.2f}%, 时间 = {epoch_time:.2f}秒')
        
        return final_loss, final_accuracy
    
    def evaluate(self, test_loader, log_callback=None):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        # 获取时间步数
        time_steps = getattr(self.model, 'time_steps', 100)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader, desc="评估")):
                # 确保将数据移动到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 重置网络状态
                self.model.reset_state()
                
                # 累积输出
                outputs_accumulated = None
                
                # 处理每个时间步
                for t in range(time_steps):
                    # 编码输入
                    encoded_inputs = ScientificEncodeInput.encode(
                        inputs, 
                        t, 
                        training=False, 
                        total_time_steps=time_steps,
                        device=self.device
                    )
                    
                    # 前向传播
                    outputs = self.model(encoded_inputs, t, training=False)
                    
                    # 累加输出
                    if outputs_accumulated is None:
                        outputs_accumulated = outputs.float()
                    else:
                        outputs_accumulated += outputs.float()
                
                # 计算预测和准确率
                if outputs_accumulated is not None:
                    _, predicted = outputs_accumulated.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
        
        # 计算最终准确率
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        # 更新测试准确率历史记录
        self.training_history['test_acc'].append(accuracy)
        
        if log_callback:
            log_callback(f'测试准确率: {accuracy:.2f}%')
        
        return accuracy
    
    def train(self, train_loader, test_loader, num_epochs, log_callback=None):
        """完整训练流程"""
        if log_callback:
            log_callback(f"开始训练，总轮次: {num_epochs}")
        
        for epoch in range(num_epochs):
            # 训练一个epoch
            loss, train_acc = self.train_epoch_stdp(
                train_loader, epoch, num_epochs, log_callback
            )
            
            # 评估
            test_acc = self.evaluate(test_loader, log_callback)
            
            if log_callback:
                log_callback(f"Epoch {epoch+1}/{num_epochs} 总结: 训练准确率 = {train_acc:.2f}%, 测试准确率 = {test_acc:.2f}%")
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            if log_callback:
                log_callback(f"训练完成，最佳准确率: {self.best_acc:.2f}%")
        
        return self.training_history

# 使用示例
def example_usage():
    # 示例：如何使用科学实现的SNN和训练管理器
    print("科学SNN训练框架已加载")
    print("使用方法：")
    print("1. 创建模型: model = ScientificSNN(input_dim=784, output_dim=10, hidden_layers=3, hidden_neurons=512)")
    print("2. 创建训练管理器: trainer = ScientificSTDPTrainingManager(model)")
    print("3. 训练模型: trainer.train(train_loader, test_loader, num_epochs=10)")

# 运行示例
if __name__ == "__main__":
    example_usage()