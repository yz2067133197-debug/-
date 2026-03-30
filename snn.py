# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class LIFSpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        out = (input > 0).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        mask = (input.abs() < 1.0).float()
        return grad_input * mask

lif_spike = LIFSpikeFunction.apply

class LIFLayer(nn.Module):
    def __init__(self, input_dim, output_dim, tau=20.0, dt=1.0, v_threshold=1.0, v_reset=0.0):
        super(LIFLayer, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau = tau
        self.dt = dt

    def forward(self, input_spikes):
        # input_spikes: [time_steps, batch, input_dim]
        time_steps = input_spikes.shape[0]
        batch_size = input_spikes.shape[1]
        device = input_spikes.device
        
        # Inject debug flag into layer if not present
        if not hasattr(self, 'debug_print'):
             # Check if parent model has debug flag, but we don't have easy access to parent here
             # We'll rely on a global or passed flag, or just check debug_count if we can find it
             pass

        v = torch.zeros(batch_size, self.fc.out_features, device=device)
        outputs = []
        for t in range(time_steps):
            in_t = input_spikes[t]
            I = self.fc(in_t)
            
            # DEBUG: Print mean input current for first layer
            # Use self.layer_index which is set in SNN.forward
            if hasattr(self, 'debug_print') and self.debug_print and hasattr(self, 'layer_index') and self.layer_index == 0 and t == 10:
                print(f"DEBUG: Layer {self.layer_index} Mean Current I: {I.mean().item():.6f}, Max I: {I.max().item():.6f}")
                
            dv = (-v/self.tau + I) * self.dt
            v = v + dv
            spikes = lif_spike(v - self.v_threshold)
            v = torch.where(spikes > 0, torch.ones_like(v)*self.v_reset, v)
            outputs.append(spikes)
        return torch.stack(outputs, dim=0)

class SNN(nn.Module):
    def __init__(self,
                 input_dim=28*28,
                 output_dim=10,
                 hidden_layers=1,
                 hidden_neurons=100,
                 tau=20.0,
                 time_steps=50,
                 v_threshold=1.0,
                 v_reset=0.0,
                 firing_rate=20.0,
                 use_synaptic_data=False,
                 synaptic_data_dim=100):
        super(SNN, self).__init__()
        self.time_steps = time_steps
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.firing_rate = firing_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_synaptic_data = use_synaptic_data
        self.synaptic_data_dim = synaptic_data_dim
        self.hidden_layers = hidden_layers
        self.hidden_neurons = hidden_neurons
        
        self.debug_count = 0

        # 如果使用突触数据，调整输入维度
        self.effective_input_dim = input_dim
        if use_synaptic_data:
            # 添加处理突触数据的投影层
            # LTP 投影层 (Excitatory)
            self.synaptic_projection_ltp = nn.Linear(synaptic_data_dim, input_dim)
            # LTD 投影层 (Inhibitory)
            self.synaptic_projection_ltd = nn.Linear(synaptic_data_dim, input_dim)
            
            # 不再拼接，而是使用门控机制融合特征
            self.effective_input_dim = input_dim  # 保持输入维度不变

        layers = []
        in_dim = self.effective_input_dim  # 使用调整后的输入维度

        for _ in range(hidden_layers):
            layers.append(LIFLayer(in_dim, hidden_neurons, tau=self.tau, 
                                 v_threshold=self.v_threshold, v_reset=self.v_reset))
            in_dim = hidden_neurons
        
        layers.append(LIFLayer(in_dim, output_dim, tau=self.tau, 
                             v_threshold=self.v_threshold, v_reset=self.v_reset))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, synaptic_data=None, ltd_data=None):
        # x: [batch, 1, 28, 28], 范围在[-1,1]
        batch_size = x.shape[0]
        
        # 展平输入并归一化到[0,1]
        x_flat = x.view(batch_size, -1)  # [batch, 784]
        pixel_vals = (x_flat + 1.0) / 2.0  # [0,1]
        
        # 如果使用突触数据，则进行特征融合
        if self.use_synaptic_data and synaptic_data is not None:
            # 检查synaptic_data是否为元组 (ltp_data, ltd_data)
            if isinstance(synaptic_data, tuple) or isinstance(synaptic_data, list):
                if len(synaptic_data) >= 2:
                    ltd_data = synaptic_data[1]
                    synaptic_data = synaptic_data[0]
                else:
                    synaptic_data = synaptic_data[0]

            # 确保突触数据是2D [batch, features]
            if len(synaptic_data.shape) == 1:
                synaptic_data = synaptic_data.unsqueeze(0)
            
            # 如果没有提供LTD数据，使用零填充（兼容旧代码）
            if ltd_data is None:
                ltd_data = torch.zeros_like(synaptic_data)
            elif len(ltd_data.shape) == 1:
                ltd_data = ltd_data.unsqueeze(0)
            
            # 确保设备一致
            if synaptic_data.device != x.device:
                synaptic_data = synaptic_data.to(x.device)
            if ltd_data.device != x.device:
                ltd_data = ltd_data.to(x.device)
            
            # 确保批大小匹配
            if synaptic_data.shape[0] == 1 and batch_size > 1:
                synaptic_data = synaptic_data.expand(batch_size, -1)
                ltd_data = ltd_data.expand(batch_size, -1)
            elif synaptic_data.shape[0] != batch_size:
                raise ValueError(f"Batch size mismatch: input {batch_size} vs synaptic_data {synaptic_data.shape[0]}")
            
            # 投影LTP和LTD数据到输入空间
            # LTP作为兴奋性调节
            ltp_proj = self.synaptic_projection_ltp(synaptic_data)
            # LTD作为抑制性调节
            ltd_proj = self.synaptic_projection_ltd(ltd_data)
            
            # 差分门控机制：Gate = Sigmoid(LTP_effect - LTD_effect)
            # 这模拟了突触权重的净变化对输入增益的影响
            net_synaptic_effect = ltp_proj - ltd_proj
            gate = torch.sigmoid(net_synaptic_effect)  # [batch, input_dim]
            
            pixel_vals = pixel_vals * gate  # [batch, input_dim]

        # 计算发放概率
        # 增加基础发放率增益，防止输入过弱
        firing_prob = pixel_vals * (self.firing_rate / 1000.0) * 2.0 
        firing_prob = torch.clamp(firing_prob, 0.0, 1.0)  # 确保在[0,1]

        # 对firing_prob进行伯努利采样
        firing_prob_expanded = firing_prob.unsqueeze(0).expand(self.time_steps, batch_size, self.input_dim)
        spike_train = torch.bernoulli(firing_prob_expanded)

        out = spike_train.to(x.device)
        
        # DEBUG: Print input spike rate (Print first few batches)
        if hasattr(self, 'debug_count') and self.debug_count < 5:
           print(f"DEBUG: Input spike rate: {out.mean().item():.4f}")

        for i, layer in enumerate(self.layers):
            # Pass debug flag to layer
            layer.layer_index = i
            if hasattr(self, 'debug_count') and self.debug_count < 5:
                layer.debug_print = True
            else:
                layer.debug_print = False
                
            out = layer(out)
            # DEBUG: Print layer spike rate
            if hasattr(self, 'debug_count') and self.debug_count < 5:
               print(f"DEBUG: Layer {i} spike rate: {out.mean().item():.4f}")
        
        if not hasattr(self, 'debug_count'):
            self.debug_count = 0
        self.debug_count = (self.debug_count + 1) % 1000
                
        out_sum = out.sum(dim=0)  # [batch, output_dim]
        
        # DEBUG: Print first sample output counts
        if self.debug_count < 5:
            print(f"DEBUG: Output counts (sample 0): {out_sum[0].tolist()}")

        return out_sum
