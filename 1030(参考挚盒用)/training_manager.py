import time
import torch
import numpy as np
import types
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from snn_scientific_implementation import ScientificSNN, ScientificSTDPTrainingManager, ScientificSTDPLearning


class STDPTrainingConfig:
    """STDP训练配置类"""

    def __init__(self, **kwargs):
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.train_batches = kwargs.get('train_batches', 1000)
        self.test_batches = kwargs.get('test_batches', 200)
        self.hidden_layers = kwargs.get('hidden_layers', 3)
        self.hidden_neurons = kwargs.get('hidden_neurons', 512)
        self.tau = kwargs.get('tau', 10.0)
        self.v_threshold = kwargs.get('v_threshold', 0.5)
        self.v_reset = kwargs.get('v_reset', 0.0)
        self.time_steps = kwargs.get('time_steps', 128)
        # STDP特定参数
        self.a_plus = kwargs.get('a_plus', 0.01)
        self.a_minus = kwargs.get('a_minus', 0.01)
        self.tau_plus = kwargs.get('tau_plus', 20.0)
        self.tau_minus = kwargs.get('tau_minus', 20.0)
        self.weight_min = kwargs.get('weight_min', 0.0)
        self.weight_max = kwargs.get('weight_max', 1.0)


class SynapticDataTransformer:
    def __init__(self, target_dim, use_pca=True):
        """
        突触数据维度转换器
        
        参数:
            target_dim: 目标维度
            use_pca: 是否使用PCA进行降维，如果为False则使用截断
        """
        self.target_dim = target_dim
        self.use_pca = use_pca
        self.pca = None
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, synaptic_data):
        """拟合转换器"""
        if self.use_pca:
            # 确保数据在CPU上且是numpy数组
            if torch.is_tensor(synaptic_data):
                synaptic_data = synaptic_data.cpu().numpy()
            
            # 确保数据是2D的 [n_samples, n_features]
            if len(synaptic_data.shape) == 1:
                synaptic_data = synaptic_data.reshape(1, -1)
            elif len(synaptic_data.shape) > 2:
                raise ValueError(f"输入数据维度 {synaptic_data.shape} 不支持，期望1D或2D数组")
                
            # 标准化数据
            self.scaler.fit(synaptic_data)
            scaled_data = self.scaler.transform(synaptic_data)
            
            # 训练PCA
            self.pca = PCA(n_components=min(self.target_dim, scaled_data.shape[1]))
            self.pca.fit(scaled_data)
            
            # 检查是否达到目标维度
            if self.pca.n_components_ < self.target_dim:
                print(f"警告: 目标维度 {self.target_dim} 大于数据特征数 {self.pca.n_components_}")
                
            self.fitted = True
        return self
    
    def transform(self, synaptic_data):
        """转换数据"""
        if not self.fitted:
            raise RuntimeError("转换器尚未拟合，请先调用 fit() 方法")
            
        # 保存原始输入是否是tensor
        is_tensor = torch.is_tensor(synaptic_data)
        
        # 转换为numpy数组并确保是2D
        if is_tensor:
            device = synaptic_data.device
            original_shape = synaptic_data.shape
            
            # 展平除batch外的所有维度
            if len(original_shape) > 2:
                synaptic_data = synaptic_data.view(original_shape[0], -1)
            elif len(original_shape) == 1:
                # 如果是1D，转换为2D [1, features]
                synaptic_data = synaptic_data.unsqueeze(0)
                
            numpy_data = synaptic_data.cpu().numpy()
        else:
            device = None
            numpy_data = np.asarray(synaptic_data)
            if numpy_data.ndim == 1:
                # 如果是1D，转换为2D [1, features]
                numpy_data = numpy_data.reshape(1, -1)
            elif numpy_data.ndim > 2:
                # 展平除batch外的所有维度
                numpy_data = numpy_data.reshape(numpy_data.shape[0], -1)
        
        # 检查数据维度
        if numpy_data.ndim != 2:
            raise ValueError(f"输入数据维度 {numpy_data.shape} 不支持，期望1D或2D数组")
        
        try:
            # 标准化
            scaled_data = self.scaler.transform(numpy_data)
            
            if self.use_pca and self.pca is not None:
                # PCA转换
                transformed = self.pca.transform(scaled_data)
            else:
                # 简单截断或填充
                if scaled_data.shape[1] > self.target_dim:
                    transformed = scaled_data[:, :self.target_dim]
                else:
                    # 如果维度不足，用零填充
                    transformed = np.zeros((scaled_data.shape[0], self.target_dim), dtype=scaled_data.dtype)
                    transformed[:, :scaled_data.shape[1]] = scaled_data
            
            # 转换回原始格式
            if is_tensor:
                return torch.tensor(transformed, dtype=torch.float32, device=device)
            return transformed
            
        except Exception as e:
            error_msg = f"转换数据时出错: {str(e)}. 输入形状: {numpy_data.shape}, 目标维度: {self.target_dim}"
            if hasattr(self, 'pca') and self.pca is not None:
                error_msg += f", PCA组件数: {self.pca.n_components_}"
            raise RuntimeError(error_msg) from e


class TrainingManager:
    """STDP训练管理器类，协调脉冲神经网络训练过程"""
    def __init__(self, dataset_manager, model, device):
        self.dataset_manager = dataset_manager
        self.model = model.to(device)
        self.device = device
        self.confusion_matrix = np.zeros((10, 10))
        self.weight_matrices = []
        self.training_history = {
            'train_acc': [],
            'test_acc': [],
            'epoch_times': []
        }
        self.best_acc = 0
        self.best_model_state = None
        self.synaptic_data = None  # 存储突触数据
        self.synaptic_transformer = None  # 突触数据转换器
        self.log_callback = None  # 日志回调函数
        self.synaptic_parameters = None  # 存储LTP/LTD拟合参数
        self.update_stdp_frequency = 1   # STDP参数更新频率
        self.optimizer = None  # 移除优化器依赖，STDP不需要优化器
    
    def set_synaptic_parameters(self, synaptic_parameters):
        """
        设置突触可塑性参数（LTP/LTD拟合结果）
        """
        self.synaptic_parameters = synaptic_parameters
        # 立即应用一次参数更新
        if hasattr(self.model, 'update_stdp_parameters'):
            self.model.update_stdp_parameters(
                synaptic_parameters, 
                synaptic_parameters
            )
            if self.log_callback:
                self.log_callback(f"已更新STDP参数: {synaptic_parameters}")

    def set_synaptic_data(self, synaptic_data, use_pca=False):  # 默认禁用PCA
        """
        设置突触数据并自动调整维度
        
        参数:
            synaptic_data: 突触数据张量或numpy数组
            use_pca: 是否使用PCA进行降维，如果为False则使用截断
        """
        if not isinstance(synaptic_data, torch.Tensor):
            synaptic_data = torch.tensor(synaptic_data, dtype=torch.float32, device=self.device)
        else:
            synaptic_data = synaptic_data.to(device=self.device, dtype=torch.float32)
        
        # 确保是2D张量 [batch_size, features]
        if len(synaptic_data.shape) == 1:
            synaptic_data = synaptic_data.unsqueeze(0)  # [features] -> [1, features]
        elif len(synaptic_data.shape) > 2:
            # 展平除batch外的所有维度
            synaptic_data = synaptic_data.view(synaptic_data.shape[0], -1)
            print(f"警告: 输入数据维度 {synaptic_data.shape} 被展平为 2D")
            
        self.synaptic_data = synaptic_data
        self.synaptic_transformer = None  # 重置转换器
        
        # 如果模型使用突触数据，则初始化转换器
        if hasattr(self.model, 'use_synaptic_data') and self.model.use_synaptic_data:
            target_dim = getattr(self.model, 'synaptic_data_dim', 0)
            if target_dim > 0:
                # 对于单个样本，总是使用截断方法
                if len(synaptic_data) == 1:
                    use_pca = False
                    print("检测到单个样本，自动使用截断方法")
                
                # 初始化转换器
                self.synaptic_transformer = SynapticDataTransformer(
                    target_dim=target_dim,
                    use_pca=use_pca
                )
                
                # 在CPU上拟合转换器
                try:
                    self.synaptic_transformer.fit(synaptic_data.cpu())
                    print(f"成功初始化突触数据转换器，使用{'PCA' if use_pca else '截断'}方法")
                except Exception as e:
                    print(f"警告: 无法拟合转换器: {str(e)}，将使用原始数据")
                    self.synaptic_transformer = None
            else:
                print("警告: 模型的突触数据维度为0，将使用原始数据")
        else:
            print("警告: 模型未启用突触数据，将忽略提供的突触数据")
    
    def get_batch_synaptic_data(self, batch_size):
        """
        获取一批处理后的突触数据
        
        参数:
            batch_size: 批大小
            
        返回:
            处理后的突触数据，形状为 [batch_size, synaptic_data_dim]
        """
        if self.synaptic_data is None:
            print("警告: 未设置突触数据，返回None")
            return None
        
        # 确保数据在CPU上处理
        if self.synaptic_data.is_cuda:
            device = self.synaptic_data.device
            data_cpu = self.synaptic_data.cpu()
        else:
            device = None
            data_cpu = self.synaptic_data
            
        # 随机选择样本
        idx = torch.randint(0, len(data_cpu), (batch_size,))
        batch_data = data_cpu[idx]
        
        # 如果有转换器且已拟合，则进行维度转换
        if self.synaptic_transformer is not None:
            if hasattr(self.synaptic_transformer, 'fitted') and self.synaptic_transformer.fitted:
                try:
                    batch_data = self.synaptic_transformer.transform(batch_data)
                    # 确保返回的是tensor
                    if not torch.is_tensor(batch_data):
                        batch_data = torch.tensor(batch_data, dtype=torch.float32, device=self.device)
                except Exception as e:
                    print(f"警告: 突触数据转换失败: {str(e)}，使用原始数据")
            else:
                print("警告: 突触数据转换器尚未拟合，使用原始数据")
        
        # 移回原设备
        if device is not None:
            batch_data = batch_data.to(device)
            
        return batch_data

    def train_epoch_stdp(self, train_loader, device, epoch, num_epochs, log_callback=None):
        """使用STDP进行训练的epoch方法 - 优化版本"""
        self.model.train()
        
        # 跟踪准确率（评估用）
        correct = 0
        total = 0
        
        # 基于训练进度动态更新STDP参数（每轮次更新一次）
        if hasattr(self.model, 'layers'):
            # 跟踪未更新的层
            updated_layers = 0
            for layer_idx, layer in enumerate(self.model.layers):
                # 增强接口校验，确保所有需要STDP的层都能更新参数
                if hasattr(layer, 'stdplearning'):
                    # 检查update_parameters方法，若无则创建默认实现
                    if not hasattr(layer.stdplearning, 'update_parameters'):
                        # 为缺失update_parameters方法的层添加默认实现
                        def default_update_parameters(self, epoch, total_epochs):
                            # 简单的线性衰减学习率
                            progress = epoch / total_epochs
                            self.a_plus = max(0.005, 0.15 * (1 - progress))
                            self.a_minus = max(0.006, 0.18 * (1 - progress))
                            self.tau_plus = min(15.0, 5.0 + progress * 10)
                            self.tau_minus = min(18.0, 7.0 + progress * 11)
                        
                        # 动态添加方法
                        setattr(layer.stdplearning, 'update_parameters', types.MethodType(default_update_parameters, layer.stdplearning))
                        
                        if log_callback:
                            log_callback(f"警告: 层{layer_idx}缺少update_parameters方法，已添加默认实现")
                    
                    # 初始化STDP参数为较大值以促进学习
                    if epoch == 0:
                        layer.stdplearning.a_plus = 0.15  # 增大初始LTP学习率
                        layer.stdplearning.a_minus = 0.18  # 增大初始LTD学习率
                        layer.stdplearning.tau_plus = 5.0
                        layer.stdplearning.tau_minus = 7.0
                        
                    # 调用动态参数更新方法
                    try:
                        layer.stdplearning.update_parameters(epoch, num_epochs)
                        updated_layers += 1
                        if log_callback and epoch % 2 == 0:  # 每2轮记录一次参数状态
                            log_callback(f"STDP参数更新: epoch={epoch}, 层={layer_idx}, "
                                       f"a_plus={layer.stdplearning.a_plus:.4f}, a_minus={layer.stdplearning.a_minus:.4f}")
                    except Exception as e:
                        if log_callback:
                            log_callback(f"警告: 层{layer_idx}参数更新失败: {str(e)}")
            
            if log_callback and updated_layers > 0:
                log_callback(f"成功更新了{updated_layers}/{len(self.model.layers)}层的STDP参数")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # 重置网络状态
            self.model.reset_state()
            
            # 确保将数据移动到设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取当前batch的突触数据（如果需要）
            batch_synaptic_data = self.get_batch_synaptic_data(inputs.size(0))
            # 直接为每层更新STDP参数
            if batch_synaptic_data is not None:
                for layer in self.model.layers:
                    if hasattr(layer, 'update_stdp_parameters'):
                        layer.update_stdp_parameters(batch_synaptic_data, batch_synaptic_data)
            
            # 处理每个时间步
            time_steps = getattr(self.model, 'time_steps', 20)
            outputs_accumulated = None
            
            for t in range(time_steps):  # 确保t从0到time_steps-1的整数
                # 正确传递时间步t和targets给模型以支持监督STDP
                outputs = self.model(inputs, t, training=True, targets=targets)
                
                # 累加输出用于评估
                if outputs_accumulated is None:
                    outputs_accumulated = outputs.float()
                else:
                    outputs_accumulated += outputs.float()
            
            # STDP训练不使用传统损失函数，仅基于脉冲时序进行学习
            
            # 评估当前batch的准确率
            if outputs_accumulated is not None:
                _, predicted = outputs_accumulated.max(1)
            else:
                _, predicted = outputs.max(1)
                
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 计算当前准确率（用于更频繁的反馈）
            current_acc = 100. * correct / total if total > 0 else 0.0
            
            # 每100个batch输出一次训练信息
            if batch_idx % 100 == 0 and log_callback:
                log_callback(f'Train Epoch: {epoch+1}/{num_epochs} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ' \
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: N/A (STDP训练)\t' \
                           f'Current Acc: {current_acc:.2f}%')
        
        # 计算并返回最终的训练准确率
        train_acc = 100. * correct / total if total > 0 else 0.0
        return None, train_acc  # 返回None作为损失值以明确表示STDP训练不使用传统损失
        
    def evaluate(self, test_loader, num_batches=None, synaptic_data=None):
        """评估模型性能（移除传统损失计算）"""
        self.model.eval()
        correct = 0
        total = 0
        total_batches = num_batches if num_batches else len(test_loader)
    
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if num_batches and batch_idx >= num_batches:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 获取当前batch的突触数据（如果需要）
                if synaptic_data is not None:
                    # 如果提供了外部突触数据，确保其维度正确
                    if len(synaptic_data) >= inputs.size(0):
                        batch_synaptic_data = synaptic_data[:inputs.size(0)].to(self.device)
                    else:
                        # 如果外部数据不足，用零填充
                        batch_synaptic_data = torch.zeros((inputs.size(0), synaptic_data.size(1)), 
                                                      device=self.device)
                        batch_synaptic_data[:len(synaptic_data)] = synaptic_data.to(self.device)
                else:
                    # 使用训练管理器中的突触数据
                    batch_synaptic_data = self.get_batch_synaptic_data(inputs.size(0))
                
                # 时间步处理
                time_steps = getattr(self.model, 'time_steps', 20)
                outputs_accumulated = None
                
                for t in range(time_steps):
                    # 将输入数据转换为脉冲序列
                    from snn_scientific_implementation import ScientificEncodeInput
                    encoded_inputs = ScientificEncodeInput.encode(
                        inputs, 
                        t, 
                        training=False, 
                        total_time_steps=time_steps,
                        device=self.device
                    )
                    
                    # 传入时间步，标记训练状态为False
                    outputs = self.model(encoded_inputs, t, training=False)
                    
                    # 累加输出
                    if outputs_accumulated is None:
                        outputs_accumulated = outputs.float()
                    else:
                        outputs_accumulated += outputs.float()
                
                # 统计准确率
                _, predicted = outputs_accumulated.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                    
                # 每100个batch输出一次评估信息
                if batch_idx % 100 == 0 and hasattr(self, 'log_callback') and self.log_callback:
                    self.log_callback(f'Test: [{batch_idx}/{total_batches} ' 
                                   f'({100. * batch_idx / total_batches:.0f}%)]')
        
            # 计算准确率（不使用损失）
            accuracy = 100. * correct / total
            
            return accuracy

        def train(self, config, dataset_name, progress_callback=None, log_callback=None):
            """训练模型"""
            # 设置日志回调
            self.log_callback = log_callback
            
            def log(msg):
                if log_callback:
                    log_callback(msg)
                else:
                    print(msg)
                    
            def adjust_learning_rate(optimizer, epoch, total_epochs, initial_lr):
                """实现学习率预热和余弦退火"""
                # 学习率预热
                warmup_epochs = 5
                if epoch < warmup_epochs:
                    lr = initial_lr * (epoch + 1) / warmup_epochs
                else:
                    # 余弦退火
                    lr = 0.5 * initial_lr * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
                    
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                return lr
            
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            
            log(f"开始训练，使用设备: {self.device}")
            if hasattr(self, 'synaptic_data') and self.synaptic_data is not None:
                log(f"检测到突触数据，形状: {self.synaptic_data.shape}")
                if hasattr(self, 'synaptic_transformer') and self.synaptic_transformer is not None:
                    log(f"使用{'PCA' if self.synaptic_transformer.use_pca else '截断'}方法处理突触数据，目标维度: {self.synaptic_transformer.target_dim}")
        
            train_loader = self.dataset_manager.get_dataloader(
                dataset_name, train=True, batch_size=config.batch_size
            )
            test_loader = self.dataset_manager.get_dataloader(
                dataset_name, train=False, batch_size=config.batch_size
            )
        
            # 初始化记录
            criterion = None  # STDP训练不需要损失函数
            scheduler = None  # STDP训练不需要学习率调度器
                    
            log(f"开始训练，共 {config.epochs} 个epoch，每个epoch {config.train_batches} 个batch，批量大小: {config.batch_size}")
            log(f"学习率: {config.learning_rate}")
        
            for epoch in range(config.epochs):
                self.model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                # 记录epoch开始时间
                epoch_start_time = time.time()
                
                log(f"\nEpoch {epoch+1}/{config.epochs}")
                log("-" * 60)
        
                for batch_idx, (inputs, labels) in enumerate(train_loader):
                    if batch_idx >= config.train_batches:
                        break

                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # STDP训练不需要传统优化器的梯度清零操作

                    # 获取当前batch的突触数据（自动处理维度）
                    batch_synaptic_data = self.get_batch_synaptic_data(inputs.size(0))

                    # 在STDP模式下，前向传播时权重更新会在模型内部自动进行
                    # 处理每个时间步
                    time_steps = getattr(self.model, 'time_steps', 128)
                    outputs_accumulated = None
                    
                    for t in range(time_steps):
                        # 将输入数据转换为脉冲序列
                        from snn_scientific_implementation import ScientificEncodeInput
                        encoded_inputs = ScientificEncodeInput.encode(
                            inputs, 
                            t, 
                            training=True, 
                            total_time_steps=time_steps,
                            device=self.device
                        )
                        
                        # 前向传播，STDP权重更新会在模型内部进行
                        outputs = self.model(encoded_inputs, t, training=True, targets=labels)
                        
                        # 累加输出
                        if outputs_accumulated is None:
                            outputs_accumulated = outputs.float()
                        else:
                            outputs_accumulated += outputs.float()
                        
                        # 使用累加的输出来计算准确率
                        outputs = outputs_accumulated

                        # 统计
                        # STDP训练不使用传统损失函数，设置伪损失值
                        loss = torch.tensor(0.0, device=self.device)
                        total_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                        
                        # 保存训练数据用于可视化
                        self.save_training_data(outputs, labels)
                        
                        # 每100个batch输出一次训练信息
                        if batch_idx % 100 == 0 and log_callback:
                            acc = 100. * correct / total if total > 0 else 0
                            # 计算已处理的样本数和百分比
                            processed_samples = batch_idx * len(inputs)
                            total_samples = len(train_loader.dataset)
                            percentage = 100. * batch_idx / len(train_loader)
                            # 优化日志格式，使其更清晰易读
                            log(f'Epoch: {epoch+1}/{config.epochs} [{processed_samples}/{total_samples} 样本 ({percentage:.0f}%)] ' \
                                f'- 训练准确率: {acc:.2f}%')

                        # 更新进度条
                        if progress_callback:
                            progress = (epoch * config.train_batches + batch_idx + 1) / (config.epochs * config.train_batches)
                            progress_callback(progress)
                
                # 计算epoch的训练指标
                avg_loss = total_loss / len(train_loader)
                train_acc = 100. * correct / total if total > 0 else 0
                
                # 在测试集上评估
                test_acc = self.evaluate(test_loader, num_batches=config.test_batches)
                
                # STDP训练不需要学习率调度器
                # scheduler.step(test_loss)  # 移除
                
                # 记录训练历史
                self.training_history['train_acc'].append(train_acc)
                self.training_history['test_acc'].append(test_acc)
                
                # 计算epoch耗时
                epoch_time = time.time() - epoch_start_time
                self.training_history['epoch_times'].append(epoch_time)
                
                # 输出epoch总结，更详细的格式
                log(f'Epoch {epoch+1}/{config.epochs} 完成')
                log(f'  - 训练准确率: {train_acc:.2f}%')
                log(f'  - 测试准确率: {test_acc:.2f}%')
                log(f'  - 耗时: {epoch_time:.2f}秒')
                
                # 保存最佳模型
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.best_model_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'train_acc': train_acc,
                        'test_acc': test_acc,
                    }
                    if hasattr(self, 'synaptic_transformer') and self.synaptic_transformer is not None:
                        self.best_model_state['synaptic_transformer'] = self.synaptic_transformer
                    
                    log(f'发现新的最佳模型，测试准确率: {test_acc:.2f}%')
            
            # 训练完成
            log(f'\n训练完成，最佳测试准确率: {self.best_acc:.2f}%')
            
            # 恢复最佳模型参数
            if self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state['model_state_dict'])
                log('已加载最佳模型参数')
            
            return self.training_history

        def save_training_data(self, outputs, labels):
            """保存训练过程中的数据用于可视化，优化内存使用"""
            # 更新混淆矩阵
            with torch.no_grad():  # 确保不创建计算图
                pred = outputs.argmax(dim=1).cpu()
                labels = labels.cpu()
                for t, p in zip(labels, pred):
                    self.confusion_matrix[t.item()][p.item()] += 1
        
                # 保存权重矩阵 - 适配SNN模型
                if hasattr(self.model, 'layers'):  # SNN模型有layers属性
                    # 获取最后一层的权重
                    last_layer = self.model.layers[-1]
                    if hasattr(last_layer, 'weights'):  # SNNSynapticLayer有weights属性
                        weight = last_layer.weights.data.cpu().numpy()
                        self.weight_matrices.append(weight)
                elif hasattr(self.model, 'fc'):  # 检查模型是否有全连接层
                    weight = self.model.fc.weight.data.cpu().numpy()
                    self.weight_matrices.append(weight)

        def save_model(self, path):
            """保存模型"""
            if self.best_model_state:
                torch.save(self.best_model_state, path)

        def load_model(self, path):
            """加载模型"""
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.best_acc = checkpoint['test_acc']

        def get_visualization_data(self):
            """返回可视化所需的数据"""
            return {
                'confusion_matrix': self.confusion_matrix,
                'weight_matrices': self.weight_matrices[-1] if self.weight_matrices else None,  # 返回最后一个权重矩阵
                'training_history': self.training_history
            }

        def get_current_learning_rate(self):
            """获取当前学习率"""
            if self.optimizer is not None:
                for param_group in self.optimizer.param_groups:
                    return param_group['lr']
            return None

        def reset_statistics(self):
            """重置统计数据"""
            self.confusion_matrix = np.zeros((10, 10))
            self.weight_matrices = []
            self.training_history = {
                'train_loss': [],
                'train_acc': [],
                'test_loss': [],
                'test_acc': [],
                'epoch_times': []
            }
            self.best_acc = 0
            self.best_model_state = None


