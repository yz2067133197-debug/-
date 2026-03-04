import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


class TrainingConfig:
    """训练配置类"""

    def __init__(self, **kwargs):
        self.epochs = kwargs.get('epochs', 10)
        self.batch_size = kwargs.get('batch_size', 32)
        self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.train_batches = kwargs.get('train_batches', 1000)
        self.test_batches = kwargs.get('test_batches', 200)
        self.hidden_layers = kwargs.get('hidden_layers', 3)
        self.hidden_neurons = kwargs.get('hidden_neurons', 512)
        self.tau = kwargs.get('tau', 10.0)
        self.v_threshold = kwargs.get('v_threshold', 0.5)
        self.v_reset = kwargs.get('v_reset', 0.0)
        self.time_steps = kwargs.get('time_steps', 128)


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
                return torch.from_numpy(transformed).float().to(device)
            return transformed
            
        except Exception as e:
            error_msg = f"转换数据时出错: {str(e)}. 输入形状: {numpy_data.shape}, 目标维度: {self.target_dim}"
            if hasattr(self, 'pca') and self.pca is not None:
                error_msg += f", PCA组件数: {self.pca.n_components_}"
            raise RuntimeError(error_msg) from e


class TrainingManager:
    """训练管理器类"""

    def __init__(self, dataset_manager, model, device):
        self.dataset_manager = dataset_manager
        self.device = device
        self.model = model.to(device)
        
        # Determine number of classes from model if available, else default to 10
        num_classes = getattr(model, 'output_dim', 10)
        self.confusion_matrix = np.zeros((num_classes, num_classes))
        
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
        self.optimizer = None
        self.synaptic_data = None  # 存储突触数据(LTP)
        self.ltd_data = None      # 存储突触数据(LTD)
        self.synaptic_transformer = None  # 突触数据转换器

    def set_synaptic_data(self, synaptic_data, ltd_data=None, use_pca=False):  # 默认禁用PCA
        """
        设置突触数据并自动调整维度
        
        参数:
            synaptic_data: 突触数据张量或numpy数组 (LTP)
            ltd_data: 突触数据张量或numpy数组 (LTD)
            use_pca: 是否使用PCA进行降维，如果为False则使用截断
        """
        if not isinstance(synaptic_data, torch.Tensor):
            synaptic_data = torch.FloatTensor(synaptic_data)
        
        # 确保是2D张量 [batch_size, features]
        if len(synaptic_data.shape) == 1:
            synaptic_data = synaptic_data.unsqueeze(0)  # [features] -> [1, features]
            
        self.synaptic_data = synaptic_data.to(self.device)
        
        # 处理LTD数据
        if ltd_data is not None:
            if not isinstance(ltd_data, torch.Tensor):
                ltd_data = torch.FloatTensor(ltd_data)
            if len(ltd_data.shape) == 1:
                ltd_data = ltd_data.unsqueeze(0)
            self.ltd_data = ltd_data.to(self.device)
        else:
            self.ltd_data = None
        
        # 如果模型使用突触数据，则初始化转换器
        if hasattr(self.model, 'use_synaptic_data') and self.model.use_synaptic_data:
            target_dim = getattr(self.model, 'synaptic_data_dim', 0)
            if target_dim > 0:
                # 检查维度是否匹配，如果匹配则不需要转换器
                if synaptic_data.shape[1] == target_dim:
                    print(f"突触数据维度({target_dim})与目标维度匹配，无需转换器")
                    self.synaptic_transformer = None
                    return
                
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
    
    def get_batch_synaptic_data(self, batch_size):
        """
        获取一批处理后的突触数据
        
        参数:
            batch_size: 批大小
            
        返回:
            (ltp_batch, ltd_batch): 处理后的突触数据
        """
        if self.synaptic_data is None:
            return None, None
        
        # 确保数据在CPU上处理
        if self.synaptic_data.is_cuda:
            device = self.synaptic_data.device
            data_cpu = self.synaptic_data.cpu()
            ltd_cpu = self.ltd_data.cpu() if self.ltd_data is not None else None
        else:
            device = None
            data_cpu = self.synaptic_data
            ltd_cpu = self.ltd_data
            
        # 随机选择样本
        idx = torch.randint(0, len(data_cpu), (batch_size,))
        batch_data = data_cpu[idx]
        batch_ltd = ltd_cpu[idx] if ltd_cpu is not None else None
        
        # 如果有转换器，则进行维度转换
        if self.synaptic_transformer is not None:
            try:
                batch_data = self.synaptic_transformer.transform(batch_data)
                # 确保返回的是tensor
                if not torch.is_tensor(batch_data):
                    batch_data = torch.FloatTensor(batch_data)
                
                # 对LTD也应用相同的转换（假设特征空间一致）
                # 注意：这可能不完全准确，但如果LTP/LTD是对称的，这是合理的近似
                if batch_ltd is not None:
                     batch_ltd = self.synaptic_transformer.transform(batch_ltd)
                     if not torch.is_tensor(batch_ltd):
                        batch_ltd = torch.FloatTensor(batch_ltd)

            except Exception as e:
                print(f"警告: 突触数据转换失败: {str(e)}，使用原始数据")
        
        # 移回原设备
        if device is not None:
            batch_data = batch_data.to(device)
            if batch_ltd is not None:
                batch_ltd = batch_ltd.to(device)
            
        return batch_data, batch_ltd

    def prepare_device_curves(self):
        """准备LTP/LTD曲线的插值函数用于梯度代理"""
        if self.synaptic_data is None:
            return

        try:
            # 获取LTP曲线 (取平均)
            ltp_curve = self.synaptic_data.cpu().numpy()
            if len(ltp_curve.shape) > 1:
                ltp_curve = np.mean(ltp_curve, axis=0)
            
            # 获取LTD曲线
            if self.ltd_data is not None:
                ltd_curve = self.ltd_data.cpu().numpy()
                if len(ltd_curve.shape) > 1:
                    ltd_curve = np.mean(ltd_curve, axis=0)
            else:
                ltd_curve = np.zeros_like(ltp_curve)

            # 计算斜率 (dG/dt)
            ltp_slope = np.gradient(ltp_curve)
            ltd_slope = np.gradient(ltd_curve)

            # 建立 G -> Slope 的映射
            # 1. LTP: 排序G值以建立查找表
            ltp_sort_idx = np.argsort(ltp_curve)
            self.ltp_g_sorted = ltp_curve[ltp_sort_idx]
            self.ltp_slope_sorted = ltp_slope[ltp_sort_idx]
            
            # 2. LTD: 排序G值
            ltd_sort_idx = np.argsort(ltd_curve)
            self.ltd_g_sorted = ltd_curve[ltd_sort_idx]
            self.ltd_slope_sorted = ltd_slope[ltd_sort_idx]
            
            self.device_curves_ready = True
            print("已准备LTP/LTD曲线用于梯度代理")
            
        except Exception as e:
            print(f"准备器件曲线失败: {str(e)}")
            self.device_curves_ready = False

    def apply_device_constraints(self, model):
        """应用器件约束到梯度（梯度代理机制）"""
        if not getattr(self, 'device_curves_ready', False):
            self.prepare_device_curves()
            if not getattr(self, 'device_curves_ready', False):
                return

        for param in model.parameters():
            if param.grad is None:
                continue
                
            # 仅对权重矩阵应用（忽略偏置等）
            if param.ndim < 2:
                continue

            # 获取当前权重值
            w_flat = param.data.cpu().numpy().ravel()
            grad_flat = param.grad.data.cpu().numpy().ravel()
            
            slope_factor = np.ones_like(w_flat)
            
            # LTP mask (grad < 0，即需要增加权重)
            # 映射规则：如果当前权重是w，沿LTP曲线增加的容易程度由LTP斜率决定
            ltp_mask = grad_flat < 0
            if np.any(ltp_mask):
                # 使用np.interp查找当前权重对应的斜率
                slope = np.interp(w_flat[ltp_mask], self.ltp_g_sorted, self.ltp_slope_sorted)
                slope_factor[ltp_mask] = np.abs(slope)
                
            # LTD mask (grad > 0，即需要减少权重)
            ltd_mask = grad_flat > 0
            if np.any(ltd_mask):
                slope = np.interp(w_flat[ltd_mask], self.ltd_g_sorted, self.ltd_slope_sorted)
                slope_factor[ltd_mask] = np.abs(slope)
            
            # 归一化斜率因子以保持梯度的总体尺度，防止过小
            # 我们希望保留相对形状，但不要让梯度消失
            if np.max(slope_factor) > 1e-6:
                slope_factor = slope_factor / np.max(slope_factor)
            
            # 增加一个基数，防止完全梯度消失
            slope_factor = slope_factor * 0.9 + 0.1
            
            # 应用调节因子
            factor_tensor = torch.from_numpy(slope_factor.reshape(param.grad.shape)).to(param.device).float()
            param.grad.data *= factor_tensor

    def train_epoch(self, train_loader, criterion, optimizer, device, epoch, num_epochs, log_callback=None, max_batches=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            if max_batches and batch_idx >= max_batches:
                break
                
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 获取当前batch的突触数据（自动处理维度）
            ltp_batch, ltd_batch = self.get_batch_synaptic_data(inputs.size(0))
            
            # 前向传播
            outputs = self.model(inputs, ltp_batch, ltd_batch)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            
            # 应用梯度代理（如果已加载突触数据）
            if self.synaptic_data is not None:
                self.apply_device_constraints(self.model)
                
            optimizer.step()

            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 每10个batch输出一次训练信息
            if batch_idx % 10 == 0 and log_callback:
                log_callback(f'Train Epoch: {epoch+1}/{num_epochs} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ' 
                           f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        # 计算平均损失和准确率
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        if log_callback:
            log_callback(f'Train Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')
        
        return avg_loss, accuracy

    def evaluate(self, test_loader, criterion, num_batches=None, synaptic_data=None, stop_event=None):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = num_batches if num_batches else len(test_loader)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                if stop_event is not None and stop_event.is_set():
                    return None, None
                if num_batches and batch_idx >= num_batches:
                    break
                    
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 获取当前batch的突触数据（自动处理维度）
                if synaptic_data is not None:
                    # 如果提供了外部突触数据，暂时只支持LTP作为单输入（保持向后兼容）
                    # 或者我们可以扩展evaluate接口
                    batch_synaptic_data = synaptic_data
                    batch_ltd_data = None
                    
                    if len(synaptic_data) >= inputs.size(0):
                        batch_synaptic_data = synaptic_data[:inputs.size(0)].to(self.device)
                    else:
                        batch_synaptic_data = torch.zeros((inputs.size(0), synaptic_data.size(1)), 
                                                       device=self.device)
                        batch_synaptic_data[:len(synaptic_data)] = synaptic_data.to(self.device)
                else:
                    # 使用训练管理器中的突触数据
                    batch_synaptic_data, batch_ltd_data = self.get_batch_synaptic_data(inputs.size(0))
                
                # 前向传播
                outputs = self.model(inputs, batch_synaptic_data, batch_ltd_data)
                loss = criterion(outputs, targets)

                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 每100个batch输出一次评估信息
                if batch_idx % 100 == 0 and hasattr(self, 'log_callback') and self.log_callback:
                    self.log_callback(f'Test: [{batch_idx}/{total_batches} ' 
                                   f'({100. * batch_idx / total_batches:.0f}%)]\tLoss: {loss.item():.6f}')

        # 计算平均损失和准确率
        actual_batches = batch_idx + 1  # 实际处理的batch数量
        avg_loss = running_loss / actual_batches if actual_batches > 0 else 0
        accuracy = 100. * correct / total if total > 0 else 0
        
        return avg_loss, accuracy

    def train(self, config, dataset_name, progress_callback=None, log_callback=None, stop_event=None):
        """训练模型"""
        # 设置日志回调
        self.log_callback = log_callback
        
        def log(msg):
            if log_callback:
                log_callback(msg)
            else:
                print(msg)
        
        def should_stop():
            return stop_event is not None and stop_event.is_set()
        
        # 确保模型在正确的设备上
        self.model = self.model.to(self.device)
        
        log(f"开始训练，使用设备: {self.device}")
        if hasattr(self, 'synaptic_data') and self.synaptic_data is not None:
            log(f"检测到突触数据，形状: {self.synaptic_data.shape}")
            if hasattr(self, 'synaptic_transformer') and self.synaptic_transformer is not None:
                log(f"使用{'PCA' if self.synaptic_transformer.use_pca else '截断'}方法处理突触数据，目标维度: {self.synaptic_transformer.target_dim}")

        # 获取数据加载器
        log("正在获取数据加载器...")
        train_loader = self.dataset_manager.get_dataloader(
            dataset_name, train=True, batch_size=config.batch_size
        )
        test_loader = self.dataset_manager.get_dataloader(
            dataset_name, train=False, batch_size=config.batch_size
        )
        log("数据加载器获取完成")

        # 初始化优化器和损失函数
        log("正在初始化损失函数...")
        criterion = nn.CrossEntropyLoss().to(self.device)
        log("正在初始化优化器...")
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        log("正在初始化调度器...")
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=2)
        log("初始化完成")
        
        # 重置训练历史
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_times': []
        }
        
        log(f"开始训练，共 {config.epochs} 个epoch，每个epoch {config.train_batches} 个batch，批量大小: {config.batch_size}")
        log(f"学习率: {config.learning_rate}, 优化器: Adam, 调度器: ReduceLROnPlateau")

        for epoch in range(config.epochs):
            if should_stop():
                return None
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # 记录epoch开始时间
            epoch_start_time = time.time()
            
            log(f"\nEpoch {epoch+1}/{config.epochs}")
            log("-" * 60)

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if should_stop():
                    return None
                if batch_idx >= config.train_batches:
                    break

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # 获取当前batch的突触数据（自动处理维度）
                batch_synaptic_data = self.get_batch_synaptic_data(inputs.size(0))

                # 前向传播
                outputs = self.model(inputs, batch_synaptic_data)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                loss.backward()
                
                # 应用梯度代理（如果已加载突触数据）
                if hasattr(self, 'synaptic_data') and self.synaptic_data is not None:
                    self.apply_device_constraints(self.model)
                
                self.optimizer.step()

                # 统计
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 保存训练数据用于可视化
                # 仅在每个epoch的第一个batch保存权重，防止内存溢出
                self.save_training_data(outputs, labels, save_weights=(batch_idx == 0))
                
                # 每10个batch输出一次训练信息
                if batch_idx % 10 == 0 and log_callback:
                    acc = 100. * correct / total if total > 0 else 0
                    log(f'Epoch: {epoch+1}/{config.epochs} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ' 
                        f'({100. * batch_idx / len(train_loader):.0f}%)]\t' 
                        f'Loss: {loss.item():.6f}\tAcc: {acc:.2f}%')

                # 更新进度条
                if progress_callback:
                    progress = (epoch * config.train_batches + batch_idx + 1) / (config.epochs * config.train_batches)
                    progress_callback(progress)
            
            # 计算epoch的训练指标
            actual_batches = min(config.train_batches, len(train_loader))
            avg_loss = total_loss / actual_batches if actual_batches > 0 else 0
            train_acc = 100. * correct / total if total > 0 else 0
            
            # 在测试集上评估
            if should_stop():
                return None
            test_loss, test_acc = self.evaluate(test_loader, criterion, num_batches=config.test_batches, stop_event=stop_event)
            if test_loss is None:
                return None
            
            # 更新学习率调度器
            scheduler.step(test_loss)
            
            # 记录训练历史
            self.training_history['train_loss'].append(avg_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['test_loss'].append(test_loss)
            self.training_history['test_acc'].append(test_acc)
            
            # 计算epoch耗时
            epoch_time = time.time() - epoch_start_time
            self.training_history['epoch_times'].append(epoch_time)
            
            # 输出epoch总结
            log(f'Epoch {epoch+1} 完成 - 训练损失: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            log(f'测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%, 耗时: {epoch_time:.2f}秒')
            log(f'学习率: {self.get_current_learning_rate():.6f}')
            
            # 保存最佳模型
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': avg_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
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
        
        # 保存训练历史到CSV文件
        import os
        csv_path = os.path.join(os.path.dirname(path) if 'path' in locals() else 'output', 
                               'training_history.csv')
        self.save_training_history_to_csv(csv_path)
        
        return self.training_history

    def save_training_data(self, outputs, labels, save_weights=False):
        """保存训练过程中的数据用于可视化，优化内存使用"""
        # 更新混淆矩阵
        with torch.no_grad():  # 确保不创建计算图
            pred = outputs.argmax(dim=1).cpu()
            labels = labels.cpu()
            for t, p in zip(labels, pred):
                self.confusion_matrix[t.item()][p.item()] += 1

            # 保存权重矩阵 (仅当请求时保存，避免内存溢出)
            if save_weights:
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
        """保存最佳模型"""
        if self.best_model_state:
            # 从模型中提取网络参数
            network_params = {
                'hidden_layers': self.model.hidden_layers,
                'hidden_neurons': self.model.hidden_neurons,
                'tau': self.model.tau,
                'v_threshold': self.model.v_threshold,
                'v_reset': self.model.v_reset,
                'time_steps': self.model.time_steps
            }
            
            # 将network_params添加到保存的字典中
            self.best_model_state['network_params'] = network_params
            
            torch.save(self.best_model_state, path)

    def save_training_history_to_csv(self, csv_path):
        """
        将训练历史保存为CSV文件
        
        参数:
            csv_path: CSV文件保存路径
        """
        import csv
        from datetime import datetime
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                
                # 写入表头
                writer.writerow(['Epoch', 'Train Loss', 'Train Accuracy (%)', 
                               'Test Loss', 'Test Accuracy (%)', 'Time (seconds)'])
                
                # 写入每轮的训练数据
                for epoch in range(len(self.training_history['train_loss'])):
                    writer.writerow([
                        epoch + 1,  # Epoch从1开始
                        f"{self.training_history['train_loss'][epoch]:.6f}",
                        f"{self.training_history['train_acc'][epoch]:.2f}",
                        f"{self.training_history['test_loss'][epoch]:.6f}",
                        f"{self.training_history['test_acc'][epoch]:.2f}",
                        f"{self.training_history['epoch_times'][epoch]:.2f}"
                    ])
                
                # 写入最佳模型信息
                writer.writerow([])
                writer.writerow(['Best Model Information'])
                writer.writerow(['Best Test Accuracy (%)', f"{self.best_acc:.2f}"])
                
                # 写入模型参数信息
                writer.writerow([])
                writer.writerow(['Model Parameters'])
                writer.writerow(['Hidden Layers', self.model.hidden_layers])
                writer.writerow(['Hidden Neurons', self.model.hidden_neurons])
                writer.writerow(['Tau', self.model.tau])
                writer.writerow(['V Threshold', self.model.v_threshold])
                writer.writerow(['V Reset', self.model.v_reset])
                writer.writerow(['Time Steps', self.model.time_steps])
                
                # 写入训练时间戳
                writer.writerow([])
                writer.writerow(['Training Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
                
            print(f"训练历史已保存到: {csv_path}")
            return True
        except Exception as e:
            print(f"保存训练历史到CSV失败: {str(e)}")
            return False

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

    def run_test_inference(self, dataset_name, batch_size):
        """
        在测试集上运行推理并返回混淆矩阵
        """
        self.model.eval()
        test_loader = self.dataset_manager.get_dataloader(
            dataset_name, train=False, batch_size=batch_size
        )
        
        num_classes = self.confusion_matrix.shape[0]
        cm = np.zeros((num_classes, num_classes))
        
        print("正在进行测试集完整推理以生成混淆矩阵...")
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 处理突触数据
                batch_synaptic_data, batch_ltd_data = self.get_batch_synaptic_data(inputs.size(0))
                
                outputs = self.model(inputs, batch_synaptic_data, batch_ltd_data)
                _, predicted = outputs.max(1)
                
                for t, p in zip(targets.view(-1), predicted.view(-1)):
                    cm[t.long(), p.long()] += 1
                    
        return cm

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
