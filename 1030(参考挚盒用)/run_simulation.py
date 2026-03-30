import torch
from data_processing import SynapticDataProcessor, load_current_time_data
from snn_scientific_implementation import ScientificSNN, ScientificSTDPTrainingManager
from training_manager import STDPTrainingConfig, TrainingManager
from utils import plot_input_data, calculate_firing_rate
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def run_simulation(current_data_path=None, dataset_name='mnist', dataset_manager=None, 
                   network_params=None, training_params=None, log_callback=None, 
                   progress_callback=None):
    """
    运行神经网络仿真

    Args:
        current_data_path (str): 电流数据文件路径
        dataset_name (str): 数据集名称
        dataset_manager (DatasetManager): 数据集管理器实例
        network_params (dict): 网络参数
        training_params (dict): 训练参数
        log_callback (callable): 日志回调函数
        progress_callback (callable): 进度回调函数
    """
    if log_callback is None:
        log_callback = lambda msg: logging.info(msg)
    if progress_callback is None:
        progress_callback = lambda p: None
    
    # 添加参数验证
    if training_params is None:
        training_params = {'epochs': 1}

    try:
        # 加载和处理突触数据
        log_callback("正在加载和处理突触数据...")
        processor = SynapticDataProcessor()
        
        if current_data_path is None:
            # 手动输入模式，使用默认数据
            if not processor.load_manual_data():
                raise ValueError("无法加载手动突触数据")
        else:
            # 文件输入模式
            if not processor.load_data(current_data_path):
                raise ValueError("无法加载突触数据")
        
        # 获取归一化后的突触数据
        normalized_data = processor.normalize_data(num_points=100)
        synaptic_data = torch.tensor(normalized_data['ltp'], dtype=torch.float32)
        
        # 在手动模式下跳过电流数据处理
        if current_data_path is None:
            # 使用手动数据的峰值数
            peak_count = len(processor.ltp_data) if processor.ltp_data is not None else 100
            firing_rate = calculate_firing_rate(peak_count, training_params.get('target_peaks', 100))
            log_callback(f"使用手动输入数据，峰值数: {peak_count}, 发放率: {firing_rate:.2f}")
        else:
            # 文件模式，处理电流数据
            log_callback("正在加载电流数据...")
            time_data, current_data, peak_count = load_current_time_data(current_data_path)
            firing_rate = calculate_firing_rate(peak_count, training_params.get('target_peaks', 100))
            log_callback(f"电流数据加载完成，峰值数: {peak_count}, 调整后发放率: {firing_rate:.2f}")

        # 创建模型时启用突触数据
        input_size = 28 * 28  # 假设输入图像大小为28x28
        output_size = 10  # 假设10个类别
        model = ScientificSNN(
            input_dim=input_size,
            output_dim=output_size,
            hidden_layers=network_params['hidden_layers'],
            hidden_neurons=network_params['hidden_neurons'],
            tau=network_params['tau'],
            v_threshold=network_params['v_threshold'],
            v_reset=network_params['v_reset'],
            time_steps=network_params['time_steps'],
            use_synaptic_data=True,  # 启用突触数据
            synaptic_data_dim=len(synaptic_data)  # 设置突触数据维度
        )
        
        # 创建训练配置
        config = STDPTrainingConfig(
            epochs=training_params['epochs'],
            batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate'],
            train_batches=training_params['train_batches'],
            test_batches=training_params['test_batches'],
            **network_params
        )
        
        # 创建训练管理器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_callback(f"使用设备: {device}")
        training_manager = TrainingManager(dataset_manager, model, device)
        
        # 设置突触数据
        training_manager.set_synaptic_data(synaptic_data)
        
        # 开始训练
        log_callback("开始训练...")
        training_history = training_manager.train(
            config=config,
            dataset_name=dataset_name,
            progress_callback=progress_callback,
            log_callback=log_callback
        )

        if training_history is not None:
            log_callback(f"训练完成，最终准确率: {training_history['test_acc'][-1]:.2f}%")
            
            # 获取训练数据并保存可视化结果
            viz_data = training_manager.get_visualization_data()
            confusion_matrix = viz_data['confusion_matrix']

            # 保存混淆矩阵图像
            plt.figure(figsize=(12, 10))
            sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
            plt.title('混淆矩阵')
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
            plt.close()
            log_callback("混淆矩阵已保存为: confusion_matrix.png")

            # 绘制训练历史
            plt.figure(figsize=(12, 5))

            # 准确率曲线
            plt.subplot(1, 2, 1)
            plt.plot(training_history['train_acc'], label='训练准确率')
            plt.plot(training_history['test_acc'], label='测试准确率')
            plt.title('训练过程准确率')
            plt.xlabel('Epoch')
            plt.ylabel('准确率 (%)')
            plt.legend()
            plt.grid(True)

            # 损失曲线
            plt.subplot(1, 2, 2)
            plt.plot(training_history['train_loss'], label='训练损失')
            plt.plot(training_history['test_loss'], label='测试损失')
            plt.title('训练过程损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig('training_history.png', dpi=300)
            plt.close()
            log_callback("训练历史图像已保存为: training_history.png")

            # 保存输入数据图像
            plot_input_data(time_data, current_data, filename="input_data_plot.png")
            log_callback("输入数据图像已保存为: input_data_plot.png")

            # 保存模型
            model_save_path = "trained_model.pth"
            training_manager.save_model(model_save_path)
            log_callback(f"模型已保存到: {model_save_path}")
            
            # 保存突触数据
            torch.save({
                'synaptic_data': synaptic_data,
                'normalization_stats': {
                    'ltp_mean': float(torch.mean(synaptic_data)),
                    'ltp_std': float(torch.std(synaptic_data))
                }
            }, 'synaptic_data.pt')
            log_callback("突触数据已保存到: synaptic_data.pt")
            
            return training_history
        else:
            log_callback("训练已终止")
            return None

    except Exception as e:
        log_callback(f"错误: {str(e)}")
        import traceback
        log_callback(traceback.format_exc())
        return None