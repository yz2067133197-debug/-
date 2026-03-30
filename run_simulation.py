import torch
from data_processing import SynapticDataProcessor, load_current_time_data
from snn import SNN
from training_manager import TrainingManager, TrainingConfig
from utils import plot_input_data, calculate_firing_rate
import logging
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def get_output_dir():
    """获取输出目录路径，确保目录存在"""
    # 获取当前脚本所在目录（项目根目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, 'output')
    
    # 如果output目录不存在，创建它
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    return output_dir


def run_simulation(
        current_data_path,
        dataset_name,
        dataset_manager,
        network_params,
        training_params,
        log_callback=None,
        progress_callback=None,
        synaptic_processor=None,
        stop_event=None,
        enable_gate=True,
        enable_gradient_proxy=True,
        enable_quantization=False,
        optimizer_type='Adam (默认)',
        random_seed=None
):
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
        synaptic_processor (SynapticDataProcessor): 预配置的突触数据处理器（可选）
    """
    if log_callback is None:
        log_callback = lambda msg: logging.info(msg)
    if progress_callback is None:
        progress_callback = lambda p: None

    # 获取输出目录
    output_dir = get_output_dir()
    log_callback(f"输出目录: {output_dir}")

    # 设置随机种子（如果启用）
    if random_seed is not None:
        import random
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        log_callback(f"已设置随机种子: {random_seed} (实验可复现模式)")

    try:
        # 加载和处理突触数据
        log_callback("正在处理突触数据...")
        
        if synaptic_processor is not None and synaptic_processor.normalized_data is not None:
            # 使用外部传入的处理器（已包含配置和数据）
            processor = synaptic_processor
            log_callback(f"使用预配置的突触数据，采样点数: {processor.normalized_data['num_points']}")
            normalized_data = processor.normalized_data
        else:
            # 创建新的处理器（后备方案）
            processor = SynapticDataProcessor()
            log_callback("重新加载突触数据 (未提供预配置数据)...")
            
            if current_data_path is None:
                # 手动输入模式，使用默认数据
                if not processor.load_manual_data():
                    raise ValueError("无法加载手动突触数据")
            else:
                # 文件输入模式
                if not processor.load_data(current_data_path):
                    raise ValueError("无法加载突触数据")
            
            # [硬核物理对齐] 废弃强制100点插值，严格锚定真实的物理脉冲个数作为网络输入维度
            true_peak_count = processor.peak_count if (processor.peak_count is not None and processor.peak_count > 0) else 100
            log_callback(f"已提取真实物理脉冲响应点数: {true_peak_count} (废除插值包装)")
            normalized_data = processor.normalize_data(num_points=true_peak_count)

        synaptic_data = torch.tensor(normalized_data['ltp'], dtype=torch.float32)
        ltd_data = torch.tensor(normalized_data['ltd'], dtype=torch.float32)
        
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

        # Determine number of classes
        try:
            dataset_info = dataset_manager.get_dataset_info(dataset_name)
            output_size = dataset_info.get('num_classes', 10)
            log_callback(f"数据集 '{dataset_name}' 类别数: {output_size}")
        except Exception as e:
            log_callback(f"获取数据集信息失败: {e}，默认使用10个类别")
            output_size = 10
            
        # 创建模型时根据门控开关决定是否启用突触数据
        input_size = 28 * 28  # 假设输入图像大小为28x28
        # output_size = 10  # 假设10个类别 (Removed hardcoded value)
        model = SNN(
            input_dim=input_size,
            output_dim=output_size,
            hidden_layers=network_params['hidden_layers'],
            hidden_neurons=network_params['hidden_neurons'],
            tau=network_params['tau'],
            v_threshold=network_params['v_threshold'],
            v_reset=network_params['v_reset'],
            time_steps=network_params['time_steps'],
            use_synaptic_data=enable_gate,  # 根据门控开关决定
            synaptic_data_dim=len(synaptic_data) if enable_gate else 0  # 门控关时不需要突触数据维度
        )
        
        # 创建训练配置
        config = TrainingConfig(
            epochs=training_params['epochs'],
            batch_size=training_params['batch_size'],
            learning_rate=training_params['learning_rate'],
            train_batches=training_params['train_batches'],
            test_batches=training_params['test_batches'],
            optimizer_type=optimizer_type,
            **network_params
        )
        
        # 创建训练管理器
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log_callback(f"使用设备: {device}")
        training_manager = TrainingManager(dataset_manager, model, device)
        
        # 设置梯度代理开关
        training_manager.enable_gradient_proxy = enable_gradient_proxy
        
        # 设置突触数据（无论门控是否开启，梯度代理可能仍需要突触数据）
        if enable_gate or enable_gradient_proxy:
            training_manager.set_synaptic_data(synaptic_data, ltd_data=ltd_data)
        else:
            log_callback("门控和梯度代理均关闭，跳过突触数据加载")
        
        # 开始训练
        log_callback("开始训练...")
        accuracy = training_manager.train(
            config=config,
            dataset_name=dataset_name,
            progress_callback=progress_callback,
            log_callback=log_callback,
            stop_event=stop_event
        )

        if stop_event is not None and stop_event.is_set():
            log_callback("训练已终止")
            return None

        if accuracy is not None:
            best_final_acc = accuracy.get('best_test_acc', accuracy['test_acc'][-1])
            log_callback(f"训练完成，模型已加载为最高精度状态，最终测得准确率: {best_final_acc:.2f}%")
            
            quant_result = None
            
            # 如果启用了量化映射，在训练完成后执行
            if enable_quantization:
                log_callback(f"\n{'='*50}")
                log_callback(f"正在执行权重→电导量化映射...")
                try:
                    # Pass synaptic_data and ltd_data directly if enable_quantization is True
                    ltp_for_quant = synaptic_data
                    ltd_for_quant = ltd_data
                    
                    quant_result = training_manager.quantize_weights_to_device(
                        dataset_name=dataset_name,
                        batch_size=training_params['batch_size'],
                        log_callback=log_callback,
                        ltp_curve=ltp_for_quant,
                        ltd_curve=ltd_for_quant,
                        test_batches=training_params.get('test_batches', None)
                    )
                    log_callback(f"量化前浮点精度: {quant_result['original_acc']:.2f}%")
                    log_callback(f"量化后精度:     {quant_result['quantized_acc']:.2f}%")
                    log_callback(f"精度损失:       {quant_result['acc_drop']:.2f}%")
                    
                    # 保存量化后模型的混淆矩阵
                    if 'confusion_matrix' in quant_result and quant_result['confusion_matrix'] is not None:
                        quant_cm_path = os.path.join(output_dir, 'confusion_matrix_quantized.png')
                        plt.figure(figsize=(12, 10))
                        sns.heatmap(quant_result['confusion_matrix'], annot=True, fmt='.0f', cmap='Oranges')
                        
                        actual_n = quant_result.get('num_levels', '未知')
                        plt.title(f'混淆矩阵 (实测曲线非线性量化后模型 @ 测试集, N={actual_n}级)')
                        plt.xlabel('预测类别')
                        plt.ylabel('真实类别')
                        plt.savefig(quant_cm_path, bbox_inches='tight', dpi=300)
                        plt.close()
                        log_callback(f"量化后混淆矩阵已保存为: {quant_cm_path}")
                    
                    log_callback(f"{'='*50}")
                except Exception as e:
                    log_callback(f"量化映射失败: {str(e)}")
            
            # 获取训练数据并保存可视化结果
            viz_data = training_manager.get_visualization_data()
            confusion_matrix = viz_data['confusion_matrix']

            # 保存混淆矩阵图像 (训练过程累计)
            confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_training.png')
            plt.figure(figsize=(12, 10))
            sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues')
            plt.title('混淆矩阵 (训练过程累计)')
            plt.xlabel('预测类别')
            plt.ylabel('真实类别')
            plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300)
            plt.close()
            log_callback(f"训练过程混淆矩阵已保存为: {confusion_matrix_path}")

            # 生成并保存测试集混淆矩阵 (最佳模型)
            try:
                log_callback("正在使用最佳模型在测试集上生成混淆矩阵...")
                test_confusion_matrix = training_manager.run_test_inference(
                    dataset_name, 
                    training_params['batch_size']
                )
                
                test_cm_path = os.path.join(output_dir, 'confusion_matrix_test_best.png')
                plt.figure(figsize=(12, 10))
                sns.heatmap(test_confusion_matrix, annot=True, fmt='.0f', cmap='Greens') # 使用不同颜色区分
                plt.title('混淆矩阵 (最佳模型 @ 测试集)')
                plt.xlabel('预测类别')
                plt.ylabel('真实类别')
                plt.savefig(test_cm_path, bbox_inches='tight', dpi=300)
                plt.close()
                log_callback(f"测试集最佳模型混淆矩阵已保存为: {test_cm_path}")
            except Exception as e:
                log_callback(f"生成测试集混淆矩阵失败: {str(e)}")

            # 绘制训练历史
            plt.figure(figsize=(12, 5))

            # 准确率曲线
            plt.subplot(1, 2, 1)
            plt.plot(viz_data['training_history']['train_acc'], label='训练准确率')
            plt.plot(viz_data['training_history']['test_acc'], label='测试准确率')
            plt.title('训练过程准确率')
            plt.xlabel('Epoch')
            plt.ylabel('准确率 (%)')
            plt.legend()
            plt.grid(True)

            # 损失曲线
            plt.subplot(1, 2, 2)
            plt.plot(viz_data['training_history']['train_loss'], label='训练损失')
            plt.plot(viz_data['training_history']['test_loss'], label='测试损失')
            plt.title('训练过程损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            training_history_path = os.path.join(output_dir, 'training_history.png')
            plt.savefig(training_history_path, dpi=300)
            plt.close()
            log_callback(f"训练历史图像已保存为: {training_history_path}")

            # 保存输入数据图像
            input_data_path = os.path.join(output_dir, 'input_data_plot.png')
            plot_input_data(time_data, current_data, filename=input_data_path)
            log_callback(f"输入数据图像已保存为: {input_data_path}")

            # 保存模型
            model_save_path = os.path.join(output_dir, 'trained_model.pth')
            training_manager.save_model(model_save_path)
            log_callback(f"模型已保存到: {model_save_path}")
            
            # 保存训练历史到CSV文件
            csv_save_path = os.path.join(output_dir, 'training_history.csv')
            training_manager.save_training_history_to_csv(csv_save_path)
            log_callback(f"训练历史已保存到: {csv_save_path}")
            
            # 追加量化映射结果到CSV
            if quant_result is not None:
                try:
                    with open(csv_save_path, 'a', encoding='utf-8') as f:
                        f.write("\n")
                        f.write("Quantization Mapping Results\n")
                        actual_n = quant_result.get('num_levels', '未知')
                        f.write(f"Quantization Levels (N),{actual_n}\n")
                        f.write(f"Original Test Accuracy (%),{quant_result['original_acc']:.2f}\n")
                        f.write(f"Quantized Test Accuracy (%),{quant_result['quantized_acc']:.2f}\n")
                        f.write(f"Accuracy Difference (%),{quant_result['acc_drop']:.2f}\n")
                    log_callback("量化映射结果已成功追加到训练历史 CSV 文件。")
                except Exception as e:
                    log_callback(f"追加量化结果到CSV失败: {str(e)}")
            
            return accuracy
        else:
            log_callback("训练已终止")
            return None

    except Exception as e:
        log_callback(f"错误: {str(e)}")
        import traceback
        log_callback(traceback.format_exc())
        return None
