"""
权重分布可视化工具
用于生成输入状态概率分布和权重分布直方图
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from snn_scientific_implementation import ScientificSNN
from dataset_manager import DatasetManager  # 移到顶部导入

class WeightDistributionVisualizer:
    """权重分布可视化类"""

    def __init__(self):
        # 设置中文字体支持
        plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    def generate_input_state(self, size=10000):
        """生成输入状态数据（示例数据，实际应用中应替换为真实输入）"""
        # 生成均值为0，标准差为0.02的正态分布数据
        input_state = np.random.normal(0, 0.02, size)
        # 限制范围在[-0.08, 0.08]之间
        input_state = np.clip(input_state, -0.08, 0.08)
        return input_state

    def generate_initial_weights(self, size=10000):
        """生成初始权重分布（示例数据）"""
        # 生成均值为-2，标准差为2的正态分布数据
        initial_weights = np.random.normal(-2, 2, size)
        # 限制范围在[-12, 6]之间
        initial_weights = np.clip(initial_weights, -12, 6)
        return initial_weights

    def generate_final_weights(self, size=10000):
        """生成最终权重分布（示例数据）"""
        # 生成均值为0，标准差为1的正态分布数据
        final_weights = np.random.normal(0, 1, size)
        # 限制范围在[-6, 6]之间
        final_weights = np.clip(final_weights, -6, 6)
        return final_weights

    def generate_input_state_from_dataset(self, dataset_name='mnist', num_samples=10000):
        """从真实数据集生成输入状态概率分布数据
        参数:
            dataset_name: 数据集名称 ('mnist' 或 'fmnist' 或自定义数据集名称)
            num_samples: 要获取的样本数量
        返回:
            处理后的输入状态数据
        """
        # 移除函数内的导入语句
        try:
            # 初始化数据集管理器
            dataset_manager = DatasetManager()
            
            # 检查数据集是否存在
            if dataset_name not in dataset_manager.list_datasets():
                raise ValueError(f"数据集 '{dataset_name}' 不存在")
            
            # 获取数据加载器
            dataloader = dataset_manager.get_dataloader(
                dataset_name, train=True, batch_size=num_samples, shuffle=True
            )
            
            # 获取一批数据
            images, _ = next(iter(dataloader))
            
            # 确保获取的样本数量正确
            images = images[:num_samples]
            
            # 将图像数据展平并转换为numpy数组
            input_data = images.view(-1, 28*28).numpy()
            
            # 计算每个样本的激活值（这里简单地取平均值作为示例）
            # 实际应用中可能需要根据SNN的输入处理方式进行调整
            input_state = np.mean(input_data, axis=1)
            
            # 标准化处理（与原有示例数据保持一致的范围）
            input_state = (input_state - np.mean(input_state)) / (np.std(input_state) + 1e-8)
            input_state = input_state * 0.02  # 缩放到与原有示例相似的范围
            input_state = np.clip(input_state, -0.08, 0.08)
            
            return input_state
        except Exception as e:
            print(f"从数据集获取输入数据失败: {e}")
            # 失败时返回示例数据
            return self.generate_input_state(num_samples)

    def load_real_weights(self, model_path='trained_model.pth'):
        """从训练好的模型中加载真实权重"""

        try:
            # 创建SNN模型实例（参数需与训练时一致）
            snn = ScientificSNN(
                input_dim=28*28,
                output_dim=10,
                hidden_layers=1,
                hidden_neurons=100,
                tau=20.0,
                time_steps=50,
                v_threshold=1.0
            )
            
            # 保存模型的初始权重
            initial_weights = snn.layers[0].fc.weight.detach().numpy().flatten()
            
            # 加载训练好的模型权重
            snn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            
            # 提取最终权重
            final_weights = snn.layers[0].fc.weight.detach().numpy().flatten()
            
            return initial_weights, final_weights
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            return None, None

    def plot_weight_distributions(self, input_state=None, initial_weights=None, final_weights=None,
                                 save_path='weight_distribution.png', show_plot=True):
        """绘制权重分布直方图"""
        # 生成示例数据（如果未提供）
        if initial_weights is None:
            initial_weights = self.generate_initial_weights()
        
        if final_weights is None:
            final_weights = self.generate_final_weights()

        # 创建图像（1行2列布局）
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        # 绘制初始权重直方图
        bins_initial = np.linspace(initial_weights.min(), initial_weights.max(), 30)
        ax1.hist(initial_weights, bins=bins_initial, color='skyblue', alpha=0.7)
        ax1.set_title('初始权重分布', fontsize=14)
        ax1.set_xlabel('权重值', fontsize=12)
        ax1.set_ylabel('概率', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # 绘制最终权重直方图
        bins_final = np.linspace(final_weights.min(), final_weights.max(), 30)
        ax2.hist(final_weights, bins=bins_final, color='salmon', alpha=0.7)
        ax2.set_title('最终权重分布', fontsize=14)
        ax2.set_xlabel('权重值', fontsize=12)
        ax2.set_ylabel('概率', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # 调整布局
        plt.tight_layout()

        # 保存图像
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")

        # 显示图像
        if show_plot:
            plt.show()

    def visualize_from_model(self, model_path='trained_model.pth', use_real_input=True, dataset_name='mnist'):
        """从模型中加载权重并可视化
        参数:
            model_path: 模型文件路径
            use_real_input: 是否使用真实输入数据
            dataset_name: 数据集名称
        """
        initial_weights, final_weights = self.load_real_weights(model_path)
        
        if final_weights is not None:
            # 生成输入状态数据
            if use_real_input:
                print(f"从数据集 '{dataset_name}' 获取真实输入数据...")
                input_state = self.generate_input_state_from_dataset(dataset_name)
            else:
                input_state = self.generate_input_state()
            
            # 绘制直方图
            self.plot_weight_distributions(input_state, initial_weights, final_weights)
        else:
            print("无法从模型加载权重，使用示例数据进行可视化。")
            # 确保即使加载失败也能正常显示
            self.plot_weight_distributions(initial_weights=None, final_weights=None)

    def visualize_from_weights(self, weight_matrix, input_state=None):
        """从权重矩阵可视化分布
        参数:
            weight_matrix: 权重矩阵数据
            input_state: 输入状态数据（可选）
        """
        try:
            # 确保权重矩阵是numpy数组
            if torch.is_tensor(weight_matrix):
                weight_matrix = weight_matrix.cpu().numpy()

            # 展平权重矩阵
            final_weights = weight_matrix.flatten()
            initial_weights = self.generate_initial_weights(len(final_weights))

            # 如果没有提供输入状态，则生成示例数据
            if input_state is None:
                input_state = self.generate_input_state()
            elif torch.is_tensor(input_state):
                input_state = input_state.cpu().numpy().flatten()

            # 绘制直方图
            self.plot_weight_distributions(input_state, initial_weights, final_weights)
        except Exception as e:
            print(f"可视化权重分布时出错: {e}")

if __name__ == '__main__':
    visualizer = WeightDistributionVisualizer()
    
    # 使用真实输入数据和模型权重进行可视化
    visualizer.visualize_from_model(use_real_input=True, dataset_name='mnist')
    
    # 如果需要使用自定义数据集，可以先导入
    # from dataset_manager import DatasetManager
    # dataset_manager = DatasetManager()
    # dataset_manager.import_custom_dataset('path/to/your/dataset', 'custom_dataset')
    # visualizer.visualize_from_model(use_real_input=True, dataset_name='custom_dataset')