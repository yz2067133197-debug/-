# -*- coding: utf-8 -*-
"""
SNN图像识别优化示例
演示如何使用优化的LTP/LTD数据提高SNN识别精度
"""

import numpy as np
import matplotlib.pyplot as plt
from data_processing import SynapticDataProcessor
from snn import SNN
from training_manager import TrainingManager, TrainingConfig
import torch

def demonstrate_snn_optimization():
    """演示SNN优化功能"""
    
    print("=== SNN图像识别优化演示 ===\n")
    
    # 1. 创建数据处理器
    processor = SynapticDataProcessor()
    
    # 2. 显示当前优化参数
    print("1. 当前SNN优化参数:")
    info = processor.get_snn_optimization_info()
    for param, value in info['current_params'].items():
        desc = info['description'][param]
        print(f"   {param}: {value} - {desc}")
    
    print("\n2. 推荐的优化配置:")
    for scenario, params in info['recommendations'].items():
        print(f"   {scenario}:")
        for param, value in params.items():
            print(f"     {param}: {value}")
    
    # 3. 配置高精度模式
    print("\n3. 应用高精度优化配置...")
    processor.configure_snn_optimization(**info['recommendations']['high_accuracy'])
    
    # 4. 加载数据（这里使用手动数据作为示例）
    print("\n4. 加载突触数据...")
    processor.load_manual_data()
    
    # 5. 定义SNN参数
    snn_params = {
        'tau': 20.0,
        'v_threshold': 1.0,
        'time_steps': 100
    }
    
    # 6. 应用SNN优化的归一化
    print("\n5. 应用SNN优化的数据归一化...")
    processor.normalize_data(num_points=100, snn_params=snn_params)
    
    # 7. 获取优化后的数据
    optimized_data = processor.get_normalized_data()
    
    print(f"\n6. 优化结果:")
    print(f"   数据点数: {optimized_data['num_points']}")
    print(f"   LTP范围: {optimized_data['ltp'].min():.4f} - {optimized_data['ltp'].max():.4f}")
    print(f"   LTD范围: {optimized_data['ltd'].min():.4f} - {optimized_data['ltd'].max():.4f}")
    print(f"   LTP趋势: {'递增' if optimized_data['ltp'][-1] > optimized_data['ltp'][0] else '非递增'}")
    print(f"   LTD趋势: {'递减' if optimized_data['ltd'][-1] < optimized_data['ltd'][0] else '非递减'}")
    
    # 8. 可视化优化效果
    visualize_optimization_results(processor)
    
    return processor, optimized_data

def visualize_optimization_results(processor):
    """可视化优化结果"""
    
    # 获取原始数据和优化数据
    raw_data = processor.raw_data
    normalized_data = processor.get_normalized_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('SNN优化效果对比', fontsize=16)
    
    # 原始数据
    axes[0, 0].plot(raw_data['pulses'], raw_data['ltp'], 'b-o', label='原始LTP', markersize=4)
    axes[0, 0].plot(raw_data['pulses'], raw_data['ltd'], 'r-^', label='原始LTD', markersize=4)
    axes[0, 0].set_title('原始突触数据')
    axes[0, 0].set_xlabel('脉冲序号')
    axes[0, 0].set_ylabel('电流值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 优化后数据
    axes[0, 1].plot(normalized_data['x'], normalized_data['ltp'], 'b-o', label='优化LTP', markersize=4)
    axes[0, 1].plot(normalized_data['x'], normalized_data['ltd'], 'r-^', label='优化LTD', markersize=4)
    axes[0, 1].set_title('SNN优化后数据')
    axes[0, 1].set_xlabel('归一化时间')
    axes[0, 1].set_ylabel('归一化值')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # LTP对比
    axes[1, 0].plot(normalized_data['x'], normalized_data['ltp'], 'b-', linewidth=2, label='优化LTP')
    axes[1, 0].set_title('LTP曲线特性')
    axes[1, 0].set_xlabel('归一化时间')
    axes[1, 0].set_ylabel('LTP强度')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # LTD对比
    axes[1, 1].plot(normalized_data['x'], normalized_data['ltd'], 'r-', linewidth=2, label='优化LTD')
    axes[1, 1].set_title('LTD曲线特性')
    axes[1, 1].set_xlabel('归一化时间')
    axes[1, 1].set_ylabel('LTD强度')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snn_optimization_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n7. 优化结果图表已保存为: snn_optimization_results.png")

def compare_optimization_strategies():
    """比较不同优化策略的效果"""
    
    print("\n=== 比较不同优化策略 ===\n")
    
    strategies = ['high_accuracy', 'fast_training', 'robust_generalization']
    results = {}
    
    for strategy in strategies:
        print(f"测试策略: {strategy}")
        
        processor = SynapticDataProcessor()
        info = processor.get_snn_optimization_info()
        
        # 应用策略
        processor.configure_snn_optimization(**info['recommendations'][strategy])
        processor.load_manual_data()
        
        # 归一化
        snn_params = {'tau': 20.0, 'v_threshold': 1.0}
        processor.normalize_data(num_points=100, snn_params=snn_params)
        
        # 获取结果
        data = processor.get_normalized_data()
        
        # 计算特征指标
        ltp_range = data['ltp'].max() - data['ltp'].min()
        ltd_range = data['ltd'].max() - data['ltd'].min()
        ltp_monotonicity = np.sum(np.diff(data['ltp']) > 0) / len(np.diff(data['ltp']))
        ltd_monotonicity = np.sum(np.diff(data['ltd']) < 0) / len(np.diff(data['ltd']))
        
        results[strategy] = {
            'ltp_range': ltp_range,
            'ltd_range': ltd_range,
            'ltp_monotonicity': ltp_monotonicity,
            'ltd_monotonicity': ltd_monotonicity
        }
        
        print(f"  LTP动态范围: {ltp_range:.4f}")
        print(f"  LTD动态范围: {ltd_range:.4f}")
        print(f"  LTP单调性: {ltp_monotonicity:.2%}")
        print(f"  LTD单调性: {ltd_monotonicity:.2%}")
        print()
    
    return results

def integration_example():
    """集成到训练流程的示例"""
    
    print("\n=== 集成到SNN训练流程 ===\n")
    
    # 1. 创建优化的数据处理器
    processor = SynapticDataProcessor()
    
    # 2. 配置为高精度模式
    info = processor.get_snn_optimization_info()
    processor.configure_snn_optimization(**info['recommendations']['high_accuracy'])
    
    # 3. 加载数据
    processor.load_manual_data()
    
    # 4. 定义SNN参数（与训练配置匹配）
    snn_params = {
        'tau': 10.0,           # 与TrainingConfig中的tau匹配
        'v_threshold': 0.5,    # 与TrainingConfig中的v_threshold匹配
        'time_steps': 128      # 与TrainingConfig中的time_steps匹配
    }
    
    # 5. 应用优化
    processor.normalize_data(num_points=100, snn_params=snn_params)
    
    print("优化的突触数据已准备就绪，可用于SNN训练")
    print("建议在训练时使用以下配置:")
    print(f"  - 使用优化后的突触数据维度: {processor.get_normalized_data()['num_points']}")
    print(f"  - SNN参数已匹配: tau={snn_params['tau']}, v_threshold={snn_params['v_threshold']}")
    print(f"  - 预期效果: 提高收敛速度和识别精度")
    
    return processor

if __name__ == "__main__":
    # 运行演示
    processor, data = demonstrate_snn_optimization()
    
    # 比较策略
    strategy_results = compare_optimization_strategies()
    
    # 集成示例
    integrated_processor = integration_example()
    
    print("\n=== 总结 ===")
    print("1. SNN优化功能已成功集成到数据处理流程中")
    print("2. 提供了三种预设优化策略：高精度、快速训练、鲁棒泛化")
    print("3. 优化后的LTP/LTD数据具有更好的动态范围和单调性")
    print("4. 可直接集成到现有的SNN训练流程中")
    print("\n建议：根据具体应用场景选择合适的优化策略以获得最佳性能")
