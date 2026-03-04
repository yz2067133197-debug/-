#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNN优化功能测试脚本
测试SNN优化参数是否能正确应用到数据处理流程中
"""

import numpy as np
import matplotlib.pyplot as plt
from data_processing import SynapticDataProcessor

def test_snn_optimization():
    """测试SNN优化功能"""
    print("[TEST] 开始测试SNN优化功能...")
    
    # 创建数据处理器
    processor = SynapticDataProcessor()
    
    # 加载手动数据
    print("[DATA] 加载默认突触数据...")
    processor.load_manual_data()
    
    # 测试不同的SNN参数配置
    test_configs = [
        {
            'name': '标准配置',
            'tau': 10.0,
            'v_threshold': 0.5,
            'time_steps': 128
        },
        {
            'name': '高精度配置',
            'tau': 20.0,
            'v_threshold': 1.0,
            'time_steps': 256
        },
        {
            'name': '快速训练配置',
            'tau': 5.0,
            'v_threshold': 0.3,
            'time_steps': 64
        }
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n[CONFIG] 测试配置: {config['name']}")
        print(f"   参数: tau={config['tau']}, v_threshold={config['v_threshold']}, time_steps={config['time_steps']}")
        
        # 应用SNN优化的归一化
        snn_params = {
            'tau': config['tau'],
            'v_threshold': config['v_threshold'],
            'time_steps': config['time_steps']
        }
        
        # 归一化数据
        processor.normalize_data(num_points=100, snn_params=snn_params)
        
        # 获取归一化后的数据
        ltp_data = processor.ltp_data
        ltd_data = processor.ltd_data
        
        if ltp_data is not None and ltd_data is not None:
            # 计算数据统计信息
            ltp_range = np.max(ltp_data[:, 1]) - np.min(ltp_data[:, 1])
            ltd_range = np.max(ltd_data[:, 1]) - np.min(ltd_data[:, 1])
            ltp_mean = np.mean(ltp_data[:, 1])
            ltd_mean = np.mean(ltd_data[:, 1])
            
            results[config['name']] = {
                'ltp_range': ltp_range,
                'ltd_range': ltd_range,
                'ltp_mean': ltp_mean,
                'ltd_mean': ltd_mean,
                'ltp_data': ltp_data.copy(),
                'ltd_data': ltd_data.copy()
            }
            
            print(f"   [OK] LTP数据范围: {ltp_range:.4f}, 均值: {ltp_mean:.4f}")
            print(f"   [OK] LTD数据范围: {ltd_range:.4f}, 均值: {ltd_mean:.4f}")
        else:
            print(f"   [ERROR] 数据归一化失败")
    
    # 可视化结果
    if results:
        print("\n[PLOT] 生成可视化结果...")
        create_comparison_plot(results)
        print("[OK] 可视化图表已保存为 snn_optimization_test_results.png")
    
    # 输出测试总结
    print("\n[SUMMARY] 测试总结:")
    for name, data in results.items():
        print(f"  {name}:")
        print(f"    - LTP动态范围: {data['ltp_range']:.4f}")
        print(f"    - LTD动态范围: {data['ltd_range']:.4f}")
        print(f"    - 数据质量: {'优秀' if data['ltp_range'] > 0.5 and data['ltd_range'] > 0.5 else '良好'}")
    
    print("\n[DONE] SNN优化功能测试完成！")
    return results

def create_comparison_plot(results):
    """创建对比图表"""
    fig, axes = plt.subplots(2, len(results), figsize=(5*len(results), 10))
    if len(results) == 1:
        axes = axes.reshape(-1, 1)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        
        # LTP曲线
        axes[0, i].plot(data['ltp_data'][:, 0], data['ltp_data'][:, 1], 
                       color=color, linewidth=2, label='LTP')
        axes[0, i].set_title(f'{name} - LTP曲线')
        axes[0, i].set_xlabel('时间')
        axes[0, i].set_ylabel('强度')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].legend()
        
        # LTD曲线
        axes[1, i].plot(data['ltd_data'][:, 0], data['ltd_data'][:, 1], 
                       color=color, linewidth=2, label='LTD')
        axes[1, i].set_title(f'{name} - LTD曲线')
        axes[1, i].set_xlabel('时间')
        axes[1, i].set_ylabel('强度')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig('snn_optimization_test_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    try:
        test_results = test_snn_optimization()
        print("\n[SUCCESS] 测试成功完成！可以在GUI中使用SNN优化功能了。")
    except Exception as e:
        print(f"\n[ERROR] 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
