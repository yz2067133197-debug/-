class PresetConfig:
    """预设配置类"""
    OPTIMAL = {
        # 网络参数
        "hidden_layers": 3,  # 增加到3层
        "neurons_per_layer": 512,  # 增加到512个神经元
        "tau": 10.0,  # 减小时间常数以加快响应
        "v_threshold": 0.5,  # 降低阈值电压使神经元更容易发放
        "v_reset": 0.0,
        "time_steps": 128,  # 增加时间步长以获取更多信息
        # 训练参数
        "epochs": 20,  # 增加训练轮次
        "batch_size": 64,  # 增加批次大小以提高稳定性
        "learning_rate": 0.001,  # 调整学习率
        "train_batches": 1000,  # 调整训练批次
        "test_batches": 200,  # 调整测试批次
        "ltp_points": 100,  # LTP采样点数
    }

    FAST = {
        "hidden_layers": 2,
        "neurons_per_layer": 256,
        "tau": 10.0,
        "v_threshold": 0.5,
        "v_reset": 0.0,
        "time_steps": 64,
        "epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.001,
        "train_batches": 200,
        "test_batches": 50,
        "ltp_points": 50,
    }