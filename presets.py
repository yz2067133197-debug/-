class PresetConfig:
    """预设配置类"""
    OPTIMAL = {
        # 网络参数
        "hidden_layers": 2,      # 隐藏层数
        "neurons_per_layer": 256, # 每层神经元数
        "tau": 20.0,             # 时间常数
        "v_threshold": 1.0,      # 阈值电压
        "v_reset": 0.0,          # 重置电压
        "time_steps": 100,       # 时间步长
        # 训练参数
        "epochs": 10,            # 训练轮次
        "batch_size": 32,        # 批次大小
        "learning_rate": 0.0005, # 学习率
        "train_batches": 1875,   # 训练批次数
        "test_batches": 313,     # 测试批次数
        # 突触参数
        "ltp_points": 100,       # LTP采样点数
    }

    FAST = {
        "hidden_layers": 2,
        "neurons_per_layer": 128,
        "tau": 20.0,
        "v_threshold": 1.0,
        "v_reset": 0.0,
        "time_steps": 50,
        "epochs": 2,
        "batch_size": 32,
        "learning_rate": 0.001,
        "train_batches": 100,
        "test_batches": 20,
        "ltp_points": 50,
    }