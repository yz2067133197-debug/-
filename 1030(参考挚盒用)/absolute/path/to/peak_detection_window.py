# ... existing code ...
def plot_ltp_ltd_fit(self):
    """绘制LTP/LTD原始数据曲线"""
    if not hasattr(self, 'fit_data'):
        return
        
    # 清空当前图表
    self.figure.clear()
    
    # 原始数据曲线（左上图）- 修改标题与主界面一致
    ax1 = self.figure.add_subplot(2, 2, 1)
    ax1.plot(self.fit_data['pulse'], self.fit_data['ltp_raw'], 'r-', label='LTP (原始数据)')
    ax1.plot(self.fit_data['pulse'], self.fit_data['ltd_raw'], 'b-', label='LTD (原始数据)')
    ax1.set_xlabel('归一化脉冲序列')
    ax1.set_ylabel('归一化权重变化')
    ax1.set_title('突触权重变化的归一化表示')  # 与主界面标题一致
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 修改第二个图的标题
    ax2 = self.figure.add_subplot(2, 2, 2)
    ax2.plot(self.fit_data['pulse'], self.fit_data['ltp_raw'], 'r--', label='LTP (原始数据)')
    ax2.plot(self.fit_data['pulse'], self.fit_data['ltd_raw'], 'b--', label='LTD (原始数据)')
    ax2.set_xlabel('归一化脉冲序列')
    ax2.set_ylabel('归一化权重变化')
    ax2.set_title('突触权重变化的归一化表示')  # 与主界面标题一致
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 移除误差对比图，因为不再有优化处理
    self.figure.delaxes(self.figure.axes[2])
    
    # 调整布局并刷新画布
    self.figure.tight_layout()
    self.canvas.draw()

# ... existing code ...
def process_data(self):
    """处理数据并生成LTP/LTD曲线"""
    try:
        # 直接获取原始数据，不进行采样点调整
        # 获取原始LTP/LTD数据（根据模式选择文件数据或手动数据）
        if self.mode_var.get() == "file" and self.current_file_path:
            ltp_values = self.data_processor.ltp_data
            ltd_values = self.data_processor.ltd_data
            # 生成对应的x轴数据（0-1范围）
            pulse_norm = np.linspace(0, 1, len(ltp_values))
        else:
            ltp_values = self.data_processor.DEFAULT_LTP_DATA
            ltd_values = self.data_processor.DEFAULT_LTD_DATA
            # 生成对应的x轴数据（0-1范围）
            pulse_norm = np.linspace(0, 1, len(ltp_values))
        
        # 完全不进行平滑或优化处理，直接使用原始数据
        self.fit_data = {
            'pulse': pulse_norm,
            'ltp_raw': ltp_values,
            'ltd_raw': ltd_values,
            # 为了保持兼容性，也设置这两个字段
            'ltp_optimized': ltp_values,
            'ltd_optimized': ltd_values
        }
        
        # 绘制曲线
        self.plot_ltp_ltd_fit()
        
        # 更新信息
        self.update_info("\n已显示原始LTP/LTD数据")
        self.update_info(f"数据点数: {len(pulse_norm)}")
        
    except Exception as e:
        messagebox.showerror("错误", f"处理数据失败: {str(e)}")
# ... existing code ...