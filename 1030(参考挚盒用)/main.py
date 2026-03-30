import os
import sys
import tkinter as tk

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 现在导入应用
from gui_app import App


def main():
    """应用程序入口函数"""
    try:
        # 创建主窗口
        root = tk.Tk()

        # 设置基本DPI感知
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass  # 在非Windows系统上忽略

        # 创建应用程序实例
        app = App(root)

        # 启动主循环
        root.mainloop()
    except Exception as e:
        print(f"程序启动错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()