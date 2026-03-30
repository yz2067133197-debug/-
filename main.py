import tkinter as tk
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