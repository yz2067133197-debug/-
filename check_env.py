import sys
import torch
import numpy
import matplotlib
import tkinter
import pandas
import PIL

def check_environment():
    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
    print(f"NumPy版本: {numpy.__version__}")
    print(f"Matplotlib版本: {matplotlib.__version__}")
    print(f"Pandas版本: {pandas.__version__}")
    print(f"Pillow版本: {PIL.__version__}")
    print("Tkinter可用" if tkinter._test() else "Tkinter不可用")

if __name__ == "__main__":
    check_environment()