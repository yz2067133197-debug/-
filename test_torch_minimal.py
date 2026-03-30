
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
x = torch.randn(10, 10)
print("Tensor created:", x.shape)
import torch.nn as nn
m = nn.Linear(10, 10)
print("Layer created")
import torch.optim as optim
o = optim.Adam(m.parameters())
print("Optimizer created")
