import torch
import torch.optim as optim
import torch.nn as nn
import sys

with open("test_optim_out.txt", "w") as f:
    f.write("Start\n")
    try:
        model = nn.Linear(10, 2)
        f.write("Model created\n")
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        f.write("Optimizer created\n")
    except Exception as e:
        f.write(f"Error: {e}\n")
    f.write("End\n")
