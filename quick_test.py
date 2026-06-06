import torch
import torch.nn as nn
import numpy as np
import sys
import os

os.chdir(r"d:\解压版本合集\SNNsys815fingerprint")
sys.path.insert(0, '.')

from snn import SNN
from dataset_manager import DatasetManager

device = torch.device("cpu")
dm = DatasetManager()
train_loader = dm.get_dataloader('fingerprint', train=True, batch_size=32)

def quick_test(label, use_synaptic_data, pass_synaptic_data):
    model = SNN(
        input_dim=28*28, output_dim=8, hidden_layers=1, hidden_neurons=256,
        tau=20.0, v_threshold=1.0, v_reset=0.0, time_steps=50, firing_rate=100.0,
        use_synaptic_data=use_synaptic_data,
        synaptic_data_dim=69 if use_synaptic_data else 0
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    ltp_data = torch.tensor(np.linspace(0, 1, 69), dtype=torch.float32).unsqueeze(0)
    ltd_data = torch.tensor(np.linspace(1, 0, 69), dtype=torch.float32).unsqueeze(0)
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if batch_idx >= 20:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        
        if pass_synaptic_data:
            outputs = model(inputs, ltp_data.expand(inputs.size(0), -1), 
                          ltd_data.expand(inputs.size(0), -1))
        else:
            outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / 20
    acc = 100. * correct / total
    return avg_loss, acc

r1_loss, r1_acc = quick_test("Baseline", False, False)
r2_loss, r2_acc = quick_test("PassNoUse", False, True)
r3_loss, r3_acc = quick_test("PassAndUse", True, True)

result = f"""Baseline (no synaptic data):          Loss={r1_loss:.4f}, Acc={r1_acc:.2f}%
Pass data but use_synaptic=False:     Loss={r2_loss:.4f}, Acc={r2_acc:.2f}%
Pass data and use_synaptic=True:      Loss={r3_loss:.4f}, Acc={r3_acc:.2f}%
"""

with open("result.txt", "w") as f:
    f.write(result)

print(result)
