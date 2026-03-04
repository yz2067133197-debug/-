
import sys
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from simple_loader import SimpleDatasetManager
from snn import SNN

# Force CPU to avoid CUDA/MPS issues if any
device = torch.device("cpu")

def log(msg):
    print(msg, flush=True)

def verify_training():
    log("Step 1: Initializing components...")
    log(f"Using device: {device}")
    
    # Initialize Dataset
    dm = SimpleDatasetManager()
    dataset_path = os.path.join(os.getcwd(), "fingerprint_orientation")
    
    if not os.path.exists(dataset_path):
        log(f"Error: Dataset path {dataset_path} does not exist!")
        return
        
    log(f"Importing dataset from {dataset_path}...")
    try:
        dm.import_custom_dataset(dataset_path, "fingerprint")
        log("Dataset imported successfully.")
    except Exception as e:
        log(f"Failed to import dataset: {e}")
        return

    info = dm.get_dataset_info("fingerprint")
    log(f"Dataset Info: {info}")
    num_classes = info['num_classes']
    
    # Initialize Model
    log("Initializing SNN model...")
    # Removed .to(device) to test if that's the issue
    model = SNN(
        input_dim=28*28,
        output_dim=num_classes,
        hidden_layers=1,       
        hidden_neurons=256,    
        tau=20.0,              
        time_steps=50,         
        v_threshold=0.5,       
        firing_rate=500.0       
    )
    
    # Hyperparameters
    epochs = 2
    batch_size = 32
    learning_rate = 0.0001
    
    # DataLoader
    log("Creating DataLoaders...")
    train_loader = dm.get_dataloader("fingerprint", train=True, batch_size=batch_size)
    test_loader = dm.get_dataloader("fingerprint", train=False, batch_size=batch_size)
    log("DataLoaders created.")
    
    # Optimizer & Loss
    log("Creating Optimizer...")
    try:
        log("Checking model parameters...")
        params = list(model.parameters())
        log(f"Number of parameter groups: {len(params)}")
        for i, p in enumerate(params):
            log(f"Param {i}: {p.shape}")
            
        log("Initializing Optimizer (Adam)...")
        optimizer = optim.Adam(params, lr=learning_rate, weight_decay=1e-4)
        # optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9)
        log("Optimizer initialized.")
    except Exception as e:
        log(f"Optimizer creation failed: {e}")
        return

    criterion = nn.CrossEntropyLoss()
    log("Optimizer created.")
    
    # Training Loop
    log(f"Starting training loop for {epochs} epochs...")
    model.to(device)
    # # torch.autograd.set_detect_anomaly(True)
    
    for epoch in range(epochs):
        log(f"Starting Epoch {epoch+1}")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        log(f"\nEpoch {epoch+1}/{epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                
                if batch_idx == 0 and epoch == 0:
                    log(f"DEBUG: Inputs shape: {inputs.shape}, min: {inputs.min():.4f}, max: {inputs.max():.4f}, mean: {inputs.mean():.4f}")
                    log(f"DEBUG: Targets: {targets.tolist()}")
                    
                    # Check target balance
                    unique, counts = torch.unique(targets, return_counts=True)
                    log(f"DEBUG: Target counts: {dict(zip(unique.tolist(), counts.tolist()))}")

                # Zero grad
                optimizer.zero_grad()
                
                # Forward (No synaptic data for baseline)
                outputs = model(inputs)
                
                # Normalize outputs to avoid gradient explosion/saturation
                # outputs are spike counts [0, time_steps]. Scale to [0, 1] approx or [0, 10]
                outputs = outputs / 10.0
                
                # Loss
                loss = criterion(outputs, targets)
                
                # Backward
                loss.backward()
                optimizer.step()
                
                # Stats
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # log(f"Batch {batch_idx}: Loss={loss.item():.4f}")
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}", flush=True)
                
                if batch_idx % 10 == 0:
                    acc = 100. * correct / total
                    log(f"Batch {batch_idx} Stats: Loss={loss.item():.4f}, Acc={acc:.2f}%")
                    
                    # DEBUG: Print output distribution
                    if batch_idx == 0:
                        out_sum = outputs.sum(dim=0).detach().cpu().numpy()
                        log(f"DEBUG: Output distribution (sum over batch): {out_sum}")
                        log(f"DEBUG: Raw output sample 0: {outputs[0].detach().cpu().numpy()}")
            except Exception as e:
                log(f"Error in batch {batch_idx}: {e}")
                import traceback
                log(traceback.format_exc())
                break

        epoch_acc = 100. * correct / total
        epoch_loss = running_loss / len(train_loader)
        log(f"Epoch {epoch+1} Finished. Avg Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")

    # Test Loop
    log("\nStarting Test Loop...")
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(8, 8, dtype=torch.long)
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = outputs / 10.0
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update confusion matrix
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            
    test_acc = 100. * correct / total
    avg_test_loss = test_loss / len(test_loader)
    log(f"Test Finished. Avg Loss: {avg_test_loss:.4f}, Acc: {test_acc:.2f}%")
    
    log("\nConfusion Matrix (Rows=True, Cols=Pred):")
    log(f"{'':<6} " + " ".join([f"{i:<6}" for i in range(8)]))
    for i in range(8):
        row_str = " ".join([f"{confusion_matrix[i][j].item():<6}" for j in range(8)])
        log(f"{i:<6} {row_str}")
        
    # Calculate per-class accuracy
    log("\nPer-class Accuracy:")
    for i in range(8):
        class_total = confusion_matrix[i].sum().item()
        class_correct = confusion_matrix[i][i].item()
        if class_total > 0:
            acc = 100.0 * class_correct / class_total
            log(f"Class {i}: {acc:.2f}% ({class_correct}/{class_total})")

if __name__ == "__main__":
    verify_training()
