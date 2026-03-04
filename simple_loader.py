import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class SimpleFingerprintDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        
        # Check for standard train/test structure
        split_name = 'train' if train else 'test'
        split_dir = os.path.join(data_dir, split_name)
        
        if os.path.isdir(split_dir):
            self.root_dir = split_dir
            self.use_subdir_split = True
        else:
            self.root_dir = data_dir
            self.use_subdir_split = False
            
        self.classes = sorted([d for d in os.listdir(self.root_dir) 
                             if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load all images
        all_samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            files = sorted([f for f in os.listdir(class_dir) if f.endswith('.png')])
            for f in files:
                all_samples.append((os.path.join(class_dir, f), class_idx))
        
        if self.use_subdir_split:
            # Already split by directory structure
            self.samples = all_samples
        else:
            # Simple split: 80% train, 20% test
            # To be deterministic, we seed the shuffle
            random = np.random.RandomState(42)
            indices = np.arange(len(all_samples))
            random.shuffle(indices)
            
            split_idx = int(0.8 * len(all_samples))
            if train:
                self.indices = indices[:split_idx]
            else:
                self.indices = indices[split_idx:]
                
            self.samples = [all_samples[i] for i in self.indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # print(f"DEBUG: Loading {img_path}", flush=True)
        try:
            image = Image.open(img_path).convert('L')
            
            # Resize to 28x28
            image = image.resize((28, 28))
            
            # ToTensor & Normalize
            # PIL image is 0-255. Convert to 0-1 float.
            img_np = np.array(image, dtype=np.float32) / 255.0
            
            # Normalize (0.5, 0.5) -> (x - 0.5) / 0.5
            img_np = (img_np - 0.5) / 0.5
            
            # Add channel dimension: [1, 28, 28]
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            
            return img_tensor, label
        except Exception as e:
            print(f"ERROR loading {img_path}: {e}", flush=True)
            # Return dummy
            return torch.zeros(1, 28, 28), label

class SimpleDatasetManager:
    def __init__(self):
        self.datasets = {}
        
    def import_custom_dataset(self, data_dir, name):
        # We just store the path and create datasets on demand
        self.data_dir = data_dir
        
        # Check for standard train/test structure
        train_dir = os.path.join(data_dir, 'train')
        if os.path.isdir(train_dir):
            scan_dir = train_dir
        else:
            scan_dir = data_dir
            
        # Check classes
        classes = sorted([d for d in os.listdir(scan_dir) 
                        if os.path.isdir(os.path.join(scan_dir, d))])
        
        return {
            'name': name,
            'classes': classes,
            'num_classes': len(classes)
        }

    def get_dataloader(self, dataset_name, train=True, batch_size=32):
        dataset = SimpleFingerprintDataset(self.data_dir, train=train)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=0, # Important: 0 workers to avoid segfault/multiprocessing issues
            pin_memory=False
        )
        
    def get_dataset_info(self, dataset_name):
        train_dir = os.path.join(self.data_dir, 'train')
        if os.path.isdir(train_dir):
            scan_dir = train_dir
        else:
            scan_dir = self.data_dir
            
        classes = sorted([d for d in os.listdir(scan_dir) 
                        if os.path.isdir(os.path.join(scan_dir, d))])
        return {'num_classes': len(classes)}
