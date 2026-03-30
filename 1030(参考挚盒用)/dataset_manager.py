import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform or transforms.ToTensor()

        # Scan directory structure
        self.classes = sorted([d for d in os.listdir(data_dir)
                               if os.path.isdir(os.path.join(data_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale

        if self.transform:
            image = self.transform(image)

        return image, label


class DatasetManager:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.datasets = {}
        self._load_default_datasets()

    def _load_default_datasets(self):
        # 使用相对于当前文件的路径
        data_dir = os.path.join(BASE_DIR, 'data')
        
        # Load MNIST
        self.datasets['mnist'] = {
            'train': torchvision.datasets.MNIST(
                root=data_dir, train=True,
                transform=self.transform, download=False),
            'test': torchvision.datasets.MNIST(
                root=data_dir, train=False,
                transform=self.transform, download=False)
        }

        # Load Fashion-MNIST
        self.datasets['fmnist'] = {
            'train': torchvision.datasets.FashionMNIST(
                root=data_dir, train=True,
                transform=self.transform, download=False),
            'test': torchvision.datasets.FashionMNIST(
                root=data_dir, train=False,
                transform=self.transform, download=False)
        }

    def import_custom_dataset(self, data_dir, name, split_ratio=0.8):
        """Import a custom dataset from directory."""
        if not os.path.exists(data_dir):
            raise ValueError(f"Directory not found: {data_dir}")

        full_dataset = CustomDataset(data_dir, self.transform)

        # Split dataset
        train_size = int(split_ratio * len(full_dataset))
        test_size = len(full_dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size])

        self.datasets[name] = {
            'train': train_dataset,
            'test': test_dataset
        }

        return {
            'name': name,
            'classes': full_dataset.classes,
            'num_samples': len(full_dataset),
            'train_samples': train_size,
            'test_samples': test_size
        }

    def get_dataloader(self, dataset_name, train=True, batch_size=32, shuffle=True):
        """Get dataloader for specified dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        split = 'train' if train else 'test'
        dataset = self.datasets[dataset_name][split]

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
    
    def get_data_loaders(self, dataset_name='mnist', batch_size=32):
        """Get both train and test dataloaders."""
        train_loader = self.get_dataloader(dataset_name, train=True, batch_size=batch_size, shuffle=True)
        test_loader = self.get_dataloader(dataset_name, train=False, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    def get_dataset_info(self, dataset_name):
        """Get information about a dataset."""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")

        dataset = self.datasets[dataset_name]['train']

        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

        info = {
            'name': dataset_name,
            'total_samples': len(dataset),
            'train_samples': len(self.datasets[dataset_name]['train']),
            'test_samples': len(self.datasets[dataset_name]['test'])
        }

        if hasattr(dataset, 'classes'):
            info['classes'] = dataset.classes
            info['num_classes'] = len(dataset.classes)

        return info

    def list_datasets(self):
        """List all available datasets."""
        return list(self.datasets.keys())

    def validate_dataset_structure(self, data_dir):
        """Validate the structure of a custom dataset directory."""
        if not os.path.isdir(data_dir):
            return False, "Not a valid directory"

        # Check for class subdirectories
        class_dirs = [d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d))]
        if not class_dirs:
            return False, "No class subdirectories found"

        # Check for images in each class directory
        for class_dir in class_dirs:
            class_path = os.path.join(data_dir, class_dir)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not images:
                return False, f"No images found in class directory: {class_dir}"

        return True, f"Valid dataset structure with {len(class_dirs)} classes"