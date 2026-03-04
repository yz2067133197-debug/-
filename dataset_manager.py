import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms


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
        # Load MNIST
        self.datasets['mnist'] = {
            'train': torchvision.datasets.MNIST(
                root='./data', train=True,
                transform=self.transform, download=True),
            'test': torchvision.datasets.MNIST(
                root='./data', train=False,
                transform=self.transform, download=True)
        }

        # Load Fashion-MNIST
        self.datasets['fmnist'] = {
            'train': torchvision.datasets.FashionMNIST(
                root='./data', train=True,
                transform=self.transform, download=True),
            'test': torchvision.datasets.FashionMNIST(
                root='./data', train=False,
                transform=self.transform, download=True)
        }
        
        # Load Fingerprint Orientation (if exists)
        fingerprint_dir = os.path.join(os.getcwd(), 'fingerprint_orientation')
        if os.path.exists(fingerprint_dir):
            try:
                # Assuming the directory structure is fingerprint_orientation/train and /test
                # But my generator created fingerprint_orientation/train/class_dirs and /test/class_dirs
                # CustomDataset expects root dir containing class dirs.
                # So we need two CustomDatasets?
                # My generator made: fingerprint_orientation/train/0_degree/... and fingerprint_orientation/test/...
                
                # Check if train/test split exists
                train_dir = os.path.join(fingerprint_dir, 'train')
                test_dir = os.path.join(fingerprint_dir, 'test')
                
                if os.path.exists(train_dir) and os.path.exists(test_dir):
                    self.datasets['fingerprint'] = {
                        'train': CustomDataset(train_dir, self.transform),
                        'test': CustomDataset(test_dir, self.transform)
                    }
                else:
                    # Fallback if no split (assume root has class dirs)
                    # But CustomDataset doesn't do split automatically unless we use import_custom_dataset logic
                    # Let's assume the generator structure is correct (it is).
                    pass
            except Exception as e:
                print(f"Failed to load fingerprint dataset: {e}")

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
            num_workers=0,
            pin_memory=False
        )

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