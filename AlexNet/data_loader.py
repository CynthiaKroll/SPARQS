import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from config import config

def get_data_loaders(distributed=False, augment=True):
    """
    Create data loaders for training and testing.
    
    Args:
        distributed: Whether to use distributed training
        augment: Whether to use data augmentation
    
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Define transforms
    if augment:
        # Enhanced data augmentation
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    else:
        # Basic transformation
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=config["data_dir"],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=config["data_dir"],
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Split training set into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    test_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["test_batch_size"],
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_dataset

def get_calibration_loader(num_batches=None):
    """
    Create a data loader for calibration during quantization.
    
    Args:
        num_batches: Number of batches to use for calibration
    
    Returns:
        calibration_loader
    """
    # Use the same transform as test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load dataset
    calib_dataset = torchvision.datasets.CIFAR10(
        root=config["data_dir"],
        train=True,
        download=True,
        transform=transform
    )
    
    # If num_batches is specified, limit the dataset size
    if num_batches is not None:
        calib_size = num_batches * config["batch_size"]
        indices = torch.randperm(len(calib_dataset))[:calib_size]
        calib_dataset = torch.utils.data.Subset(calib_dataset, indices)
    
    # Create data loader
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return calib_loader
