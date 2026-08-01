import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(batch_size=None, test_batch_size=None, distributed=False, augment=True):
    """
    Get MNIST data loaders for training and testing.
    
    Args:
        batch_size: Training batch size (default from config)
        test_batch_size: Test batch size (default from config)
        distributed: Whether to use distributed training
        augment: Whether to apply data augmentation
    
    Returns:
        train_loader, test_loader
    """
    # Import config here to avoid circular imports
    from config import config
    
    if batch_size is None:
        batch_size = config["batch_size"]
    if test_batch_size is None:
        test_batch_size = config["test_batch_size"]
    
    # Define transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and create datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=False, 
        download=True, 
        transform=test_transform
    )
    
    # Create data loaders
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            sampler=train_sampler,
            num_workers=4, 
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4, 
            pin_memory=True,
            drop_last=True  # Prevent BatchNorm issues with batch_size=1
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=test_batch_size, 
        shuffle=False,
        num_workers=4, 
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_calibration_loader(num_batches=None):
    """
    Get a subset of training data for calibration (quantization, temperature scaling, etc.)
    
    Args:
        num_batches: Number of batches to use for calibration
    
    Returns:
        calibration_loader
    """
    # Import config here to avoid circular imports
    from config import config
    
    if num_batches is None:
        num_batches = config["quantization"]["calibration_batches"]
    
    # Use test transform (no augmentation for calibration)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Create dataset
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=False,  # Should already be downloaded
        transform=transform
    )
    
    # Create subset for calibration
    batch_size = config["batch_size"]
    total_samples = num_batches * batch_size
    indices = np.random.choice(len(dataset), total_samples, replace=False)
    calibration_dataset = Subset(dataset, indices)
    
    # Create data loader
    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return calibration_loader

def get_validation_loader(split_ratio=0.1, batch_size=None, augment=True):
    """
    Create train/validation loaders by splitting the MNIST training set.

    The train split keeps the training augmentation; the validation split uses
    the plain eval transform. (The previous version applied a single
    non-augmented transform to both, so models trained through it silently lost
    augmentation, and it ignored the caller's batch size.)

    Args:
        split_ratio: Fraction of training data to use for validation
        batch_size: Training batch size (default from config)
        augment: Whether to apply data augmentation to the train split

    Returns:
        train_loader, val_loader
    """
    from config import config

    if batch_size is None:
        batch_size = config["batch_size"]

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = eval_transform

    # Two dataset views over the same files, differing only in transform, so the
    # validation subset is NOT augmented.
    train_view = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform)
    val_view = torchvision.datasets.MNIST(
        root='./data', train=True, download=False, transform=eval_transform)

    dataset_size = len(train_view)
    val_size = int(split_ratio * dataset_size)
    # Fixed-seed permutation so the split is reproducible across runs/models.
    perm = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    val_indices, train_indices = perm[:val_size], perm[val_size:]

    train_loader = DataLoader(
        Subset(train_view, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Prevent BatchNorm issues with batch_size=1
    )

    val_loader = DataLoader(
        Subset(val_view, val_indices),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader 