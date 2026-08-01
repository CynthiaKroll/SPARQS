import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from config import config

def get_data_loaders(batch_size=None, test_batch_size=None, distributed=False, augment=True):
    
    train_batch_size = batch_size if batch_size is not None else config["batch_size"]
    eval_batch_size = test_batch_size if test_batch_size is not None else config["test_batch_size"]
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  
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
    
    train_dataset = torchvision.datasets.MNIST(
        root=config["data_path"],
        train=True,
        download=True,
        transform=train_transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=config["data_path"],
        train=False,
        download=True,
        transform=test_transform
    )
    
    train_sampler = None
    test_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, test_loader

def get_validation_loader(split_ratio=0.1, batch_size=None, augment=True):
    """Create train/validation loaders by splitting the MNIST TRAINING set.

    train.py used to validate directly on the test set (val_loader =
    test_loader), so best-checkpoint selection and early stopping leaked the
    test data. The train split keeps the training augmentation; the validation
    split uses the plain eval transform.

    Returns:
        train_loader, val_loader
    """
    if batch_size is None:
        batch_size = config["batch_size"]

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        train_transform = eval_transform

    # Two dataset views over the same files, differing only in transform, so
    # the validation subset is NOT augmented.
    train_view = torchvision.datasets.MNIST(
        root=config["data_path"], train=True, download=True, transform=train_transform)
    val_view = torchvision.datasets.MNIST(
        root=config["data_path"], train=True, download=False, transform=eval_transform)

    dataset_size = len(train_view)
    val_size = int(split_ratio * dataset_size)
    # Fixed-seed permutation so the split is reproducible across runs/models.
    perm = torch.randperm(dataset_size, generator=torch.Generator().manual_seed(42)).tolist()
    val_indices, train_indices = perm[:val_size], perm[val_size:]

    train_loader = DataLoader(
        torch.utils.data.Subset(train_view, train_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Prevent BatchNorm issues with batch_size=1
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(val_view, val_indices),
        batch_size=config["test_batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, val_loader


def get_calibration_loader(num_batches=None):
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  
    ])
    
    calib_dataset = torchvision.datasets.MNIST(
        root=config["data_path"],
        train=True,
        download=True,
        transform=transform
    )
    
    if num_batches is not None:
        calib_size = num_batches * config["batch_size"]
        indices = torch.randperm(len(calib_dataset))[:calib_size]
        calib_dataset = torch.utils.data.Subset(calib_dataset, indices)
    
    calib_loader = DataLoader(
        calib_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return calib_loader