"""
Common utilities shared across all AlexNet-Clean subprojects.
This module contains utility functions used by AlexNet, LeNet, and MLP implementations.
"""

import torch


def get_device():
    """
    Check for available GPU devices and return the appropriate device.
    Priority: CUDA > MPS > CPU
    
    Returns:
        torch.device: The best available device for computation
    """
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    # Check for MPS (Apple Silicon)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS device")
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    
    return device


def safe_model_load(model_path, device='cpu', weights_only=True):
    """
    Safely load PyTorch model checkpoints with proper error handling.
    
    Args:
        model_path (str): Path to the model checkpoint
        device (str): Device to load the model on
        weights_only (bool): Only load model weights for security
        
    Returns:
        dict: Loaded state dictionary or None if loading fails
    """
    try:
        return torch.load(model_path, map_location=device, weights_only=weights_only)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def print_model_summary(model, model_name="Model"):
    """
    Print a summary of model parameters and architecture.
    
    Args:
        model (nn.Module): PyTorch model
        model_name (str): Name of the model for display
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{model_name} Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")