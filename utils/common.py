"""
Common utility functions for Human Detection Model.

This module contains frequently used helper functions for device management,
model parameter counting, and other common operations.
"""

import torch
import torch.nn as nn
from typing import Union, List, Dict, Any
from pathlib import Path


def get_device() -> str:
    """
    Get the best available device for computation.

    Checks for CUDA (NVIDIA GPU), MPS (Apple Silicon), and CPU in that order.
    This is an enhanced version of the function in Main.ipynb.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> print(f"Using device: {device}")
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count the number of parameters in a PyTorch model.

    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters.
                       If False, count all parameters.

    Returns:
        Number of parameters

    Example:
        >>> model = SSD300(num_classes=2)
        >>> total_params = count_parameters(model)
        >>> print(f"Model has {total_params:,} trainable parameters")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def get_model_size(model: nn.Module) -> float:
    """
    Calculate the size of a model in megabytes (MB).

    Args:
        model: PyTorch model

    Returns:
        Model size in MB

    Example:
        >>> size_mb = get_model_size(model)
        >>> print(f"Model size: {size_mb:.2f} MB")
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = param_size + buffer_size
    return total_size / (1024 ** 2)  # Convert to MB


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    loss: float,
    map_score: float,
    path: str,
    **kwargs
) -> None:
    """
    Save a training checkpoint.

    Args:
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler state
        loss: Current loss value
        map_score: Current mAP score
        path: Path to save checkpoint
        **kwargs: Additional items to save (e.g., additional metrics)
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'map_score': map_score,
        **kwargs
    }

    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint to

    Returns:
        Dictionary containing checkpoint information
    """
    checkpoint = torch.load(path, map_location=device)

    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        state = checkpoint['scheduler_state_dict']
        if state is not None:
            scheduler.load_state_dict(state)

    print(f"Checkpoint loaded from {path}")
    print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  - Loss: {checkpoint.get('loss', 'N/A'):.4f}")
    print(f"  - mAP: {checkpoint.get('map_score', 'N/A'):.4f}")

    return checkpoint


def clip_gradients(model: nn.Module, max_norm: float = 10.0) -> float:
    """
    Clip gradients to prevent exploding gradients.

    Args:
        model: Model with gradients to clip
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping

    Example:
        >>> total_norm = clip_gradients(model, max_norm=10.0)
        >>> print(f"Gradient norm: {total_norm:.2f}")
    """
    return torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    initial_lr: float,
    decay_steps: List[int],
    decay_gamma: float
) -> float:
    """
    Adjust learning rate based on epoch.

    Args:
        optimizer: Optimizer to adjust
        epoch: Current epoch
        initial_lr: Initial learning rate
        decay_steps: List of epochs at which to decay
        decay_gamma: Multiplicative decay factor

    Returns:
        Current learning rate
    """
    lr = initial_lr
    for step in decay_steps:
        if epoch >= step:
            lr *= decay_gamma

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.

    Example:
        >>> losses = AverageMeter()
        >>> losses.update(0.5, batch_size=16)
        >>> losses.update(0.6, batch_size=16)
        >>> print(f"Average loss: {losses.avg:.4f}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update statistics with new value.

        Args:
            val: New value to add
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)
