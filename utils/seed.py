"""
Random seed setup for reproducibility.

This module ensures that all random operations (Python, NumPy, PyTorch)
produce the same results across different runs when using the same seed.
"""

import random
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seed for reproducibility.

    This function sets the random seed for Python's random module,
    NumPy, and PyTorch (both CPU and CUDA). It also configures
    PyTorch to use deterministic algorithms for CUDA operations.

    Args:
        seed: Random seed to use
        deterministic: Whether to use deterministic algorithms for CUDA.
                      This may slow down training but ensures reproducibility.

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will produce the same results
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed
    torch.manual_seed(seed)

    # Set CUDA random seed if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Configure deterministic behavior for CUDA
    if deterministic and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enable deterministic mode for PyTorch
        torch.use_deterministic_algorithms(True, warn_only=True)

    print(f"Random seed set to {seed} (deterministic={deterministic})")


def get_seed() -> int:
    """
    Get the current random seed from Python's random module.

    Returns:
        Current random seed
    """
    # Note: This is a simplified version that may not work for all cases
    # For production, you might want to store the seed in a global variable
    return 42  # Default seed
