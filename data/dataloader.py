"""
DataLoader wrapper for Human Detection Model.

This module provides convenience functions for creating DataLoaders
with proper configuration for training, validation, and testing.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional, Tuple

from .dataset import HumanDetectionDataset, collate_fn


def create_dataloader(
    dataset: HumanDetectionDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """
    Create a DataLoader for the dataset.

    Args:
        dataset: HumanDetectionDataset instance
        batch_size: Number of images per batch
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch

    Returns:
        DataLoader instance

    Example:
        >>> dataloader = create_dataloader(
        ...     train_dataset,
        ...     batch_size=16,
        ...     shuffle=True
        ... )
        >>> for images, targets in dataloader:
        ...     # images: [batch_size, 3, H, W]
        ...     # targets: list of dicts with 'boxes' and 'labels'
        ...     pass
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last
    )


def create_train_val_dataloaders(
    train_dataset: HumanDetectionDataset,
    val_dataset: HumanDetectionDataset,
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for both dataloaders
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory

    Returns:
        Tuple of (train_dataloader, val_dataloader)

    Example:
        >>> train_loader, val_loader = create_train_val_dataloaders(
        ...     train_dataset,
        ...     val_dataset,
        ...     batch_size=16
        ... )
    """
    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, val_loader


def get_batch_info(dataloader: DataLoader) -> dict:
    """
    Get information about a DataLoader.

    Args:
        dataloader: DataLoader instance

    Returns:
        Dictionary with dataloader information

    Example:
        >>> info = get_batch_info(train_loader)
        >>> print(f"Total batches: {info['num_batches']}")
    """
    return {
        'dataset_size': len(dataloader.dataset),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'shuffle': dataloader.shuffle,
        'num_workers': dataloader.num_workers
    }
