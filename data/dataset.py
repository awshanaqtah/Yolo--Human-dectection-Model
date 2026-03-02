"""
Custom Dataset class for Human Detection Model.

This module provides a PyTorch Dataset class that loads images and annotations
from custom datasets in various formats (COCO, VOC, YOLO).
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Tuple, Optional, Callable
from PIL import Image
import numpy as np

from .utils import parse_coco_annotations, parse_voc_annotations, parse_yolo_annotations
from .transforms import Compose


class HumanDetectionDataset(Dataset):
    """
    PyTorch Dataset for human detection.

    Supports COCO, Pascal VOC, and YOLO annotation formats.

    Args:
        image_dir: Directory containing images
        annotation_path: Path to annotation file (COCO JSON or directory for VOC/YOLO)
        annotation_format: Format of annotations ('coco', 'voc', or 'yolo')
        transform: Transform to apply to images and targets
        return_image_id: Whether to return image ID along with image and target

    Attributes:
        image_dir: Directory containing images
        annotations: Parsed annotations dictionary
        transform: Transform to apply
        image_files: List of image filenames

    Example:
        >>> dataset = HumanDetectionDataset(
        ...     image_dir='datasets/images/train',
        ...     annotation_path='datasets/annotations/train.json',
        ...     annotation_format='coco',
        ...     transform=get_train_transform()
        ... )
        >>> image, target = dataset[0]
        >>> print(image.shape)  # torch.Size([3, 300, 300])
        >>> print(target['boxes'].shape)  # torch.Tensor([N, 4])
    """

    def __init__(
        self,
        image_dir: str,
        annotation_path: str,
        annotation_format: str = 'coco',
        transform: Optional[Compose] = None,
        return_image_id: bool = False
    ):
        self.image_dir = Path(image_dir)
        self.annotation_path = annotation_path
        self.annotation_format = annotation_format.lower()
        self.transform = transform
        self.return_image_id = return_image_id

        # Validate annotation format
        assert self.annotation_format in ['coco', 'voc', 'yolo'], \
            f"Invalid annotation format: {annotation_format}"

        # Parse annotations
        self.annotations = self._parse_annotations()

        # Get list of image files
        self.image_files = list(self.annotations.keys())

        print(f"Loaded {len(self.image_files)} images from {image_dir}")

    def _parse_annotations(self) -> Dict[str, Dict]:
        """Parse annotations based on format."""
        if self.annotation_format == 'coco':
            return parse_coco_annotations(self.annotation_path)
        elif self.annotation_format == 'voc':
            return parse_voc_annotations(self.annotation_path)
        elif self.annotation_format == 'yolo':
            return parse_yolo_annotations(self.annotation_path, str(self.image_dir))
        else:
            raise ValueError(f"Unsupported annotation format: {self.annotation_format}")

    def _load_image(self, filename: str) -> Image.Image:
        """
        Load image from disk.

        Args:
            filename: Image filename

        Returns:
            PIL Image

        Raises:
            FileNotFoundError: If image doesn't exist
            IOError: If image can't be loaded
        """
        # Try exact filename first
        image_path = self.image_dir / filename

        # If not found, try different extensions
        if not image_path.exists():
            stem = Path(filename).stem
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                image_path = self.image_dir / (stem + ext)
                if image_path.exists():
                    break

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
            return image
        except Exception as e:
            raise IOError(f"Error loading image {image_path}: {e}")

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get an item from the dataset.

        Args:
            idx: Index of the item

        Returns:
            Tuple of (image, target) where:
            - image: Tensor of shape (3, H, W)
            - target: Dictionary with 'boxes' and 'labels' keys
                    boxes: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
                    labels: Tensor of shape (N,) with class labels

        Example:
            >>> image, target = dataset[0]
            >>> print(image.shape)  # torch.Size([3, 300, 300])
            >>> print(target['boxes'].shape)  # torch.Size([N, 4])
        """
        # Get image filename
        filename = self.image_files[idx]

        # Load image
        image = self._load_image(filename)

        # Get annotations
        target = self.annotations[filename].copy()

        # Convert to tensors
        if len(target['boxes']) > 0:
            target['boxes'] = torch.tensor(target['boxes'], dtype=torch.float32)
            target['labels'] = torch.tensor(target['labels'], dtype=torch.int64)
        else:
            # Handle empty annotations
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        # Add image_id if requested
        if self.return_image_id:
            target['image_id'] = idx

        # Apply transforms
        if self.transform is not None:
            image, target = self.transform(image, target)

        return image, target

    def get_image_info(self, idx: int) -> Dict:
        """
        Get information about an image without loading it.

        Args:
            idx: Index of the image

        Returns:
            Dictionary with image information
        """
        filename = self.image_files[idx]
        ann = self.annotations[filename]

        return {
            'filename': filename,
            'num_objects': len(ann['labels']),
            'image_id': ann.get('image_id', idx)
        }


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Since each image may have different number of objects,
    we need to handle variable-sized targets.

    Args:
        batch: List of (image, target) tuples

    Returns:
        Tuple of (batched_images, list_of_targets)

    Example:
        >>> dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        >>> images, targets = next(iter(dataloader))
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack images into a batch
    images = torch.stack(images, dim=0)

    return images, targets


class DetectionDatasetWrapper:
    """
    Wrapper to conveniently create train/val/test datasets.

    Args:
        data_root: Root directory of the dataset
        annotation_format: Format of annotations
        train_transform: Transform for training
        val_transform: Transform for validation/test

    Example:
        >>> wrapper = DetectionDatasetWrapper(
        ...     data_root='datasets',
        ...     annotation_format='coco'
        ... )
        >>> train_dataset = wrapper.get_train_dataset()
        >>> val_dataset = wrapper.get_val_dataset()
    """

    def __init__(
        self,
        data_root: str,
        annotation_format: str = 'coco',
        train_transform: Optional[Compose] = None,
        val_transform: Optional[Compose] = None
    ):
        self.data_root = Path(data_root)
        self.annotation_format = annotation_format
        self.train_transform = train_transform
        self.val_transform = val_transform

    def get_train_dataset(self) -> HumanDetectionDataset:
        """Get training dataset."""
        return HumanDetectionDataset(
            image_dir=str(self.data_root / 'raw' / 'images' / 'train'),
            annotation_path=str(self.data_root / 'raw' / 'annotations' / 'train.json'),
            annotation_format=self.annotation_format,
            transform=self.train_transform
        )

    def get_val_dataset(self) -> HumanDetectionDataset:
        """Get validation dataset."""
        return HumanDetectionDataset(
            image_dir=str(self.data_root / 'raw' / 'images' / 'val'),
            annotation_path=str(self.data_root / 'raw' / 'annotations' / 'val.json'),
            annotation_format=self.annotation_format,
            transform=self.val_transform
        )

    def get_test_dataset(self) -> HumanDetectionDataset:
        """Get test dataset."""
        return HumanDetectionDataset(
            image_dir=str(self.data_root / 'raw' / 'images' / 'test'),
            annotation_path=str(self.data_root / 'raw' / 'annotations' / 'test.json'),
            annotation_format=self.annotation_format,
            transform=self.val_transform
        )
