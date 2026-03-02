"""
Data augmentation and transformation utilities for Human Detection Model.

This module provides transforms that correctly handle both images and their
corresponding bounding boxes during augmentation.
"""

import random
import numpy as np
from typing import Tuple, List, Dict, Optional
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms


class Compose:
    """
    Compose multiple transforms together.

    Args:
        transforms: List of transform objects

    Example:
        >>> transform = Compose([
        ...     Resize((300, 300)),
        ...     ToTensor(),
        ...     Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ... ])
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image: Image.Image, target: Dict = None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Resize:
    """
    Resize image and adjust bounding boxes accordingly.

    Args:
        size: Target size (height, width) or single size for square resize
        keep_aspect_ratio: If True, maintain aspect ratio and pad

    Example:
        >>> transform = Resize((300, 300))
        >>> image, target = transform(image, {'boxes': boxes})
    """

    def __init__(self, size: Tuple[int, int], keep_aspect_ratio: bool = False):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.keep_aspect_ratio = keep_aspect_ratio

    def __call__(self, image: Image.Image, target: Dict = None):
        # Get original size
        orig_w, orig_h = image.size
        target_h, target_w = self.size

        if self.keep_aspect_ratio:
            # Resize maintaining aspect ratio
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)

            image = TF.resize(image, (new_h, new_w))

            # Pad to target size
            pad_left = (target_w - new_w) // 2
            pad_top = (target_h - new_h) // 2
            pad_right = target_w - new_w - pad_left
            pad_bottom = target_h - new_h - pad_top

            image = TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)

            # Adjust boxes
            if target is not None and 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_left  # x coordinates
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_top   # y coordinates
                target['boxes'] = boxes
        else:
            # Direct resize
            image = TF.resize(image, self.size)

            # Adjust boxes
            if target is not None and 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                boxes[:, [0, 2]] *= (target_w / orig_w)  # x coordinates
                boxes[:, [1, 3]] *= (target_h / orig_h)  # y coordinates
                target['boxes'] = boxes

        return image, target


class RandomHorizontalFlip:
    """
    Randomly flip image horizontally and adjust bounding boxes.

    Args:
        prob: Probability of flipping

    Example:
        >>> transform = RandomHorizontalFlip(prob=0.5)
    """

    def __init__(self, prob: float = 0.5):
        self.prob = prob

    def __call__(self, image: Image.Image, target: Dict = None):
        if random.random() < self.prob:
            image = TF.hflip(image)

            if target is not None and 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes']
                width = image.width

                # Flip x coordinates: new_x = width - old_x
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                target['boxes'] = boxes

        return image, target


class ColorJitter:
    """
    Randomly change the brightness, contrast, saturation and hue of an image.

    Args:
        brightness: Brightness jittering factor
        contrast: Contrast jittering factor
        saturation: Saturation jittering factor
        hue: Hue jittering factor

    Example:
        >>> transform = ColorJitter(brightness=0.2, contrast=0.2)
    """

    def __init__(
        self,
        brightness: float = 0,
        contrast: float = 0,
        saturation: float = 0,
        hue: float = 0
    ):
        self.transform = transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, image: Image.Image, target: Dict = None):
        image = self.transform(image)
        return image, target


class ToTensor:
    """
    Convert PIL Image to PyTorch Tensor and normalize to [0, 1].

    Also converts boxes and labels to tensors if present.

    Example:
        >>> transform = ToTensor()
    """

    def __call__(self, image: Image.Image, target: Dict = None):
        image = TF.to_tensor(image)

        if target is not None:
            if 'boxes' in target and len(target['boxes']) > 0:
                target['boxes'] = target['boxes']
            if 'labels' in target and len(target['labels']) > 0:
                target['labels'] = target['labels']

        return image, target


class Normalize:
    """
    Normalize tensor image with mean and standard deviation.

    Args:
        mean: Mean for each channel
        std: Standard deviation for each channel

    Example:
        >>> transform = Normalize(
        ...     mean=[0.485, 0.456, 0.406],
        ...     std=[0.229, 0.224, 0.225]
        ... )
    """

    def __init__(
        self,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float]
    ):
        self.mean = mean
        self.std = std

    def __call__(self, image, target: Dict = None):
        image = TF.normalize(image, self.mean, self.std)
        return image, target


def get_train_transform(
    input_size: Tuple[int, int] = (300, 300),
    horizontal_flip_prob: float = 0.5,
    color_jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.2, 0.1),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Compose:
    """
    Get training data transformations.

    Args:
        input_size: Target image size
        horizontal_flip_prob: Probability of horizontal flip
        color_jitter: Color jitter parameters (brightness, contrast, saturation, hue)
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transform

    Example:
        >>> transform = get_train_transform()
        >>> image, target = transform(image, target)
    """
    return Compose([
        Resize(input_size),
        RandomHorizontalFlip(prob=horizontal_flip_prob),
        ColorJitter(*color_jitter),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])


def get_val_transform(
    input_size: Tuple[int, int] = (300, 300),
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> Compose:
    """
    Get validation/test data transformations (no augmentation).

    Args:
        input_size: Target image size
        mean: Normalization mean
        std: Normalization std

    Returns:
        Composed transform
    """
    return Compose([
        Resize(input_size),
        ToTensor(),
        Normalize(mean=mean, std=std)
    ])
