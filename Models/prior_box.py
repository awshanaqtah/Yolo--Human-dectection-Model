"""
Prior Box (Default Box) Generation for SSD.

This module generates default/prior boxes at multiple scales and aspect ratios,
which are essential for the SSD detection architecture.
"""

import torch
import itertools
from typing import List, Tuple


class PriorBox:
    """
    Generate prior boxes (default boxes) for SSD.

    Prior boxes are pre-defined boxes at different scales and aspect ratios
    that serve as reference anchors for object detection.

    Args:
        input_size: Input image size (height, width)
        feature_maps: List of feature map sizes for each detection layer
        min_sizes: Minimum box sizes for each feature map
        max_sizes: Maximum box sizes for each feature map
        aspect_ratios: List of aspect ratios for each feature map
        clip: Whether to clip coordinates to [0, 1]

    Example:
        >>> prior_box = PriorBox(
        ...     input_size=(300, 300),
        ...     feature_maps=[38, 19, 10, 5, 3, 1],
        ...     min_sizes=[30, 60, 111, 162, 213, 264],
        ...     max_sizes=[60, 111, 162, 213, 264, 315],
        ...     aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        ... )
        >>> priors = prior_box.forward()  # [8732, 4]
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (300, 300),
        feature_maps: List[int] = [38, 19, 10, 5, 3, 1],
        min_sizes: List[int] = [30, 60, 111, 162, 213, 264],
        max_sizes: List[int] = [60, 111, 162, 213, 264, 315],
        aspect_ratios: List[List[int]] = [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        clip: bool = True
    ):
        self.input_size = input_size
        self.feature_maps = feature_maps
        self.min_sizes = min_sizes
        self.max_sizes = max_sizes
        self.aspect_ratios = aspect_ratios
        self.clip = clip

    def forward(self) -> torch.Tensor:
        """
        Generate prior boxes.

        Returns:
            Prior boxes of shape [num_priors, 4] in [cx, cy, w, h] format
            normalized to [0, 1]

        Example:
            >>> priors = prior_box.forward()
            >>> print(priors.shape)  # torch.Size([8732, 4])
        """
        priors = []

        for k, f in enumerate(self.feature_maps):
            # Calculate scale for this feature map
            scale = self.input_size[0] / self.feature_maps[0]  # 300 / 38 ≈ 7.9

            # Iterate over each position in the feature map
            for i, j in itertools.product(range(f), repeat=2):
                # Unit center x, y (normalized to feature map)
                cx = (j + 0.5) * scale / self.input_size[1]
                cy = (i + 0.5) * scale / self.input_size[0]

                # Minimum size box (aspect ratio 1)
                size_min = self.min_sizes[k] / self.input_size[0]
                priors.append([cx, cy, size_min, size_min])

                # Maximum size box (aspect ratio 1)
                size_max = self.max_sizes[k] / self.input_size[0]
                priors.append([cx, cy, size_max, size_max])

                # Boxes with different aspect ratios
                for ar in self.aspect_ratios[k]:
                    # For each aspect ratio, create boxes of different sizes
                    size = self.min_sizes[k] / self.input_size[0]

                    # Aspect ratio > 1: width > height
                    # Aspect ratio < 1: height > width
                    w = size * (ar ** 0.5)
                    h = size / (ar ** 0.5)

                    priors.append([cx, cy, w, h])

                    # Also add flipped version (reciprocal aspect ratio)
                    if ar != 1:
                        priors.append([cx, cy, h, w])

        # Convert to tensor
        priors = torch.tensor(priors, dtype=torch.float32)

        if self.clip:
            priors = torch.clamp(priors, 0.0, 1.0)

        return priors

    def __call__(self) -> torch.Tensor:
        """Make prior box generator callable."""
        return self.forward()


def generate_priors(
    input_size: Tuple[int, int] = (300, 300),
    feature_maps: List[int] = [38, 19, 10, 5, 3, 1],
    min_sizes: List[int] = [30, 60, 111, 162, 213, 264],
    max_sizes: List[int] = [60, 111, 162, 213, 264, 315],
    aspect_ratios: List[List[int]] = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
) -> torch.Tensor:
    """
    Convenience function to generate prior boxes.

    Args:
        input_size: Input image size
        feature_maps: Feature map sizes
        min_sizes: Minimum box sizes
        max_sizes: Maximum box sizes
        aspect_ratios: Aspect ratios for each feature map

    Returns:
        Prior boxes tensor

    Example:
        >>> priors = generate_priors()
        >>> print(f"Generated {len(priors)} prior boxes")
    """
    prior_box = PriorBox(
        input_size=input_size,
        feature_maps=feature_maps,
        min_sizes=min_sizes,
        max_sizes=max_sizes,
        aspect_ratios=aspect_ratios
    )
    return prior_box.forward()


def decode_boxes(
    loc_pred: torch.Tensor,
    priors: torch.Tensor,
    variances: List[float] = [0.1, 0.2]
) -> torch.Tensor:
    """
    Decode predicted box offsets to actual boxes.

    SSD predicts offsets from prior boxes, not absolute coordinates.
    This function converts those offsets to actual box coordinates.

    Args:
        loc_pred: Predicted localization offsets [N, num_priors, 4]
        priors: Prior boxes [num_priors, 4] in [cx, cy, w, h] format
        variances: Variance values for decoding

    Returns:
        Decoded boxes [N, num_priors, 4] in [cx, cy, w, h] format

    Example:
        >>> decoded_boxes = decode_boxes(loc_pred, priors)
        >>> print(decoded_boxes.shape)  # [batch_size, 8732, 4]
    """
    if loc_pred.dim() == 2:
        loc_pred = loc_pred.unsqueeze(0)

    batch_size = loc_pred.size(0)
    num_priors = priors.size(0)

    # Decode bounding boxes
    # Formula: cx = cx_prior + cx_offset * cx_variance * w_prior
    #          cy = cy_prior + cy_offset * cy_variance * h_prior
    #          w = w_prior * exp(w_offset * w_variance)
    #          h = h_prior * exp(h_offset * h_variance)

    boxes = torch.cat([
        priors[:, :2] + loc_pred[:, :, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc_pred[:, :, 2:] * variances[1])
    ], dim=2)

    return boxes


def encode_boxes(
    boxes: torch.Tensor,
    priors: torch.Tensor,
    variances: List[float] = [0.1, 0.2]
) -> torch.Tensor:
    """
    Encode ground truth boxes relative to priors.

    This is the inverse of decode_boxes - converts absolute box coordinates
    to offsets from prior boxes.

    Args:
        boxes: Ground truth boxes [N, 4] in [cx, cy, w, h] format
        priors: Prior boxes [M, 4] in [cx, cy, w, h] format
        variances: Variance values for encoding

    Returns:
        Encoded offsets [N, M, 4]

    Example:
        >>> encoded = encode_boxes(gt_boxes, priors)
        >>> print(encoded.shape)  # [num_gt, 8732, 4]
    """
    # Convert to center format if needed
    # Assuming boxes are already in [cx, cy, w, h] format

    # Encode
    # Formula: cx_offset = (cx - cx_prior) / (cx_variance * w_prior)
    #          cy_offset = (cy - cy_prior) / (cy_variance * h_prior)
    #          w_offset = log(w / w_prior) / w_variance
    #          h_offset = log(h / h_prior) / h_variance

    encoded = torch.cat([
        (boxes[:, :2] - priors[:, :2]) / (variances[0] * priors[:, 2:]),
        torch.log(boxes[:, 2:] / priors[:, 2:]) / variances[1]
    ], dim=1)

    return encoded


def convert_boxes_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        boxes: Boxes in [cx, cy, w, h] format

    Returns:
        Boxes in [x1, y1, x2, y2] format
    """
    return torch.cat([
        boxes[:, :2] - boxes[:, 2:] / 2,
        boxes[:, :2] + boxes[:, 2:] / 2
    ], dim=1)


def convert_boxes_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] format.

    Args:
        boxes: Boxes in [x1, y1, x2, y2] format

    Returns:
        Boxes in [cx, cy, w, h] format
    """
    return torch.cat([
        (boxes[:, :2] + boxes[:, 2:]) / 2,
        boxes[:, 2:] - boxes[:, :2]
    ], dim=1)
