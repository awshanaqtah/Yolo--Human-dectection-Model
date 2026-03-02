"""
SSD Detection Head for Human Detection Model.

This module implements the detection head that produces
class predictions and bounding box offsets for each prior box.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class SSDHead(nn.Module):
    """
    Detection head for SSD.

    Produces classification scores and localization offsets
    for each feature map scale.

    Args:
        num_classes: Number of classes (including background)
        in_channels: Number of input channels for each feature map
        num_priors: Number of prior boxes for each feature map location
        aspect_ratios: Aspect ratios for each feature map

    Example:
        >>> head = SSDHead(
        ...     num_classes=2,
        ...     in_channels=[64, 128, 256, 512, 512, 256],
        ...     num_priors=[4, 6, 6, 6, 4, 4]
        ... )
        >>> class_pred, loc_pred = head(features)
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: List[int],
        num_priors: List[int],
        aspect_ratios: List[List[int]] = None
    ):
        super(SSDHead, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_priors = num_priors

        # Classification convolutions
        # Output: num_classes scores for each prior box
        self.classification_headers = nn.ModuleList()

        # Localization convolutions
        # Output: 4 offsets (cx, cy, w, h) for each prior box
        self.localization_headers = nn.ModuleList()

        # Create headers for each feature map
        for i, (in_c, n_priors) in enumerate(zip(in_channels, num_priors)):
            # Classification: n_priors * num_classes outputs
            self.classification_headers.append(
                nn.Conv2d(
                    in_c,
                    n_priors * num_classes,
                    kernel_size=3,
                    padding=1
                )
            )

            # Localization: n_priors * 4 outputs
            self.localization_headers.append(
                nn.Conv2d(
                    in_c,
                    n_priors * 4,
                    kernel_size=3,
                    padding=1
                )
            )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize convolution weights."""
        for header in self.classification_headers:
            # Xavier initialization for classification
            nn.init.xavier_uniform_(header.weight)
            nn.init.constant_(header.bias, 0)

        for header in self.localization_headers:
            # Initialize localization to produce small offsets initially
            nn.init.xavier_uniform_(header.weight)
            nn.init.constant_(header.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through detection head.

        Args:
            features: List of feature maps from backbone

        Returns:
            Tuple of (class_predictions, loc_predictions)
            - class_predictions: [batch_size, num_priors, num_classes]
            - loc_predictions: [batch_size, num_priors, 4]

        Example:
            >>> class_pred, loc_pred = head(features)
            >>> print(class_pred.shape)  # [batch_size, 8732, 2]
            >>> print(loc_pred.shape)    # [batch_size, 8732, 4]
        """
        batch_size = features[0].size(0)
        class_preds = []
        loc_preds = []

        # Apply headers to each feature map
        for i, (feature, cls_header, loc_header) in enumerate(
            zip(features, self.classification_headers, self.localization_headers)
        ):
            # Classification
            cls_output = cls_header(feature)  # [batch, n_priors * num_classes, H, W]

            # Permute to [batch, H, W, n_priors * num_classes]
            cls_output = cls_output.permute(0, 2, 3, 1).contiguous()

            # Reshape to [batch, H * W * n_priors, num_classes]
            num_priors = self.num_priors[i]
            cls_output = cls_output.view(
                batch_size,
                -1,
                self.num_classes
            )
            class_preds.append(cls_output)

            # Localization
            loc_output = loc_header(feature)  # [batch, n_priors * 4, H, W]

            # Permute to [batch, H, W, n_priors * 4]
            loc_output = loc_output.permute(0, 2, 3, 1).contiguous()

            # Reshape to [batch, H * W * n_priors, 4]
            loc_output = loc_output.view(
                batch_size,
                -1,
                4
            )
            loc_preds.append(loc_output)

        # Concatenate all feature maps
        class_pred = torch.cat(class_preds, dim=1)  # [batch, total_priors, num_classes]
        loc_pred = torch.cat(loc_preds, dim=1)      # [batch, total_priors, 4]

        return class_pred, loc_pred


def get_num_priors_per_feature(
    feature_maps: List[int],
    aspect_ratios: List[List[int]]
) -> List[int]:
    """
    Calculate number of priors for each feature map.

    Each location in a feature map has:
    - 2 boxes with aspect ratio 1 (min_size and max_size)
    - 2 boxes for each additional aspect ratio (ar and 1/ar)

    Args:
        feature_maps: Feature map sizes
        aspect_ratios: Aspect ratios for each feature map

    Returns:
        Number of priors for each feature map

    Example:
        >>> num_priors = get_num_priors_per_feature(
        ...     feature_maps=[38, 19, 10, 5, 3, 1],
        ...     aspect_ratios=[[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        ... )
        >>> print(num_priors)  # [4, 6, 6, 6, 4, 4]
    """
    num_priors = []
    for ars in aspect_ratios:
        # 2 for aspect ratio 1 (min and max size)
        # 2 for each additional aspect ratio
        n = 2 + 2 * len(ars)
        num_priors.append(n)
    return num_priors


def calculate_total_priors(
    feature_maps: List[int],
    num_priors_per_location: List[int]
) -> int:
    """
    Calculate total number of prior boxes.

    Args:
        feature_maps: Feature map sizes
        num_priors_per_location: Number of priors at each location

    Returns:
        Total number of priors

    Example:
        >>> total = calculate_total_priors(
        ...     feature_maps=[38, 19, 10, 5, 3, 1],
        ...     num_priors_per_location=[4, 6, 6, 6, 4, 4]
        ... )
        >>> print(total)  # 8732
    """
    total = 0
    for fm, n_priors in zip(feature_maps, num_priors_per_location):
        total += fm * fm * n_priors
    return total
