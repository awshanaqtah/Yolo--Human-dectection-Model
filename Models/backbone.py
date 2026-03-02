"""
Backbone network for SSD Human Detection Model.

This module implements feature extraction backbones (MobileNetV2)
with multi-scale feature extraction for object detection.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from typing import List, Dict


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 backbone for SSD with multi-scale feature extraction.

    Extracts features at multiple scales from MobileNetV2 for use in SSD.
    Uses pre-trained weights for transfer learning.

    Args:
        pretrained: Whether to use ImageNet pre-trained weights
        output_channels: Number of output channels for each detection layer

    Attributes:
        features: MobileNetV2 feature extractor
        output_channels: List of channel counts for each feature map

    Example:
        >>> backbone = MobileNetV2Backbone(pretrained=True)
        >>> features = backbone(x)
        >>> print([f.shape for f in features])  # List of feature map shapes
    """

    def __init__(self, pretrained: bool = True):
        super(MobileNetV2Backbone, self).__init__()

        # Load pre-trained MobileNetV2
        if pretrained:
            weights = MobileNet_V2_Weights.IMAGENET1K_V1
            mobilenet = models.mobilenet_v2(weights=weights)
        else:
            mobilenet = models.mobilenet_v2(weights=None)

        # Extract the feature layers
        # MobileNetV2 structure:
        # - Conv1 + BN + ReLU6
        # - Inverted residual blocks (18 layers total)
        # - Conv 1x1

        self.features = mobilenet.features

        # SSD detection layers for MobileNetV2
        # We extract features from these layers:
        # Layer 4 (after initial conv): 64 channels, 75x75 feature map
        # Layer 7: 64 channels, 38x38 feature map (first detection layer)
        # Layer 11: 96 channels, 19x19 feature map
        # Layer 14: 1280 channels, 10x10 feature map
        # Last layer: 1280 channels, 5x5 (we need to add convolutions)

        # For SSD300, we need feature maps at these scales:
        # 38x38, 19x19, 10x10, 5x5, 3x3, 1x1

        # Output channels for each detection layer
        # Based on MobileNetV2 feature extraction points
        self.output_channels = [64, 128, 256, 512, 512, 256]

        # Additional layers to get feature maps at the right scales
        # These are extra convolutions applied on top of MobileNetV2
        self.extra_layers = nn.ModuleList([
            # From 19x19 to 10x10
            nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            ),
            # From 10x10 to 5x5
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            ),
            # From 5x5 to 3x3
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            ),
            # From 3x3 to 1x1
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            ),
        ])

        # Detection layer indices in MobileNetV2
        # Layer 7: output after 7th layer (38x38)
        # Layer 11: output after 11th layer (19x19)
        # Layer 14: output after 14th layer (10x10)
        self.detection_indices = [7, 11, 14]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through backbone.

        Args:
            x: Input tensor [batch_size, 3, 300, 300]

        Returns:
            List of feature maps at different scales:
            [38x38, 19x19, 10x10, 5x5, 3x3, 1x1]

        Example:
            >>> x = torch.randn(2, 3, 300, 300)
            >>> features = backbone(x)
            >>> print([f.shape for f in features])
        """
        features = []

        # Extract features from MobileNetV2
        for i, layer in enumerate(self.features):
            x = layer(x)

            # Save features at detection layer indices
            if i in self.detection_indices:
                features.append(x)

        # Get the last feature map (19x19) as base for extra layers
        base = features[-1]

        # Apply extra layers to get smaller feature maps
        for extra_layer in self.extra_layers:
            base = extra_layer(base)
            features.append(base)

        return features


class ResNet50Backbone(nn.Module):
    """
    ResNet50 backbone for SSD (alternative to MobileNetV2).

    More accurate but slower and larger. Use if you have more compute resources.

    Args:
        pretrained: Whether to use ImageNet pre-trained weights

    Example:
        >>> backbone = ResNet50Backbone(pretrained=True)
        >>> features = backbone(x)
    """

    def __init__(self, pretrained: bool = True):
        super(ResNet50Backbone, self).__init__()

        # Load pre-trained ResNet50
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet50(weights=None)

        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 75x75
        self.layer2 = resnet.layer2  # 38x38
        self.layer3 = resnet.layer3  # 19x19
        self.layer4 = resnet.layer4  # 10x10

        # Output channels for each detection layer
        self.output_channels = [512, 512, 512, 256, 256, 128]

        # Extra layers for additional scales
        self.extra_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            ),
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass through ResNet50 backbone."""
        features = []

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        features.append(x)  # 75x75

        x = self.layer2(x)
        features.append(x)  # 38x38

        x = self.layer3(x)
        features.append(x)  # 19x19

        x = self.layer4(x)
        features.append(x)  # 10x10

        # Extra layers
        for extra_layer in self.extra_layers:
            x = extra_layer(x)
            features.append(x)

        # Return only 6 feature maps
        return features[-6:]


def create_backbone(
    backbone_name: str = 'mobilenet_v2',
    pretrained: bool = True
) -> nn.Module:
    """
    Create a backbone network.

    Args:
        backbone_name: Name of backbone ('mobilenet_v2' or 'resnet50')
        pretrained: Whether to use pre-trained weights

    Returns:
        Backbone network

    Example:
        >>> backbone = create_backbone('mobilenet_v2', pretrained=True)
    """
    if backbone_name == 'mobilenet_v2':
        return MobileNetV2Backbone(pretrained=pretrained)
    elif backbone_name == 'resnet50':
        return ResNet50Backbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")
