"""
SSD (Single Shot MultiBox Detector) Model for Human Detection.

This module implements the complete SSD architecture combining
backbone, detection head, and post-processing.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from .backbone import create_backbone
from .ssd_head import SSDHead, get_num_priors_per_feature, calculate_total_priors
from .prior_box import PriorBox, decode_boxes, convert_boxes_to_xyxy
from .loss import MultiBoxLoss


class SSD(nn.Module):
    """
    Single Shot MultiBox Detector for human detection.

    Args:
        num_classes: Number of classes (including background)
        input_size: Input image size (height, width)
        backbone: Backbone network name ('mobilenet_v2' or 'resnet50')
        pretrained: Whether to use pre-trained backbone

    Example:
        >>> model = SSD(num_classes=2, input_size=(300, 300))
        >>> predictions = model(x)
        >>> # predictions: (class_scores, box_offsets, priors)
    """

    def __init__(
        self,
        num_classes: int = 2,
        input_size: Tuple[int, int] = (300, 300),
        backbone: str = 'mobilenet_v2',
        pretrained: bool = True
    ):
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.backbone_name = backbone

        # Feature map configuration for SSD300
        self.feature_maps = [38, 19, 10, 5, 3, 1]

        # Prior box configuration
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        # Create backbone
        self.backbone = create_backbone(backbone, pretrained=pretrained)

        # Get number of priors for each feature map
        self.num_priors = get_num_priors_per_feature(self.feature_maps, self.aspect_ratios)

        # Create detection head
        self.detection_head = SSDHead(
            num_classes=num_classes,
            in_channels=self.backbone.output_channels,
            num_priors=self.num_priors,
            aspect_ratios=self.aspect_ratios
        )

        # Generate prior boxes
        self.prior_box = PriorBox(
            input_size=input_size,
            feature_maps=self.feature_maps,
            min_sizes=self.min_sizes,
            max_sizes=self.max_sizes,
            aspect_ratios=self.aspect_ratios
        )
        self.priors = self.prior_box.forward()  # [num_priors, 4]

        # Total number of prior boxes
        self.total_priors = calculate_total_priors(self.feature_maps, self.num_priors)
        print(f"SSD initialized with {self.total_priors} prior boxes")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SSD.

        Args:
            x: Input tensor [batch_size, 3, height, width]

        Returns:
            Tuple of (class_predictions, loc_predictions)
            - class_predictions: [batch_size, num_priors, num_classes]
            - loc_predictions: [batch_size, num_priors, 4]

        Example:
            >>> class_pred, loc_pred = model(x)
            >>> print(class_pred.shape)  # [batch, 8732, num_classes]
            >>> print(loc_pred.shape)    # [batch, 8732, 4]
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Apply detection head
        class_pred, loc_pred = self.detection_head(features)

        return class_pred, loc_pred

    def detect(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        top_k: int = 200,
        keep_top_k: int = 100
    ) -> List[Dict]:
        """
        Perform object detection on input images.

        Args:
            x: Input tensor [batch_size, 3, height, width]
            conf_threshold: Confidence threshold for filtering
            nms_threshold: IoU threshold for Non-Maximum Suppression
            top_k: Maximum number of detections to keep before NMS
            keep_top_k: Maximum number of detections to keep after NMS

        Returns:
            List of detections for each image:
            [{
                'boxes': [N, 4] detected boxes in [x1, y1, x2, y2] format,
                'scores': [N] confidence scores,
                'labels': [N] predicted labels
            }]

        Example:
            >>> detections = model.detect(images)
            >>> boxes = detections[0]['boxes']
            >>> scores = detections[0]['scores']
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            # Forward pass
            class_pred, loc_pred = self.forward(x)

            # Process each image in the batch
            batch_size = x.size(0)
            detections = []

            for idx in range(batch_size):
                # Get predictions for this image
                class_scores = class_pred[idx]  # [num_priors, num_classes]
                loc_preds = loc_pred[idx]       # [num_priors, 4]

                # Get max class scores and labels for each prior
                max_scores, max_labels = class_scores.max(dim=1)  # [num_priors]

                # Filter by confidence threshold
                mask = max_scores > conf_threshold

                if mask.sum() == 0:
                    # No detections
                    detections.append({
                        'boxes': torch.zeros((0, 4), device=x.device),
                        'scores': torch.zeros((0,), device=x.device),
                        'labels': torch.zeros((0,), dtype=torch.long, device=x.device)
                    })
                    continue

                # Filter predictions
                filtered_scores = max_scores[mask]
                filtered_labels = max_labels[mask]
                filtered_loc = loc_preds[mask]
                filtered_priors = self.priors.to(x.device)[mask]

                # Decode boxes
                decoded_boxes = decode_boxes(
                    filtered_loc.unsqueeze(0),
                    filtered_priors
                ).squeeze(0)  # [N, 4] in [cx, cy, w, h] format

                # Convert to [x1, y1, x2, y2] format
                decoded_boxes = convert_boxes_to_xyxy(decoded_boxes)

                # Clip to image boundaries
                decoded_boxes[:, [0, 2]] = torch.clamp(decoded_boxes[:, [0, 2]], 0, 1)
                decoded_boxes[:, [1, 3]] = torch.clamp(decoded_boxes[:, [1, 3]], 0, 1)

                # Scale to input size
                decoded_boxes[:, [0, 2]] *= self.input_size[1]  # width
                decoded_boxes[:, [1, 3]] *= self.input_size[0]  # height

                # Keep only top_k detections
                if filtered_scores.size(0) > top_k:
                    top_k_scores, top_k_idx = filtered_scores.topk(top_k)
                    filtered_scores = top_k_scores
                    filtered_labels = filtered_labels[top_k_idx]
                    decoded_boxes = decoded_boxes[top_k_idx]

                # Apply Non-Maximum Suppression
                keep = self.nms(decoded_boxes, filtered_scores, nms_threshold)

                # Keep only keep_top_k after NMS
                if keep.sum() > keep_top_k:
                    keep = keep[filtered_scores[keep].topk(keep_top_k)[1]]

                # Final detections
                final_boxes = decoded_boxes[keep]
                final_scores = filtered_scores[keep]
                final_labels = filtered_labels[keep]

                detections.append({
                    'boxes': final_boxes,
                    'scores': final_scores,
                    'labels': final_labels
                })

        return detections

    def nms(
        self,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        threshold: float
    ) -> torch.Tensor:
        """
        Non-Maximum Suppression.

        Args:
            boxes: [N, 4] boxes in [x1, y1, x2, y2] format
            scores: [N] confidence scores
            threshold: IoU threshold

        Returns:
            Boolean tensor indicating which boxes to keep
        """
        if boxes.size(0) == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes.device)

        # Sort by score (descending)
        sorted_scores, sorted_idx = scores.sort(descending=True)

        # Initialize keep mask
        keep = torch.ones(boxes.size(0), dtype=torch.bool, device=boxes.device)

        # Process each box
        for i in range(boxes.size(0)):
            if not keep[i]:
                continue

            # Get current box
            box_i = boxes[sorted_idx[i]]

            # Compare with all remaining boxes
            for j in range(i + 1, boxes.size(0)):
                if not keep[j]:
                    continue

                box_j = boxes[sorted_idx[j]]

                # Compute IoU
                iou = self.compute_iou(box_i, box_j)

                # Suppress if IoU > threshold
                if iou > threshold:
                    keep[j] = False

        return keep[sorted_idx].argsort()  # Return in original order

    def compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """
        Compute IoU between two boxes.

        Args:
            box1: [4] box in [x1, y1, x2, y2] format
            box2: [4] box in [x1, y1, x2, y2] format

        Returns:
            IoU value
        """
        # Compute intersection
        inter_x1 = torch.max(box1[0], box2[0])
        inter_y1 = torch.max(box1[1], box2[1])
        inter_x2 = torch.min(box1[2], box2[2])
        inter_y2 = torch.min(box1[3], box2[3])

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

        # Compute union
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area

        return inter_area / (union_area + 1e-7)

    def get_loss_criterion(self) -> MultiBoxLoss:
        """
        Get the loss function for training.

        Returns:
            MultiBoxLoss instance

        Example:
            >>> criterion = model.get_loss_criterion()
            >>> loss, loss_dict = criterion(predictions, targets, priors)
        """
        return MultiBoxLoss(num_classes=self.num_classes)

    def freeze_backbone(self):
        """Freeze backbone parameters (for fine-tuning only the head)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen")


def create_ssd(
    num_classes: int = 2,
    input_size: Tuple[int, int] = (300, 300),
    backbone: str = 'mobilenet_v2',
    pretrained: bool = True
) -> SSD:
    """
    Convenience function to create an SSD model.

    Args:
        num_classes: Number of classes (including background)
        input_size: Input image size
        backbone: Backbone network name
        pretrained: Whether to use pre-trained backbone

    Returns:
        SSD model

    Example:
        >>> model = create_ssd(num_classes=2, backbone='mobilenet_v2')
    """
    return SSD(
        num_classes=num_classes,
        input_size=input_size,
        backbone=backbone,
        pretrained=pretrained
    )
