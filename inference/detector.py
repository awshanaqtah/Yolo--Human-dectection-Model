"""
Human Detection Inference Module.

This module provides a high-level interface for running
human detection on images and videos.
"""

import torch
from pathlib import Path
from typing import List, Dict, Union, Optional
from PIL import Image
import numpy as np

from ..models import SSD
from ..data.transforms import get_val_transform, ToTensor, Normalize, Compose, Resize
from .visualizer import draw_detections


class HumanDetector:
    """
    High-level detector for human detection.

    Args:
        model: SSD model
        device: Device to run inference on
        conf_threshold: Confidence threshold for detections
        nms_threshold: IoU threshold for NMS

    Example:
        >>> detector = HumanDetector(model, device='cuda')
        >>> results = detector.detect_image('image.jpg')
        >>> detector.visualize_results('image.jpg', results, 'output.jpg')
    """

    def __init__(
        self,
        model: SSD,
        device: str = 'cuda',
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45
    ):
        self.model = model
        self.device = device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        # Move model to device
        self.model.to(device)
        self.model.eval()

        # Get input size from model
        self.input_size = model.input_size

        # Create transform
        self.transform = get_val_transform(
            input_size=self.input_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from disk.

        Args:
            image_path: Path to image

        Returns:
            PIL Image

        Example:
            >>> image = detector.load_image('path/to/image.jpg')
        """
        image = Image.open(image_path).convert('RGB')
        return image

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for model input.

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor [1, 3, H, W]

        Example:
            >>> tensor = detector.preprocess(image)
        """
        # Apply transform
        image_tensor, _ = self.transform(image)

        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)

        return image_tensor

    @torch.no_grad()
    def detect(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect humans in an image.

        Args:
            image: Image (path, PIL Image, or tensor)
            conf_threshold: Override confidence threshold
            nms_threshold: Override NMS threshold

        Returns:
            Dictionary with detection results:
            {
                'boxes': [N, 4] detected boxes in [x1, y1, x2, y2] format (pixel coordinates),
                'scores': [N] confidence scores,
                'labels': [N] predicted labels,
                'image': Original PIL Image
            }

        Example:
            >>> results = detector.detect('image.jpg')
            >>> print(f"Detected {len(results['boxes'])} humans")
        """
        # Use default thresholds if not provided
        conf_threshold = conf_threshold if conf_threshold is not None else self.conf_threshold
        nms_threshold = nms_threshold if nms_threshold is not None else self.nms_threshold

        # Load image if path is provided
        if isinstance(image, str):
            original_image = self.load_image(image)
            image_tensor = self.preprocess(original_image)
        elif isinstance(image, Image.Image):
            original_image = image
            image_tensor = self.preprocess(image)
        elif isinstance(image, torch.Tensor):
            original_image = None
            image_tensor = image
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Move to device
        image_tensor = image_tensor.to(self.device)

        # Run detection
        detections = self.model.detect(
            image_tensor,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            top_k=200,
            keep_top_k=100
        )

        # Extract results (first image in batch)
        detection = detections[0]

        # Convert to numpy
        boxes = detection['boxes'].cpu().numpy()
        scores = detection['scores'].cpu().numpy()
        labels = detection['labels'].cpu().numpy()

        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels,
            'image': original_image
        }

    def detect_image(self, image_path: str) -> Dict:
        """
        Detect humans in an image file.

        Args:
            image_path: Path to image

        Returns:
            Detection results dictionary

        Example:
            >>> results = detector.detect_image('image.jpg')
        """
        return self.detect(image_path)

    def detect_batch(
        self,
        image_paths: List[str],
        batch_size: int = 8
    ) -> List[Dict]:
        """
        Detect humans in multiple images.

        Args:
            image_paths: List of image paths
            batch_size: Batch size for inference

        Returns:
            List of detection results

        Example:
            >>> results = detector.detect_batch(['img1.jpg', 'img2.jpg'])
        """
        all_results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]

            # Load and preprocess batch
            images = []
            original_images = []
            for path in batch_paths:
                img = self.load_image(path)
                original_images.append(img)
                img_tensor = self.preprocess(img)
                images.append(img_tensor)

            # Stack into batch
            batch_tensor = torch.cat(images, dim=0).to(self.device)

            # Run detection
            detections = self.model.detect(
                batch_tensor,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )

            # Convert to results
            for j, detection in enumerate(detections):
                all_results.append({
                    'boxes': detection['boxes'].cpu().numpy(),
                    'scores': detection['scores'].cpu().numpy(),
                    'labels': detection['labels'].cpu().numpy(),
                    'image': original_images[j]
                })

        return all_results

    def visualize_results(
        self,
        image: Union[str, Image.Image],
        results: Dict,
        output_path: Optional[str] = None,
        show_scores: bool = True,
        score_threshold: Optional[float] = None
    ) -> Image.Image:
        """
        Visualize detection results on an image.

        Args:
            image: Original image
            results: Detection results
            output_path: Path to save visualization (optional)
            show_scores: Whether to show confidence scores
            score_threshold: Minimum score to display

        Returns:
            PIL Image with drawn detections

        Example:
            >>> annotated = detector.visualize_results(
            ...     'image.jpg',
            ...     results,
            ...     'output.jpg'
            ... )
        """
        if isinstance(image, str):
            image = self.load_image(image)

        # Draw detections
        annotated_image = draw_detections(
            image=image,
            boxes=results['boxes'],
            scores=results['scores'],
            labels=results['labels'],
            class_names={1: 'person'},
            show_scores=show_scores,
            score_threshold=score_threshold
        )

        # Save if output path provided
        if output_path is not None:
            annotated_image.save(output_path)
            print(f"Saved visualization to {output_path}")

        return annotated_image

    def detect_and_visualize(
        self,
        image_path: str,
        output_path: str,
        conf_threshold: Optional[float] = None
    ) -> Dict:
        """
        Detect humans and save visualization.

        Args:
            image_path: Path to input image
            output_path: Path to save output image
            conf_threshold: Confidence threshold override

        Returns:
            Detection results

        Example:
            >>> results = detector.detect_and_visualize(
            ...     'input.jpg',
            ...     'output.jpg'
            ... )
        """
        # Run detection
        results = self.detect_image(image_path)

        # Visualize
        self.visualize_results(
            image=image_path,
            results=results,
            output_path=output_path,
            score_threshold=conf_threshold
        )

        return results


def load_detector(
    checkpoint_path: str,
    device: str = 'cuda',
    conf_threshold: float = 0.5,
    nms_threshold: float = 0.45
) -> HumanDetector:
    """
    Load a detector from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to run inference on
        conf_threshold: Confidence threshold
        nms_threshold: NMS threshold

    Returns:
        HumanDetector instance

    Example:
        >>> detector = load_detector('checkpoints/best_model.pth')
        >>> results = detector.detect('image.jpg')
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    model = SSD(
        num_classes=checkpoint.get('num_classes', 2),
        input_size=tuple(checkpoint.get('input_size', (300, 300))),
        backbone=checkpoint.get('backbone', 'mobilenet_v2'),
        pretrained=False
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])

    # Create detector
    detector = HumanDetector(
        model=model,
        device=device,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold
    )

    print(f"Loaded detector from {checkpoint_path}")

    return detector
