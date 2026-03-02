"""
Image detection script for Human Detection Model.

This script runs human detection on images and saves visualizations.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference import load_detector


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect humans in images')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output image or directory')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    return parser.parse_args()


def main():
    """Main detection function."""
    args = parse_args()

    print("=" * 60)
    print("Human Detection - Image Inference")
    print("=" * 60)

    # Load detector
    print(f"\nLoading detector from {args.checkpoint}")
    detector = load_detector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold
    )

    # Process input
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        print(f"\nProcessing: {input_path}")

        # Determine output path
        if args.output is None:
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        else:
            output_path = Path(args.output)

        # Detect and visualize
        results = detector.detect_and_visualize(
            str(input_path),
            str(output_path)
        )

        print(f"Detected {len(results['boxes'])} humans")
        print(f"Output saved to: {output_path}")

    elif input_path.is_dir():
        # Directory of images
        output_dir = Path(args.output) if args.output else input_path / 'detected'
        output_dir.mkdir(exist_ok=True)

        # Find all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))

        print(f"\nFound {len(image_files)} images")
        print(f"Output directory: {output_dir}")

        # Process each image
        for i, image_file in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_file.name}")

            output_file = output_dir / image_file.name

            try:
                results = detector.detect_and_visualize(
                    str(image_file),
                    str(output_file)
                )
                print(f"  Detected {len(results['boxes'])} humans")
            except Exception as e:
                print(f"  Error: {e}")

    else:
        print(f"Error: Input path not found: {args.input}")
        return

    print("\nDetection complete!")


if __name__ == '__main__':
    main()
