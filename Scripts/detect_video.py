"""
Video detection script for Human Detection Model.

This script runs human detection on videos and saves annotated videos.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from inference import load_detector
from inference.video_processor import VideoProcessor


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Detect humans in videos')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input video')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output video')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--show-live', action='store_true',
                        help='Display video in real-time while processing')
    parser.add_argument('--webcam', type=int, default=None,
                        help='Use webcam with specified camera ID')

    return parser.parse_args()


def main():
    """Main detection function."""
    args = parse_args()

    print("=" * 60)
    print("Human Detection - Video Inference")
    print("=" * 60)

    # Load detector
    print(f"\nLoading detector from {args.checkpoint}")
    detector = load_detector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        conf_threshold=args.conf_threshold
    )

    # Create video processor
    processor = VideoProcessor(
        detector=detector,
        output_dir='inference/outputs'
    )

    if args.webcam is not None:
        # Webcam mode
        print(f"\nStarting webcam detection (camera {args.webcam})")
        print("Press 'q' to quit")
        processor.process_webcam(camera_id=args.webcam)

    else:
        # Video file mode
        input_path = Path(args.input)

        if not input_path.exists():
            print(f"Error: Input video not found: {args.input}")
            return

        # Determine output path
        if args.output is None:
            output_path = input_path.parent / f"{input_path.stem}_detected.mp4"
        else:
            output_path = Path(args.output)

        print(f"\nInput: {input_path}")
        print(f"Output: {output_path}")

        # Process video
        results = processor.process_video(
            input_path=str(input_path),
            output_path=str(output_path),
            show_live=args.show_live
        )

        print(f"\nProcessing complete!")
        print(f"Output saved to: {output_path}")
        print(f"Total detections: {results['total_detections']}")
        print(f"Average FPS: {results['avg_fps']:.2f}")


if __name__ == '__main__':
    main()
