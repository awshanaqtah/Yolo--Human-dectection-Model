"""
Video processing utilities for Human Detection Model.

This module provides functions to process videos, detect humans
in each frame, and save annotated videos.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict
import time

from .detector import HumanDetector


class VideoProcessor:
    """
    Process videos for human detection.

    Args:
        detector: HumanDetector instance
        output_dir: Directory to save output videos
        fps: Output video FPS (None = use input FPS)
        display_size: Size to display frames (None = use original size)

    Example:
        >>> processor = VideoProcessor(detector, output_dir='outputs')
        >>> processor.process_video('input.mp4', 'output.mp4')
    """

    def __init__(
        self,
        detector: HumanDetector,
        output_dir: str = 'inference/outputs',
        fps: Optional[float] = None,
        display_size: Optional[tuple] = None
    ):
        self.detector = detector
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.display_size = display_size

    def process_video(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        show_live: bool = False,
        save_output: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict:
        """
        Process a video file for human detection.

        Args:
            input_path: Path to input video
            output_path: Path to output video (optional)
            show_live: Whether to display video in real-time
            save_output: Whether to save output video
            progress_callback: Optional callback(frame_num, total_frames)

        Returns:
            Dictionary with processing results

        Example:
            >>> results = processor.process_video(
            ...     'input.mp4',
            ...     'output.mp4',
            ...     show_live=True
            ... )
        """
        # Open video capture
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing video: {input_path}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {video_fps:.2f}")
        print(f"  Total frames: {total_frames}")

        # Setup output video writer
        writer = None
        if save_output:
            if output_path is None:
                output_path = self.output_dir / f"processed_{Path(input_path).stem}.mp4"
            else:
                output_path = Path(output_path)

            output_path = str(output_path)

            # Use specified FPS or input FPS
            out_fps = self.fps if self.fps is not None else video_fps

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

        # Process frames
        frame_count = 0
        total_detections = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize frame for display if specified
            if self.display_size is not None:
                frame_rgb = cv2.resize(frame_rgb, self.display_size)

            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)

            # Run detection
            results = self.detector.detect(pil_image)

            # Draw detections
            annotated = self.detector.visualize_results(
                image=pil_image,
                results=results,
                show_scores=True
            )

            # Convert back to BGR for OpenCV
            annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)

            # Write to output video
            if writer is not None:
                # Resize back to original size if needed
                if self.display_size is not None:
                    annotated_bgr = cv2.resize(annotated_bgr, (width, height))
                writer.write(annotated_bgr)

            # Display live
            if show_live:
                # Resize for display
                display_frame = annotated_bgr
                if width > 1280:
                    scale = 1280 / width
                    display_frame = cv2.resize(display_frame, None, fx=scale, fy=scale)

                cv2.imshow('Human Detection', display_frame)

                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Update progress
            total_detections += len(results['boxes'])

            if progress_callback is not None:
                progress_callback(frame_count, total_frames)

            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"Processed {frame_count}/{total_frames} frames ({fps_current:.1f} FPS)")

        # Release resources
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_fps = frame_count / elapsed_time

        results = {
            'input_path': input_path,
            'output_path': output_path if save_output else None,
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_fps': avg_fps,
            'processing_time': elapsed_time
        }

        print(f"\nVideo processing complete!")
        print(f"  Output: {output_path if save_output else 'Not saved'}")
        print(f"  Total detections: {total_detections}")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Processing time: {elapsed_time:.2f}s")

        return results

    def process_webcam(
        self,
        camera_id: int = 0,
        display_size: tuple = (1280, 720),
        duration: Optional[int] = None
    ):
        """
        Process live webcam feed.

        Args:
            camera_id: Camera device ID
            display_size: Size to display frames
            duration: Duration in seconds (None = run until 'q' pressed)

        Example:
            >>> processor.process_webcam(camera_id=0)
        """
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")

        print("Starting webcam detection. Press 'q' to quit.")

        start_time = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # Check duration
            if duration is not None and (time.time() - start_time) > duration:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize for display
            frame_rgb = cv2.resize(frame_rgb, display_size)

            # Convert to PIL Image
            from PIL import Image
            pil_image = Image.fromarray(frame_rgb)

            # Run detection
            results = self.detector.detect(pil_image)

            # Draw detections
            annotated = self.detector.visualize_results(
                image=pil_image,
                results=results,
                show_scores=True
            )

            # Convert back to BGR
            annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)

            # Calculate FPS
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Draw FPS counter
            cv2.putText(
                annotated_bgr,
                f"FPS: {fps:.1f} | Detections: {len(results['boxes'])}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display
            cv2.imshow('Webcam Human Detection', annotated_bgr)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nWebcam detection stopped.")
        print(f"  Processed {frame_count} frames in {elapsed:.1f}s")
        print(f"  Average FPS: {fps:.1f}")


def process_video_stream(
    detector: HumanDetector,
    input_path: str,
    output_path: str,
    conf_threshold: float = 0.5
):
    """
    Convenience function to process a video file.

    Args:
        detector: HumanDetector instance
        input_path: Input video path
        output_path: Output video path
        conf_threshold: Confidence threshold

    Example:
        >>> process_video_stream(detector, 'input.mp4', 'output.mp4')
    """
    processor = VideoProcessor(detector)
    results = processor.process_video(input_path, output_path)
    return results
