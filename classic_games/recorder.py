"""
Game Recording Utility for YouTube Shorts creation.
"""

import cv2
import numpy as np
import pygame
from typing import List, Optional, Tuple
import os
from datetime import datetime


class GameRecorder:
    """Records gameplay footage optimized for YouTube Shorts."""

    def __init__(self,
                 output_dir: str = "recordings",
                 fps: int = 60,
                 quality: str = "high"):
        """
        Initialize game recorder.

        Args:
            output_dir: Directory to save recordings
            fps: Frames per second for recording
            quality: Recording quality ('low', 'medium', 'high')
        """
        self.output_dir = output_dir
        self.fps = fps
        self.quality = quality

        # Quality settings
        quality_settings = {
            'low': {'bitrate': 1000, 'crf': 28},
            'medium': {'bitrate': 2500, 'crf': 23},
            'high': {'bitrate': 5000, 'crf': 18}
        }
        self.bitrate = quality_settings[quality]['bitrate']
        self.crf = quality_settings[quality]['crf']

        # Recording state
        self.recording = False
        self.frames = []
        self.start_time = None
        self.video_writer = None

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def start_recording(self, game_name: str = "game") -> str:
        """Start recording gameplay."""
        if self.recording:
            self.stop_recording()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_name}_{timestamp}.mp4"
        self.output_path = os.path.join(self.output_dir, filename)

        self.recording = True
        self.frames = []
        self.start_time = datetime.now()

        print(f"🎥 Started recording: {filename}")
        return self.output_path

    def add_frame(self, frame: np.ndarray):
        """Add a frame to the recording."""
        if not self.recording:
            return

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            bgr_frame = frame

        self.frames.append(bgr_frame.copy())

        # Memory management - save to file if too many frames in memory
        if len(self.frames) >= 300:  # ~5 seconds at 60fps
            self._flush_frames()

    def stop_recording(self) -> Optional[str]:
        """Stop recording and save video file."""
        if not self.recording:
            return None

        self.recording = False

        if not self.frames:
            print("❌ No frames recorded")
            return None

        # Get video dimensions from first frame
        height, width = self.frames[0].shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (width, height)
        )

        # Write remaining frames
        self._flush_frames()

        # Release video writer
        if self.video_writer:
            self.video_writer.release()

        duration = datetime.now() - self.start_time
        print(f"✅ Recording saved: {self.output_path}")
        print(f"📊 Duration: {duration.seconds}.{duration.microseconds//1000:03d}s")
        print(f"🎬 Frames: {len(self.frames)} at {self.fps}fps")

        # Optimize for YouTube Shorts
        optimized_path = self._optimize_for_youtube_shorts()

        return optimized_path or self.output_path

    def _flush_frames(self):
        """Write frames to video file."""
        if not self.frames:
            return

        # Initialize video writer if needed
        if not self.video_writer and hasattr(self, 'output_path'):
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (width, height)
            )

        # Write frames
        if self.video_writer:
            for frame in self.frames:
                self.video_writer.write(frame)

        # Clear frames from memory
        self.frames = []

    def _optimize_for_youtube_shorts(self) -> Optional[str]:
        """Optimize video for YouTube Shorts format."""
        try:
            # Create optimized filename
            base_name = os.path.splitext(self.output_path)[0]
            optimized_path = f"{base_name}_optimized.mp4"

            print("🔧 Optimizing for YouTube Shorts...")

            # Use ffmpeg to optimize (if available)
            import subprocess

            cmd = [
                'ffmpeg', '-i', self.output_path,
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', str(self.crf),
                '-c:a', 'aac',
                '-b:a', '128k',
                '-movflags', '+faststart',
                '-y',  # Overwrite output file
                optimized_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Optimized video: {optimized_path}")

                # Compare file sizes
                original_size = os.path.getsize(self.output_path)
                optimized_size = os.path.getsize(optimized_path)
                compression_ratio = (1 - optimized_size / original_size) * 100

                print(f"📦 Size reduction: {compression_ratio:.1f}%")

                return optimized_path
            else:
                print(f"⚠️ FFmpeg optimization failed: {result.stderr}")
                return None

        except (ImportError, FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"⚠️ Optimization skipped (ffmpeg not available): {e}")
            return None

    def create_highlight_reel(self,
                            frame_sequences: List[List[np.ndarray]],
                            game_name: str,
                            transition_frames: int = 30) -> str:
        """Create a highlight reel from multiple game sequences."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{game_name}_highlights_{timestamp}.mp4"
        output_path = os.path.join(self.output_dir, filename)

        print(f"🎬 Creating highlight reel: {filename}")

        if not frame_sequences:
            print("❌ No frame sequences provided")
            return ""

        # Get dimensions from first frame
        first_frame = frame_sequences[0][0]
        height, width = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))

        total_frames = 0

        for i, sequence in enumerate(frame_sequences):
            print(f"📝 Adding sequence {i+1}/{len(frame_sequences)} ({len(sequence)} frames)")

            # Add frames from sequence
            for frame in sequence:
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    bgr_frame = frame

                video_writer.write(bgr_frame)
                total_frames += 1

            # Add transition if not last sequence
            if i < len(frame_sequences) - 1:
                self._add_transition_effect(video_writer, first_frame, transition_frames)
                total_frames += transition_frames

        video_writer.release()

        duration = total_frames / self.fps
        print(f"✅ Highlight reel created: {output_path}")
        print(f"📊 Duration: {duration:.2f}s, Frames: {total_frames}")

        # Optimize for YouTube
        optimized_path = self._optimize_for_youtube_shorts()
        return optimized_path or output_path

    def _add_transition_effect(self, video_writer, reference_frame: np.ndarray, frames: int):
        """Add transition effect between sequences."""
        height, width = reference_frame.shape[:2]

        for i in range(frames):
            # Create fade to black transition
            alpha = abs(math.sin((i / frames) * math.pi))

            # Create black frame
            black_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Blend with reference frame for smooth transition
            transition_frame = black_frame.copy()

            # Add some visual interest
            center_x, center_y = width // 2, height // 2
            radius = int((i / frames) * min(width, height) // 2)

            if radius > 0:
                cv2.circle(transition_frame, (center_x, center_y), radius, (50, 50, 150), -1)

            video_writer.write(transition_frame)

    def add_text_overlay(self, frame: np.ndarray, text: str,
                        position: Tuple[int, int] = None,
                        font_scale: float = 1.0,
                        color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """Add text overlay to frame."""
        frame_with_text = frame.copy()

        if position is None:
            position = (50, frame.shape[0] - 50)

        # Add text with border for better visibility
        cv2.putText(frame_with_text, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3)  # Black border
        cv2.putText(frame_with_text, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)      # Colored text

        return frame_with_text

    def get_recording_stats(self) -> dict:
        """Get current recording statistics."""
        if not self.recording:
            return {'status': 'not_recording'}

        current_time = datetime.now()
        duration = current_time - self.start_time if self.start_time else 0

        return {
            'status': 'recording',
            'duration_seconds': duration.total_seconds() if duration else 0,
            'frames_recorded': len(self.frames),
            'estimated_size_mb': len(self.frames) * 0.1,  # Rough estimate
            'fps': self.fps
        }