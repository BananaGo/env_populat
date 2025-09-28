#!/usr/bin/env python3
"""
Audio-Video Merger Script
Combines video with synchronized audio track for sound effects
"""

import numpy as np
import subprocess
import os
import tempfile
import math
from typing import List, Tuple

class AudioVideoMerger:
    """Merges video with audio track containing sound effects"""

    def __init__(self):
        self.audio_events = []  # List of (timestamp, sound_file) tuples

    def add_sound_event(self, timestamp: float, sound_type: str):
        """Add a sound event at specific timestamp"""
        assets_dir = os.path.join(os.path.dirname(__file__), "assets")

        sound_files = {
            "food_eaten": "tiktok_pop.mp3",
            "episode_start": "tiktok_bling.mp3"
        }

        if sound_type in sound_files:
            sound_path = os.path.join(assets_dir, sound_files[sound_type])
            if os.path.exists(sound_path):
                self.audio_events.append((timestamp, sound_path))
                print(f"🎵 Added {sound_type} sound at {timestamp:.2f}s")

    def create_audio_track(self, duration: float, output_path: str):
        """Create audio track with all sound events"""
        if not self.audio_events:
            print("⚠️ No audio events to process")
            return False

        try:
            # Limit the number of audio events to prevent FFmpeg complexity issues
            max_events = 50
            events_to_use = self.audio_events[:max_events] if len(self.audio_events) > max_events else self.audio_events

            if len(self.audio_events) > max_events:
                print(f"🔧 Limiting to first {max_events} audio events (had {len(self.audio_events)})")

            # Create a simpler approach - create separate audio files and mix them
            temp_audio_files = []

            # Create individual delayed audio files
            for i, (timestamp, sound_path) in enumerate(events_to_use):
                temp_file = f"/tmp/delayed_audio_{i}.wav"
                temp_audio_files.append(temp_file)

                # Create delayed audio file
                delay_cmd = [
                    "ffmpeg", "-y",
                    "-i", sound_path,
                    "-af", f"adelay={int(timestamp*1000)}|{int(timestamp*1000)},apad=pad_dur={duration}",
                    "-t", str(duration),
                    temp_file
                ]

                result = subprocess.run(delay_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode != 0:
                    print(f"⚠️ Failed to create delayed audio {i}: {result.stderr}")

            if not temp_audio_files:
                print("❌ No audio files created")
                return False

            # Mix all delayed audio files together
            if len(temp_audio_files) == 1:
                # Simple case - just one audio file
                mix_cmd = ["ffmpeg", "-y", "-i", temp_audio_files[0], "-c:a", "aac", "-b:a", "128k", output_path]
            else:
                # Mix multiple files
                mix_cmd = ["ffmpeg", "-y"]
                for temp_file in temp_audio_files:
                    mix_cmd.extend(["-i", temp_file])

                # Create amix filter
                amix_filter = f"amix=inputs={len(temp_audio_files)}:duration=longest"
                mix_cmd.extend(["-filter_complex", amix_filter, "-c:a", "aac", "-b:a", "128k", output_path])

            print(f"🔧 Mixing {len(temp_audio_files)} audio tracks...")
            result = subprocess.run(mix_cmd, capture_output=True, text=True, timeout=30)

            # Cleanup temp files
            for temp_file in temp_audio_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass

            if result.returncode == 0:
                print(f"✅ Audio track created: {output_path}")
                return True
            else:
                print(f"❌ Audio mixing failed: {result.stderr}")
                return False

        except Exception as e:
            print(f"❌ Audio creation error: {e}")
            return False

    def merge_video_audio(self, video_path: str, output_path: str):
        """Merge video with generated audio track"""
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return False

        if not self.audio_events:
            print("⚠️ No audio events, copying video as-is")
            import shutil
            shutil.copy2(video_path, output_path)
            return True

        try:
            # Get video duration
            probe_cmd = [
                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                "-of", "csv=p=0", video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip())

            # Create temporary audio track
            with tempfile.NamedTemporaryFile(suffix=".aac", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name

            if self.create_audio_track(duration, temp_audio_path):
                # Merge video and audio
                merge_cmd = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-i", temp_audio_path,
                    "-c:v", "copy",  # Copy video without re-encoding
                    "-c:a", "aac",
                    "-map", "0:v:0", "-map", "1:a:0",
                    output_path
                ]

                print(f"🎬 Merging video and audio...")
                result = subprocess.run(merge_cmd, capture_output=True, text=True, timeout=60)

                # Cleanup temp file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)

                if result.returncode == 0:
                    print(f"✅ Video with audio saved: {output_path}")
                    return True
                else:
                    print(f"❌ Merge failed: {result.stderr}")
                    return False
            else:
                # Cleanup temp file
                if os.path.exists(temp_audio_path):
                    os.remove(temp_audio_path)
                return False

        except Exception as e:
            print(f"❌ Merge error: {e}")
            return False

def enhance_snake_video_with_audio():
    """Enhance the snake video by adding audio track"""

    video_file = "snake_master_growing_longer.mp4"
    output_file = "snake_master_growing_longer_with_audio.mp4"

    if not os.path.exists(video_file):
        print(f"❌ Video file not found: {video_file}")
        print("Please run the snake game first to generate the video.")
        return

    merger = AudioVideoMerger()

    # Add example sound events (you would collect these during game recording)
    # These are just examples - in real implementation, you'd track these during gameplay
    merger.add_sound_event(2.0, "episode_start")  # Episode 1 starts
    merger.add_sound_event(5.5, "food_eaten")     # First food eaten
    merger.add_sound_event(8.2, "food_eaten")     # Second food eaten
    merger.add_sound_event(12.0, "episode_start") # Episode 2 starts
    merger.add_sound_event(15.8, "food_eaten")    # More food eaten

    success = merger.merge_video_audio(video_file, output_file)

    if success:
        print(f"🎉 Enhanced video created: {output_file}")
    else:
        print("❌ Failed to create enhanced video")

if __name__ == "__main__":
    enhance_snake_video_with_audio()