import numpy as np
import pygame
import pygame.mixer
import time
import random
import imageio
import os
import subprocess
from classic_games.env_collection import env_raw_snake


class SoundManager:
    """Manages sound effects for the snake game"""

    def __init__(self):
        self.enabled = True
        self.sounds = {}
        self.recorder = None  # Will be set later
        self._init_mixer()
        self._load_sounds()

    def set_recorder(self, recorder):
        """Link recorder for audio event tracking"""
        self.recorder = recorder

    def _init_mixer(self):
        """Initialize pygame mixer for audio playback"""
        try:
            pygame.mixer.pre_init(frequency=22050, size=-16, channels=2, buffer=512)
            pygame.mixer.init()
            print("🔊 Audio system initialized")
        except pygame.error as e:
            print(f"⚠️ Audio initialization failed: {e}")
            self.enabled = False

    def _load_sounds(self):
        """Load sound files from assets directory"""
        if not self.enabled:
            return

        assets_dir = os.path.join(os.path.dirname(__file__), "assets")
        sound_files = {
            "food_eaten": "tiktok_pop.mp3",  # Sound when snake eats food
            "episode_start": "tiktok_bling.mp3"   # Sound when new episode starts
        }

        for sound_name, filename in sound_files.items():
            filepath = os.path.join(assets_dir, filename)
            try:
                if os.path.exists(filepath):
                    self.sounds[sound_name] = pygame.mixer.Sound(filepath)
                    print(f"🎵 Loaded {sound_name}: {filename}")
                else:
                    print(f"⚠️ Sound file not found: {filepath}")
            except pygame.error as e:
                print(f"⚠️ Failed to load {filename}: {e}")

    def play_sound(self, sound_name):
        """Play a sound effect"""
        if not self.enabled:
            print(f"🔇 Audio disabled, cannot play {sound_name}")
            return

        if sound_name not in self.sounds:
            print(f"⚠️ Sound '{sound_name}' not loaded")
            return

        try:
            # Stop any currently playing instances of this sound
            self.sounds[sound_name].stop()
            # Play the sound
            self.sounds[sound_name].play()
            print(f"🔊 Playing sound: {sound_name}")

            # Track the audio event for video post-processing
            if self.recorder:
                self.recorder.add_audio_event(sound_name)

        except pygame.error as e:
            print(f"⚠️ Failed to play {sound_name}: {e}")

    def play_food_eaten(self):
        """Play sound when snake eats food"""
        self.play_sound("food_eaten")

    def play_episode_start(self):
        """Play sound when new episode starts"""
        self.play_sound("episode_start")

    def set_volume(self, volume):
        """Set volume for all sounds (0.0 to 1.0)"""
        if not self.enabled:
            return

        for sound in self.sounds.values():
            sound.set_volume(volume)

    def cleanup(self):
        """Clean up audio resources"""
        if self.enabled:
            pygame.mixer.quit()


class MasterSnakeAI:
    def __init__(self):
        """Master Snake AI that gets progressively better at growing longer"""
        self.episode = 0

    def get_skill_level(self):
        """Return current skill level (0-1)"""
        # Rapid improvement over episodes
        return min(0.95, 0.4 + (self.episode * 0.15))

    def get_noise_level(self):
        """Get current randomness level"""
        skill = self.get_skill_level()
        return max(0.02, 0.3 * (1 - skill))

    def predict_collision(self, head_x, head_y, action, steps_ahead=3):
        """Predict if an action will cause collision"""
        # Movement deltas per action
        deltas = {
            119: (0, -1.5),  # up
            115: (0, 1.5),   # down
            97: (-1.5, 0),   # left
            100: (1.5, 0)    # right
        }

        if action not in deltas:
            return True

        dx, dy = deltas[action]
        future_x = head_x + (dx * steps_ahead)
        future_y = head_y + (dy * steps_ahead)

        # Check boundaries (refined from testing)
        return future_x < 1 or future_x > 63 or future_y < 1 or future_y > 47

    def calculate_food_direction_score(self, head_x, head_y, food_x, food_y, action):
        """Calculate how good an action is for reaching food"""
        deltas = {119: (0, -1), 115: (0, 1), 97: (-1, 0), 100: (1, 0)}

        if action not in deltas:
            return -100

        dx, dy = deltas[action]
        new_x = head_x + dx
        new_y = head_y + dy

        # Distance to food after this move
        distance = abs(food_x - new_x) + abs(food_y - new_y)

        # Score is inverse of distance (closer = better)
        return -distance

    def choose_action(self, game_state):
        """Choose the best action to eat food and avoid walls"""
        head_x = game_state['snake_head_x']
        head_y = game_state['snake_head_y']
        food_x = game_state['food_x']
        food_y = game_state['food_y']

        actions = [119, 115, 97, 100]  # up, down, left, right
        noise_level = self.get_noise_level()

        # Sometimes be random for learning effect
        if random.random() < noise_level:
            random.shuffle(actions)
            for action in actions:
                if not self.predict_collision(head_x, head_y, action):
                    return action
            return actions[0]  # desperate fallback

        # Smart strategy: find best action
        action_scores = []
        for action in actions:
            if self.predict_collision(head_x, head_y, action):
                continue  # Skip unsafe actions

            # Score based on how well it moves toward food
            score = self.calculate_food_direction_score(head_x, head_y, food_x, food_y, action)
            action_scores.append((action, score))

        if action_scores:
            # Sort by score (higher is better)
            action_scores.sort(key=lambda x: x[1], reverse=True)
            return action_scores[0][0]

        # Emergency: just try to survive
        for action in actions:
            if not self.predict_collision(head_x, head_y, action, steps_ahead=1):
                return action

        return 119  # up as last resort


class MasterVideoRecorder:
    def __init__(self, filename="master_snake_growing.mp4", fps=60):
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.recording = False
        self.target_width = 1080
        self.target_height = 1920
        self.video_writer = None
        self.audio_events = []  # Track audio events for post-processing
        self.start_time = None

    def start_recording(self):
        self.recording = True
        self.frames = []
        self.audio_events = []
        self.start_time = time.time()
        print(f"🎥 Recording MASTER Snake AI: {self.filename}")

    def add_audio_event(self, event_type: str):
        """Track audio events with timestamps for post-processing"""
        if self.recording and self.start_time:
            timestamp = time.time() - self.start_time
            self.audio_events.append((timestamp, event_type))
            print(f"🎵 Audio event: {event_type} at {timestamp:.2f}s")

    def create_overlay(self, frame, episode, score, total_food, current_length, skill_level, elapsed_time):
        """Create minimal overlay showing just snake growth"""
        try:
            import cv2
            overlay_frame = frame.copy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 3.0
            thickness = 8

            # Single attention-grabbing line showing snake growth
            if total_food > 0:
                growth_text = f"Growing! (+{total_food} food)"
                text_color = (0, 255, 100)  # Bright green for success

                # Center the text horizontally
                text_size = cv2.getTextSize(growth_text, font, font_scale, thickness)[0]
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = 120  # Top of screen for maximum visibility

                cv2.putText(overlay_frame, growth_text, (text_x, text_y), font, font_scale, text_color, thickness)

            """
                else:
                    # Multi-line text for the question and snake length
                    text_color = (255, 255, 255)  # White for normal state

                    # First line: "Will my AI learn to play snake?"
                    line1 = ""
                    text_size1 = cv2.getTextSize(line1, font, font_scale, thickness)[0]
                    text_x1 = (frame.shape[1] - text_size1[0]) // 2
                    text_y1 = 120
                    cv2.putText(overlay_frame, line1, (text_x1, text_y1), font, font_scale, text_color, thickness)

                    # Second line: "Snake Length: X"
                    line2 = f"Snake Length: {current_length}"
                    text_size2 = cv2.getTextSize(line2, font, font_scale, thickness)[0]
                    text_x2 = (frame.shape[1] - text_size2[0]) // 2
                    text_y2 = text_y1 + text_size1[1] + 20  # Add some spacing between lines
                    cv2.putText(overlay_frame, line2, (text_x2, text_y2), font, font_scale, text_color, thickness)
            """

            return overlay_frame
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame

    def capture_frame(self, surface, episode=0, score=0, total_food=0, current_length=3, skill_level=0.5, elapsed_time=0):
        if not self.recording:
            return

        # Convert pygame surface to numpy array
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))

        game_height, game_width = frame.shape[:2]
        scale = self.target_width / game_width
        new_height = int(game_height * scale)

        import cv2
        # Use high-quality interpolation for better upscaling
        resized_frame = cv2.resize(frame, (self.target_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        # Add overlay
        resized_frame = self.create_overlay(resized_frame, episode, score, total_food, current_length, skill_level, elapsed_time)

        # Create final frame with padding
        final_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        y_offset = (self.target_height - new_height) // 2
        final_frame[y_offset:y_offset + new_height] = resized_frame

        # Convert RGB to BGR for video writer
        final_frame_bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
        self.frames.append(final_frame_bgr)

    def stop_recording(self):
        if not self.recording or not self.frames:
            return

        self.recording = False
        print(f"💾 Saving {len(self.frames)} frames with high quality...")

        try:
            import cv2

            # Use high-quality video codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                self.filename,
                fourcc,
                self.fps,
                (self.target_width, self.target_height),
                True
            )

            # Write frames with high quality
            for frame in self.frames:
                video_writer.write(frame)

            video_writer.release()
            print(f"✅ High-quality video saved: {self.filename}")

            # Try to optimize with ffmpeg if available
            self._optimize_video_with_ffmpeg()

            # Create version with audio if we have audio events
            if self.audio_events:
                self._create_audio_enhanced_version()

        except Exception as e:
            print(f"❌ Video save error: {e}")
            # Fallback to imageio if cv2 fails
            try:
                # Convert BGR back to RGB for imageio
                rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in self.frames]
                imageio.mimsave(self.filename, rgb_frames, fps=self.fps, quality=10)
                print(f"✅ MASTER video saved (fallback): {self.filename}")
            except Exception as e2:
                print(f"❌ Fallback save error: {e2}")

    def _optimize_video_with_ffmpeg(self):
        """Optimize video with ffmpeg for better quality and smaller size"""
        try:
            import subprocess
            import os

            # Create optimized filename
            base_name = os.path.splitext(self.filename)[0]
            optimized_name = f"{base_name}_optimized.mp4"

            print("🔧 Optimizing video with FFmpeg...")

            # FFmpeg command for high-quality optimization
            cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-i', self.filename,  # Input file
                '-c:v', 'libx264',  # Video codec
                '-preset', 'medium',  # Encoding preset
                '-crf', '18',  # Quality (lower = better, 18 is very high quality)
                '-pix_fmt', 'yuv420p',  # Pixel format for compatibility
                '-movflags', '+faststart',  # Optimize for streaming
                optimized_name
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                # Replace original with optimized version
                os.replace(optimized_name, self.filename)
                print(f"✅ Video optimized successfully!")
            else:
                print(f"⚠️ FFmpeg optimization failed: {result.stderr}")
                if os.path.exists(optimized_name):
                    os.remove(optimized_name)

        except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError) as e:
            print(f"⚠️ FFmpeg not available or failed: {e}")
        except Exception as e:
            print(f"⚠️ Optimization error: {e}")

    def _create_audio_enhanced_version(self):
        """Create separate audio file with sound effects"""
        try:
            base_name = os.path.splitext(self.filename)[0]
            audio_file = f"{base_name}_audio.mp3"

            print(f"🎵 Creating separate audio file with {len(self.audio_events)} events...")

            # Create a simple audio track with the recorded events
            if self.audio_events:
                self._create_separate_audio_file(audio_file)
            else:
                print("⚠️ No audio events recorded")

        except Exception as e:
            print(f"⚠️ Audio file creation error: {e}")

    def _create_separate_audio_file(self, audio_path: str):
        """Create separate audio file with sound effects at correct timestamps"""
        try:
            # Get video duration from the recorded frames
            duration = len(self.frames) / self.fps

            # Create silent base track
            silent_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
                "-t", str(duration),
                "-c:a", "mp3",
                "-b:a", "128k",
                audio_path
            ]

            # If we have audio events, overlay them
            if self.audio_events:
                # Group events by sound type to avoid too many overlays
                food_events = [(t, "food_eaten") for t, event_type in self.audio_events if event_type == "food_eaten"]
                episode_events = [(t, "episode_start") for t, event_type in self.audio_events if event_type == "episode_start"]

                # Create audio file with just a few key sound effects (to avoid complexity)
                selected_events = []

                # Add first few episode starts
                selected_events.extend(episode_events[:5])

                # Add some food eating sounds (every 3rd one to not overwhelm)
                selected_events.extend(food_events[::3][:10])

                if selected_events:
                    self._create_audio_with_events(audio_path, selected_events, duration)
                else:
                    # Just create silent track
                    subprocess.run(silent_cmd, capture_output=True, timeout=30)
            else:
                # Create silent track
                result = subprocess.run(silent_cmd, capture_output=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ Silent audio track created: {audio_path}")

        except Exception as e:
            print(f"❌ Audio file creation failed: {e}")

    def _create_audio_with_events(self, audio_path: str, events: list, duration: float):
        """Create audio file with selected sound events using the simplest possible approach"""
        try:
            assets_dir = os.path.join(os.path.dirname(__file__), "assets")
            sound_files = {
                "food_eaten": "tiktok_pop.mp3",
                "episode_start": "tiktok_bling.mp3"
            }

            print(f"🔧 Creating audio with {len(events)} sound events for {duration:.1f}s duration...")

            # Sort events by timestamp
            events.sort(key=lambda x: x[0])

            # Use the simplest approach: create individual audio clips and concatenate
            # This avoids complex FFmpeg filter syntax entirely

            temp_clips = []

            # Create individual audio clips for each time segment
            current_time = 0.0
            clip_index = 0

            for timestamp, sound_type in events:
                # Create silent segment before this sound (if needed)
                if timestamp > current_time:
                    silence_duration = timestamp - current_time
                    if silence_duration > 0.1:  # Only create if significant duration
                        silence_clip = f"/tmp/silence_{clip_index}.wav"
                        silence_cmd = [
                            "ffmpeg", "-y",
                            "-f", "lavfi",
                            "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={silence_duration}",
                            "-c:a", "pcm_s16le",
                            silence_clip
                        ]
                        result = subprocess.run(silence_cmd, capture_output=True, timeout=30)
                        if result.returncode == 0:
                            temp_clips.append(silence_clip)
                        clip_index += 1

                # Add the sound effect
                if sound_type in sound_files:
                    sound_path = os.path.join(assets_dir, sound_files[sound_type])
                    if os.path.exists(sound_path):
                        # Convert sound to WAV for consistency
                        sound_clip = f"/tmp/sound_{clip_index}.wav"
                        sound_cmd = [
                            "ffmpeg", "-y",
                            "-i", sound_path,
                            "-c:a", "pcm_s16le",
                            "-ar", "44100",
                            sound_clip
                        ]
                        result = subprocess.run(sound_cmd, capture_output=True, timeout=30)
                        if result.returncode == 0:
                            temp_clips.append(sound_clip)
                            # Update current time to account for sound duration
                            probe_cmd = [
                                "ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                                "-of", "csv=p=0", sound_clip
                            ]
                            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                            if probe_result.returncode == 0:
                                sound_duration = float(probe_result.stdout.strip())
                                current_time = timestamp + sound_duration
                            else:
                                current_time = timestamp + 1.0  # Assume 1 second
                        clip_index += 1

            # Add final silence to reach full duration
            if current_time < duration:
                final_silence = duration - current_time
                if final_silence > 0.1:
                    final_clip = f"/tmp/final_silence.wav"
                    silence_cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={final_silence}",
                        "-c:a", "pcm_s16le",
                        final_clip
                    ]
                    result = subprocess.run(silence_cmd, capture_output=True, timeout=30)
                    if result.returncode == 0:
                        temp_clips.append(final_clip)

            if temp_clips:
                # Create concat file list
                concat_file = "/tmp/concat_list.txt"
                with open(concat_file, "w") as f:
                    for clip in temp_clips:
                        f.write(f"file '{clip}'\n")

                # Use concat demuxer (much simpler than complex filters)
                concat_cmd = [
                    "ffmpeg", "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", concat_file,
                    "-c:a", "mp3",
                    "-b:a", "128k",
                    audio_path
                ]

                print(f"🎵 Concatenating {len(temp_clips)} audio segments...")
                result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=90)

                # Cleanup temp files
                for clip in temp_clips:
                    if os.path.exists(clip):
                        os.remove(clip)
                if os.path.exists(concat_file):
                    os.remove(concat_file)

                if result.returncode == 0:
                    print(f"✅ Audio file created: {audio_path}")
                    return
                else:
                    print(f"⚠️ Concatenation failed: {result.stderr}")

            # If we get here, something failed - create simple silent audio
            print(f"📝 Creating fallback silent audio for {duration:.1f}s...")
            self._create_simple_silent_audio(audio_path, duration)

        except Exception as e:
            print(f"❌ Audio with events creation failed: {e}")
            print(f"📝 Creating fallback silent audio for {duration:.1f}s...")
            self._create_simple_silent_audio(audio_path, duration)

    def _create_simple_silent_audio(self, audio_path: str, duration: float):
        """Create a simple silent audio file as fallback"""
        try:
            simple_cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100:duration={duration}",
                "-c:a", "mp3",
                "-b:a", "128k",
                audio_path
            ]
            print(f"🔧 Creating {duration:.1f}s silent audio track...")
            result = subprocess.run(simple_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                print(f"✅ Created fallback silent audio track: {audio_path} ({duration:.1f}s)")
            else:
                print(f"❌ Even simple audio creation failed: {result.stderr}")
        except Exception as e:
            print(f"❌ Fallback audio creation failed: {e}")


def run_master_snake_demo(target_duration=50):
    """
    Run the master Snake AI demo - shows impressive food eating and growth!
    Targets 40-60 second duration with amazing results.
    """

    # Initialize sound first
    sound_manager = SoundManager()
    sound_manager.set_volume(0.8)  # Set to 80% volume

    env = env_raw_snake()
    env.init()
    ai = MasterSnakeAI()
    recorder = MasterVideoRecorder("snake_master_growing_longer.mp4", fps=60)

    # Link recorder to sound manager for audio event tracking
    sound_manager.set_recorder(recorder)

    print("🐍 MASTER SNAKE AI - Growing Champion!")
    print(f"🎯 Target: {target_duration}s | Watch the snake grow longer!")

    recorder.start_recording()

    episode_scores = []
    episode_food_counts = []
    start_time = time.time()

    try:
        episode = 0
        max_episodes = 48

        while episode < max_episodes:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Stop if approaching target duration
            if elapsed_time > target_duration - 8:
                break

            ai.episode = episode
            env.reset_game()

            # Play episode start sound
            sound_manager.play_episode_start()

            episode_start_time = time.time()
            steps = 0
            food_count_this_episode = 0
            last_score = 0
            max_steps = min(6000, int((target_duration - elapsed_time) * 100))

            skill_level = ai.get_skill_level()

            # Minimal episode start message
            if episode == 0:
                print(f"🎮 Starting Snake AI Training...")

            while not env.game_over() and steps < max_steps:
                # Get state and decide action
                state = env.getGameState()
                action = ai.choose_action(state)

                # Execute action
                reward = env.act(action)
                current_score = env.score()

                # Track food consumption
                if current_score > last_score:
                    food_count_this_episode += 1
                    # Play sound when snake eats food
                    sound_manager.play_food_eaten()
                    # Minimal feedback - only show significant growth
                    if food_count_this_episode == 1:
                        print(f"   🍎 Growing!")

                last_score = current_score

                # Record frame
                screen = pygame.display.get_surface()
                if screen:
                    current_elapsed = time.time() - start_time
                    snake_length = len(state.get('snake_body_pos', []))

                    recorder.capture_frame(
                        screen,
                        episode=episode,
                        score=current_score,
                        total_food=food_count_this_episode,
                        current_length=snake_length,
                        skill_level=skill_level,
                        elapsed_time=current_elapsed
                    )

                steps += 1

                # Variable speed - slower when doing well to show growth
                if food_count_this_episode > 0:
                    time.sleep(0.015)  # Slower to show off growth
                else:
                    time.sleep(0.01)   # Faster when just learning

            episode_duration = time.time() - episode_start_time
            final_score = env.score()
            episode_scores.append(final_score)
            episode_food_counts.append(food_count_this_episode)
            total_elapsed = time.time() - start_time

            # Minimal episode summary
            print(f"✅ Episode {episode + 1}: {food_count_this_episode} food eaten")

            episode += 1

        total_time = time.time() - start_time
        total_food = sum(episode_food_counts)

        print(f"\n🎉 COMPLETE! Duration: {total_time:.1f}s | Total Food: {total_food} | Best Score: {max(episode_scores) if episode_scores else 0}")

        if total_food >= 5:
            print(f"🎯 INCREDIBLE! Snake mastered the game! 🐍📈")
        elif total_food >= 2:
            print(f"✅ SUCCESS! Snake learned to grow! 🐍✨")

    except KeyboardInterrupt:
        print("\\n⏹️ Demo interrupted")

    finally:
        recorder.stop_recording()
        sound_manager.cleanup()  # Clean up audio resources
        pygame.quit()

        total_time = time.time() - start_time
        print(f"\n🎬 Video: snake_master_growing_longer.mp4 | {total_time:.1f}s | 1080x1920@60fps")


if __name__ == "__main__":
    # Run the master demo targeting 45-55 seconds
    run_master_snake_demo(target_duration=6000)