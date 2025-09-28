import numpy as np
import pygame
import pygame.mixer
import time
import random
import imageio
import os
import subprocess
from classic_games.env_collection import env_raw_flappybird


class SoundManager:
    """Manages sound effects for the flappy bird game"""

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
            "score": "tiktok_pop.mp3",      # Sound when bird scores
            "jump": "tiktok_bling.mp3",     # Sound when bird jumps (occasionally)
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
            return

        if sound_name not in self.sounds:
            return

        try:
            # Stop any currently playing instances of this sound
            self.sounds[sound_name].stop()
            # Play the sound
            self.sounds[sound_name].play()

            # Track the audio event for video post-processing
            if self.recorder:
                self.recorder.add_audio_event(sound_name)

        except pygame.error as e:
            print(f"⚠️ Failed to play {sound_name}: {e}")

    def play_score(self):
        """Play sound when bird scores"""
        self.play_sound("score")

    def play_jump(self):
        """Play sound when bird jumps (occasionally)"""
        # Only play jump sound occasionally to avoid noise
        if random.random() < 0.1:  # 10% chance
            self.play_sound("jump")

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


class MasterFlappyBirdAI:
    def __init__(self):
        """Master Flappy Bird AI that gets progressively better at navigating pipes"""
        self.episode = 0
        self.jump_cooldown = 0
        self.last_pipe_distance = float('inf')
        self.consecutive_scores = 0

    def get_skill_level(self):
        """Return current skill level (0-1)"""
        # Rapid improvement over episodes
        return min(0.95, 0.2 + (self.episode * 0.08))

    def get_noise_level(self):
        """Get current randomness level"""
        skill = self.get_skill_level()
        return max(0.02, 0.4 * (1 - skill))

    def analyze_pipe_situation(self, game_state):
        """Analyze the current pipe situation and determine optimal action"""
        bird_y = game_state['player_y']
        bird_vel = game_state['player_vel']

        # Get pipe information
        next_pipe_dist_x = game_state.get('next_pipe_dist_x', float('inf'))
        next_pipe_top_y = game_state.get('next_pipe_top_y', 0)
        next_pipe_bottom_y = game_state.get('next_pipe_bottom_y', 512)

        # Calculate pipe gap center and size
        gap_center = (next_pipe_top_y + next_pipe_bottom_y) / 2
        gap_size = next_pipe_bottom_y - next_pipe_top_y

        return {
            'bird_y': bird_y,
            'bird_vel': bird_vel,
            'pipe_distance': next_pipe_dist_x,
            'gap_center': gap_center,
            'gap_size': gap_size,
            'pipe_top': next_pipe_top_y,
            'pipe_bottom': next_pipe_bottom_y
        }

    def predict_bird_trajectory(self, bird_y, bird_vel, steps_ahead=10):
        """Predict where bird will be after a certain number of steps"""
        # Flappy Bird physics simulation
        gravity = 0.5  # Approximate gravity effect
        predicted_y = bird_y
        predicted_vel = bird_vel

        for _ in range(steps_ahead):
            predicted_vel += gravity
            predicted_y += predicted_vel

        return predicted_y, predicted_vel

    def should_jump(self, pipe_info):
        """Determine if bird should jump based on pipe analysis"""
        bird_y = pipe_info['bird_y']
        bird_vel = pipe_info['bird_vel']
        pipe_distance = pipe_info['pipe_distance']
        gap_center = pipe_info['gap_center']
        gap_size = pipe_info['gap_size']

        # Enhanced strategy for PLE Flappy Bird
        # The goal is to stay alive and pass through pipes

        # If no pipes nearby, maintain middle position
        if pipe_distance > 200:
            target_y = 256  # Middle of screen (512/2)
            # Jump if too low or falling fast
            return bird_y > target_y + 40 or bird_vel > 8

        # For close pipes, be more precise
        skill_level = self.get_skill_level()

        # Predict where bird will be when it reaches the pipe
        time_to_pipe = max(1, pipe_distance / 4)  # Approximate time to reach pipe
        future_y = bird_y + (bird_vel * time_to_pipe) + (0.25 * time_to_pipe * time_to_pipe)

        # Calculate safety margins based on skill
        margin = max(15, (1 - skill_level) * 35)

        # Jump conditions:
        # 1. Bird is falling too fast
        if bird_vel > 6:
            return True

        # 2. Future position would be too low
        if future_y > gap_center + margin:
            return True

        # 3. Current position is already too low
        if bird_y > gap_center + margin:
            return True

        # 4. Bird is approaching ground
        if bird_y > 450:
            return True

        # 5. Bird is falling and below gap center
        if bird_vel > 3 and bird_y > gap_center:
            return True

        # Don't jump if well positioned
        return False

    def choose_action(self, game_state):
        """Choose the best action to navigate through pipes"""
        # Handle jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            return None  # Don't jump during cooldown

        # Analyze current situation
        pipe_info = self.analyze_pipe_situation(game_state)

        # Add some randomness for learning effect
        noise_level = self.get_noise_level()
        if random.random() < noise_level:
            if random.random() < 0.3:  # 30% chance to jump randomly
                self.jump_cooldown = random.randint(3, 8)  # Random cooldown
                return 119  # Jump
            return None  # Don't jump

        # Smart decision making
        if self.should_jump(pipe_info):
            # Set cooldown to prevent rapid jumping
            skill_level = self.get_skill_level()
            base_cooldown = max(4, int(12 * (1 - skill_level)))
            self.jump_cooldown = random.randint(base_cooldown - 2, base_cooldown + 2)
            return 119  # Jump (space bar equivalent)

        return None  # Don't jump


class MasterVideoRecorder:
    def __init__(self, filename="flappy_bird_master.mp4", fps=60):
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
        print(f"🎥 Recording MASTER Flappy Bird AI: {self.filename}")

    def add_audio_event(self, event_type: str):
        """Track audio events with timestamps for post-processing"""
        if self.recording and self.start_time:
            timestamp = time.time() - self.start_time
            self.audio_events.append((timestamp, event_type))

    def create_overlay(self, frame, episode, score, total_score, skill_level, elapsed_time):
        """Create overlay showing AI progress"""
        try:
            import cv2
            overlay_frame = frame.copy()

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2.8
            thickness = 7

            # Main attention-grabbing text
            if score > 0:
                success_text = f"Flying! Score: {score}"
                text_color = (0, 255, 100)  # Bright green for success
            else:
                success_text = f"Learning to fly..."
                text_color = (255, 255, 255)  # White for normal state

            # Center the text horizontally
            text_size = cv2.getTextSize(success_text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 120  # Top of screen for maximum visibility

            cv2.putText(overlay_frame, success_text, (text_x, text_y), font, font_scale, text_color, thickness)

            # Add skill level indicator
            if skill_level > 0.5:
                skill_text = f"AI Skill: {skill_level*100:.0f}%"
                skill_color = (100, 200, 255)  # Light blue
                font_scale_small = 2.0
                thickness_small = 5

                skill_size = cv2.getTextSize(skill_text, font, font_scale_small, thickness_small)[0]
                skill_x = (frame.shape[1] - skill_size[0]) // 2
                skill_y = text_y + 80

                cv2.putText(overlay_frame, skill_text, (skill_x, skill_y), font, font_scale_small, skill_color, thickness_small)

            return overlay_frame
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame

    def capture_frame(self, surface, episode=0, score=0, total_score=0, skill_level=0.5, elapsed_time=0):
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
        resized_frame = self.create_overlay(resized_frame, episode, score, total_score, skill_level, elapsed_time)

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


def run_master_flappy_bird_demo(target_duration=45):
    """
    Run the master Flappy Bird AI demo - shows impressive flying and scoring!
    Targets 40-50 second duration with amazing results.
    """

    # Initialize sound first
    sound_manager = SoundManager()
    sound_manager.set_volume(0.8)  # Set to 80% volume

    env = env_raw_flappybird()
    env.init()
    ai = MasterFlappyBirdAI()
    recorder = MasterVideoRecorder("flappy_bird_master_flying.mp4", fps=60)

    # Link recorder to sound manager for audio event tracking
    sound_manager.set_recorder(recorder)

    print("🐦 MASTER FLAPPY BIRD AI - Flying Champion!")
    print(f"🎯 Target: {target_duration}s | Watch the bird master flight!")

    recorder.start_recording()

    episode_scores = []
    total_score = 0
    start_time = time.time()

    try:
        episode = 0
        max_episodes = 30

        while episode < max_episodes:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Stop if approaching target duration
            if elapsed_time > target_duration - 5:
                break

            ai.episode = episode
            env.reset_game()

            # Play episode start sound
            sound_manager.play_episode_start()

            episode_start_time = time.time()
            steps = 0
            last_score = 0
            max_steps = min(3000, int((target_duration - elapsed_time) * 80))

            skill_level = ai.get_skill_level()

            # Minimal episode start message
            if episode == 0:
                print(f"🎮 Starting Flappy Bird AI Training...")

            while not env.game_over() and steps < max_steps:
                # Get state and decide action
                state = env.getGameState()
                action = ai.choose_action(state)

                # Execute action
                if action is not None:
                    reward = env.act(action)
                    sound_manager.play_jump()  # Occasional jump sound
                else:
                    reward = env.act(None)  # No action (let bird fall)

                current_score = env.score()

                # Track scoring
                if current_score > last_score:
                    # Play sound when bird scores
                    sound_manager.play_score()
                    print(f"   🎯 Score! ({current_score})")

                last_score = current_score

                # Record frame
                screen = pygame.display.get_surface()
                if screen:
                    current_elapsed = time.time() - start_time

                    recorder.capture_frame(
                        screen,
                        episode=episode,
                        score=current_score,
                        total_score=total_score,
                        skill_level=skill_level,
                        elapsed_time=current_elapsed
                    )

                steps += 1
                time.sleep(0.016)  # ~60 FPS

            episode_duration = time.time() - episode_start_time
            final_score = env.score()
            episode_scores.append(final_score)
            total_score += final_score
            total_elapsed = time.time() - start_time

            # Minimal episode summary
            if final_score > 0:
                print(f"✅ Episode {episode + 1}: Score {final_score}!")
            else:
                print(f"📝 Episode {episode + 1}: Learning...")

            episode += 1

        total_time = time.time() - start_time
        best_score = max(episode_scores) if episode_scores else 0

        print(f"\n🎉 COMPLETE! Duration: {total_time:.1f}s | Total Score: {total_score} | Best: {best_score}")

        if best_score >= 5:
            print(f"🎯 INCREDIBLE! Bird mastered the pipes! 🐦🚀")
        elif best_score >= 2:
            print(f"✅ SUCCESS! Bird learned to fly! 🐦✨")
        else:
            print(f"📚 LEARNING! Bird is getting better! 🐦📈")

    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")

    finally:
        recorder.stop_recording()
        sound_manager.cleanup()  # Clean up audio resources
        pygame.quit()

        total_time = time.time() - start_time
        print(f"\n🎬 Video: flappy_bird_master_flying.mp4 | {total_time:.1f}s | 1080x1920@60fps")


if __name__ == "__main__":
    # Run the master demo targeting 40-50 seconds
    run_master_flappy_bird_demo(target_duration=45)
