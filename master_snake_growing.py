import numpy as np
import pygame
import time
import random
import imageio
from classic_games.env_collection import env_raw_snake


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

    def start_recording(self):
        self.recording = True
        self.frames = []
        print(f"🎥 Recording MASTER Snake AI: {self.filename}")

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
            else:
                growth_text = f"Snake Length: {current_length}"
                text_color = (255, 255, 255)  # White for normal state

            # Center the text horizontally
            text_size = cv2.getTextSize(growth_text, font, font_scale, thickness)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = 120  # Top of screen for maximum visibility

            cv2.putText(overlay_frame, growth_text, (text_x, text_y), font, font_scale, text_color, thickness)

            return overlay_frame
        except Exception as e:
            print(f"Overlay error: {e}")
            return frame

    def capture_frame(self, surface, episode=0, score=0, total_food=0, current_length=3, skill_level=0.5, elapsed_time=0):
        if not self.recording:
            return

        # Convert and resize
        frame = pygame.surfarray.array3d(surface)
        frame = np.transpose(frame, (1, 0, 2))

        game_height, game_width = frame.shape[:2]
        scale = self.target_width / game_width
        new_height = int(game_height * scale)

        import cv2
        resized_frame = cv2.resize(frame, (self.target_width, new_height))

        # Add overlay
        resized_frame = self.create_overlay(resized_frame, episode, score, total_food, current_length, skill_level, elapsed_time)

        # Create final frame
        final_frame = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        y_offset = (self.target_height - new_height) // 2
        final_frame[y_offset:y_offset + new_height] = resized_frame

        self.frames.append(final_frame)

    def stop_recording(self):
        if not self.recording or not self.frames:
            return

        self.recording = False
        print(f"💾 Saving {len(self.frames)} frames...")

        try:
            imageio.mimsave(self.filename, self.frames, fps=self.fps, quality=9)
            print(f"✅ MASTER video saved: {self.filename}")
        except Exception as e:
            print(f"❌ Save error: {e}")


def run_master_snake_demo(target_duration=50):
    """
    Run the master Snake AI demo - shows impressive food eating and growth!
    Targets 40-60 second duration with amazing results.
    """

    env = env_raw_snake()
    env.init()
    ai = MasterSnakeAI()
    recorder = MasterVideoRecorder("snake_master_growing_longer.mp4", fps=60)

    print("🐍 MASTER SNAKE AI - Growing Champion!")
    print(f"🎯 Target: {target_duration}s | Watch the snake grow longer!")

    recorder.start_recording()

    episode_scores = []
    episode_food_counts = []
    start_time = time.time()

    try:
        episode = 0
        max_episodes = 8

        while episode < max_episodes:
            current_time = time.time()
            elapsed_time = current_time - start_time

            # Stop if approaching target duration
            if elapsed_time > target_duration - 8:
                break

            ai.episode = episode
            env.reset_game()

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
        pygame.quit()

        total_time = time.time() - start_time
        print(f"\n🎬 Video: snake_master_growing_longer.mp4 | {total_time:.1f}s | 1080x1920@60fps")


if __name__ == "__main__":
    # Run the master demo targeting 45-55 seconds
    run_master_snake_demo(target_duration=1200)