import numpy as np
import pygame
import time
import random
from classic_games.env_collection import env_raw_flappybird


class OptimizedFlappyBirdAI:
    """Optimized Flappy Bird AI with better understanding of game mechanics"""

    def __init__(self):
        self.episode = 0
        self.jump_cooldown = 0
        self.last_score = 0
        self.score_improved = False

    def choose_action(self, game_state):
        """Optimized strategy for PLE Flappy Bird"""

        # Handle jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            return None

        bird_y = game_state['player_y']
        bird_vel = game_state['player_vel']

        # Get pipe information - handle missing keys gracefully
        next_pipe_dist_x = game_state.get('next_pipe_dist_x', 999)
        next_pipe_top_y = game_state.get('next_pipe_top_y', 0)
        next_pipe_bottom_y = game_state.get('next_pipe_bottom_y', 512)

        # Calculate gap center and determine target
        gap_center = (next_pipe_top_y + next_pipe_bottom_y) / 2
        gap_size = next_pipe_bottom_y - next_pipe_top_y

        # Strategy based on distance to pipe
        should_jump = False

        if next_pipe_dist_x > 100:
            # Pipe is far - maintain safe altitude (slight above center)
            target_y = 240
            if bird_y > target_y or (bird_y > target_y - 30 and bird_vel > 4):
                should_jump = True
        else:
            # Pipe is close - be more precise
            # Aim for slightly above gap center
            target_offset = -10  # Slightly above center
            target_y = gap_center + target_offset

            # More aggressive jumping conditions when pipe is close
            if bird_y > target_y:
                should_jump = True
            elif bird_vel > 3 and bird_y > target_y - 20:
                should_jump = True
            elif bird_y > 400:  # Emergency - too low
                should_jump = True

        # Additional safety checks
        if bird_vel > 7:  # Falling very fast
            should_jump = True
        elif bird_y > 450:  # Very close to ground
            should_jump = True

        if should_jump:
            # Adaptive cooldown based on situation
            if next_pipe_dist_x < 50:
                self.jump_cooldown = 6  # Shorter cooldown near pipes
            else:
                self.jump_cooldown = 10  # Longer cooldown far from pipes
            return 119  # Jump action

        return None  # Don't jump


def run_optimized_flappy_demo():
    """Test the optimized Flappy Bird AI"""

    env = env_raw_flappybird()
    env.init()
    ai = OptimizedFlappyBirdAI()

    print("🐦 Optimized Flappy Bird AI Test")
    print("=" * 50)

    episode_scores = []
    episode_steps = []

    for episode in range(15):
        ai.episode = episode
        env.reset_game()

        steps = 0
        max_steps = 2000  # Longer episodes
        last_score = 0

        while not env.game_over() and steps < max_steps:
            state = env.getGameState()
            action = ai.choose_action(state)

            if action is not None:
                reward = env.act(action)
            else:
                reward = env.act(None)

            current_score = env.score()

            # Track if we're improving
            if current_score > last_score:
                print(f"   Episode {episode + 1}: Score improved to {current_score}!")
                last_score = current_score

            steps += 1

        final_score = env.score()
        episode_scores.append(final_score)
        episode_steps.append(steps)

        print(f"Episode {episode + 1:2d}: Score {final_score:6.1f} | Steps {steps:4d}")

    pygame.quit()

    print("=" * 50)
    print(f"Average Score: {np.mean(episode_scores):6.1f}")
    print(f"Best Score:    {max(episode_scores):6.1f}")
    print(f"Average Steps: {np.mean(episode_steps):6.1f}")
    print(f"Max Steps:     {max(episode_steps):6.0f}")
    print(f"Episodes with positive scores: {sum(1 for s in episode_scores if s > 0)}/15")

    # Success criteria
    if max(episode_scores) > 5:
        print("🎉 EXCELLENT! AI is mastering Flappy Bird!")
    elif max(episode_scores) > 0:
        print("✅ SUCCESS! AI can navigate pipes!")
    elif max(episode_steps) > 200:
        print("📈 PROGRESS! AI is surviving longer!")
    else:
        print("📚 AI is learning - needs more refinement")


if __name__ == "__main__":
    run_optimized_flappy_demo()