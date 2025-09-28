import numpy as np
import pygame
import time
import random
from classic_games.env_collection import env_raw_flappybird


class QuickFlappyBirdAI:
    """A simplified but effective Flappy Bird AI for quick testing"""

    def __init__(self):
        self.episode = 0
        self.jump_cooldown = 0

    def choose_action(self, game_state):
        """Simple but effective strategy for Flappy Bird"""
        # Handle jump cooldown
        if self.jump_cooldown > 0:
            self.jump_cooldown -= 1
            return None

        bird_y = game_state['player_y']
        bird_vel = game_state['player_vel']

        # Get pipe information
        next_pipe_dist_x = game_state.get('next_pipe_dist_x', float('inf'))
        next_pipe_top_y = game_state.get('next_pipe_top_y', 0)
        next_pipe_bottom_y = game_state.get('next_pipe_bottom_y', 512)

        # Calculate gap center
        gap_center = (next_pipe_top_y + next_pipe_bottom_y) / 2

        # Simple strategy:
        # 1. If bird is below gap center and falling, jump
        # 2. If bird is approaching bottom of screen, jump
        # 3. If bird is falling fast, jump

        should_jump = False

        if bird_y > gap_center + 20:  # Below desired position
            should_jump = True
        elif bird_y > 400:  # Too low on screen
            should_jump = True
        elif bird_vel > 5:  # Falling too fast
            should_jump = True

        if should_jump:
            self.jump_cooldown = 8 + random.randint(0, 4)  # Prevent rapid jumping
            return 119  # Jump action

        return None  # Don't jump


def run_quick_flappy_demo():
    """Quick demo to test Flappy Bird AI performance"""

    env = env_raw_flappybird()
    env.init()
    ai = QuickFlappyBirdAI()

    print("🐦 Quick Flappy Bird AI Test")
    print("=" * 40)

    episode_scores = []

    for episode in range(10):
        ai.episode = episode
        env.reset_game()

        steps = 0
        max_steps = 1000

        while not env.game_over() and steps < max_steps:
            state = env.getGameState()
            action = ai.choose_action(state)

            if action is not None:
                reward = env.act(action)
            else:
                reward = env.act(None)

            steps += 1

        final_score = env.score()
        episode_scores.append(final_score)

        print(f"Episode {episode + 1:2d}: Score {final_score:6.1f} | Steps {steps:4d}")

    pygame.quit()

    print("=" * 40)
    print(f"Average Score: {np.mean(episode_scores):6.1f}")
    print(f"Best Score:    {max(episode_scores):6.1f}")
    print(f"Episodes > 0:  {sum(1 for s in episode_scores if s > 0)}/10")

    if max(episode_scores) > 0:
        print("✅ SUCCESS! AI can navigate pipes!")
    else:
        print("📚 AI is learning - needs more training")


if __name__ == "__main__":
    run_quick_flappy_demo()