"""
Base Game Environment for YouTube Shorts optimized classic games.
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class BaseGameEnv(gym.Env, ABC):
    """
    Base class for all classic game environments.
    Provides common functionality for YouTube Shorts optimization.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 game_name: str,
                 window_size: Tuple[int, int] = (405, 720),  # 9:16 aspect ratio
                 max_episode_steps: int = 1000,
                 render_mode: Optional[str] = None):
        """
        Initialize base game environment.

        Args:
            game_name: Name of the game
            window_size: Window dimensions (width, height) for 9:16 aspect ratio
            max_episode_steps: Maximum steps per episode
            render_mode: Render mode ('human', 'rgb_array', or None)
        """
        super().__init__()

        self.game_name = game_name
        self.window_size = window_size
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # Game state
        self.current_step = 0
        self.episode_score = 0
        self.episode_reward = 0
        self.game_over = False

        # Rendering
        self.window = None
        self.clock = None
        self.font_cache = {}

        # Colors for consistent theming
        self.colors = {
            'background': (15, 15, 25),           # Dark blue-black
            'primary': (100, 149, 237),           # Cornflower blue
            'secondary': (255, 215, 0),           # Gold
            'success': (50, 205, 50),             # Lime green
            'danger': (220, 20, 60),              # Crimson
            'warning': (255, 140, 0),             # Dark orange
            'text_primary': (255, 255, 255),      # White
            'text_secondary': (180, 180, 190),    # Light gray
            'ui_bg': (30, 30, 40),                # Dark gray
            'ui_border': (80, 80, 90),            # Light gray
        }

        # Performance tracking
        self.performance_history = {
            'scores': [],
            'rewards': [],
            'steps': [],
            'actions': []
        }

        # Initialize pygame if needed
        self._initialize_pygame()

    def _initialize_pygame(self):
        """Initialize pygame if not already done."""
        if not pygame.get_init():
            pygame.init()
            pygame.display.init()

    def get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Get cached font of specified size."""
        key = (size, bold)
        if key not in self.font_cache:
            self.font_cache[key] = pygame.font.Font(None, size)
            if bold:
                self.font_cache[key].set_bold(True)
        return self.font_cache[key]

    @abstractmethod
    def _get_observation(self) -> np.ndarray:
        """Get current game observation."""
        pass

    @abstractmethod
    def _take_action(self, action: int) -> Tuple[float, bool, bool]:
        """
        Take game action and return (reward, terminated, truncated).
        """
        pass

    @abstractmethod
    def _reset_game(self):
        """Reset game to initial state."""
        pass

    @abstractmethod
    def _render_game(self, surface: pygame.Surface):
        """Render game-specific content to surface."""
        pass

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Reset game state
        self.current_step = 0
        self.episode_score = 0
        self.episode_reward = 0
        self.game_over = False

        # Reset game-specific state
        self._reset_game()

        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        if self.game_over:
            # Game is over, return current state
            return self._get_observation(), 0.0, True, False, self._get_info()

        self.current_step += 1

        # Take action and get reward
        reward, terminated, truncated = self._take_action(action)

        # Update tracking
        self.episode_reward += reward
        self.performance_history['actions'].append(action)

        # Check for episode end
        if self.current_step >= self.max_episode_steps:
            truncated = True

        if terminated:
            self.game_over = True
            self.performance_history['scores'].append(self.episode_score)
            self.performance_history['rewards'].append(self.episode_reward)
            self.performance_history['steps'].append(self.current_step)

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.window is None:
            if self.render_mode == "human":
                pygame.display.init()
                self.window = pygame.display.set_mode(self.window_size)
                pygame.display.set_caption(f"{self.game_name} - RL Environment")
            else:
                self.window = pygame.Surface(self.window_size)

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Clear screen
        self.window.fill(self.colors['background'])

        # Render game content
        self._render_game(self.window)

        # Render UI overlay
        self._render_ui_overlay()

        if self.render_mode == "human":
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        else:  # rgb_array
            # Convert surface to RGB array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)),
                axes=(1, 0, 2)
            )

    def _render_ui_overlay(self):
        """Render UI overlay with score, progress, etc."""
        # Header section
        self._render_header()

        # Progress section
        self._render_progress()

        # Stats section
        self._render_stats()

    def _render_header(self):
        """Render header with game name and score."""
        font_title = self.get_font(32, bold=True)
        font_score = self.get_font(24)

        # Game title
        title_text = font_title.render(f"🎮 {self.game_name.upper()} 🎮", True, self.colors['text_primary'])
        title_rect = title_text.get_rect(center=(self.window_size[0] // 2, 30))
        self.window.blit(title_text, title_rect)

        # Score
        score_text = font_score.render(f"Score: {self.episode_score}", True, self.colors['secondary'])
        score_rect = score_text.get_rect(center=(self.window_size[0] // 2, 60))
        self.window.blit(score_text, score_rect)

        # Separator line
        line_y = 80
        pygame.draw.line(
            self.window,
            self.colors['primary'],
            (50, line_y),
            (self.window_size[0] - 50, line_y),
            2
        )

    def _render_progress(self):
        """Render progress bar and episode info."""
        progress_y = self.window_size[1] - 120

        # Episode progress
        progress = min(self.current_step / self.max_episode_steps, 1.0)

        # Progress bar
        bar_width = self.window_size[0] - 60
        bar_height = 20
        bar_x = 30

        # Background
        bg_rect = pygame.Rect(bar_x - 2, progress_y - 2, bar_width + 4, bar_height + 4)
        pygame.draw.rect(self.window, self.colors['text_primary'], bg_rect, border_radius=12)

        progress_bg = pygame.Rect(bar_x, progress_y, bar_width, bar_height)
        pygame.draw.rect(self.window, self.colors['ui_bg'], progress_bg, border_radius=10)

        # Fill
        if progress > 0:
            fill_width = int(bar_width * progress)
            fill_rect = pygame.Rect(bar_x, progress_y, fill_width, bar_height)

            # Color based on progress
            if progress < 0.3:
                color = self.colors['success']
            elif progress < 0.7:
                color = self.colors['warning']
            else:
                color = self.colors['danger']

            pygame.draw.rect(self.window, color, fill_rect, border_radius=10)

        # Progress text
        font = self.get_font(16)
        progress_text = font.render(f"Episode Progress: {progress*100:.1f}%", True, self.colors['text_secondary'])
        text_rect = progress_text.get_rect(center=(self.window_size[0] // 2, progress_y + 35))
        self.window.blit(progress_text, text_rect)

    def _render_stats(self):
        """Render performance statistics."""
        stats_y = self.window_size[1] - 70
        font = self.get_font(14)

        stats = [
            f"Step: {self.current_step}/{self.max_episode_steps}",
            f"Reward: {self.episode_reward:.1f}",
        ]

        for i, stat in enumerate(stats):
            x_offset = 30 + (i * (self.window_size[0] // 2 - 30))
            stat_text = font.render(stat, True, self.colors['text_secondary'])
            self.window.blit(stat_text, (x_offset, stats_y))

    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary."""
        return {
            'score': self.episode_score,
            'reward': self.episode_reward,
            'step': self.current_step,
            'max_steps': self.max_episode_steps,
            'game_over': self.game_over,
            'episode_progress': min(self.current_step / self.max_episode_steps, 1.0),
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics across episodes."""
        if not self.performance_history['scores']:
            return {'episodes': 0}

        scores = np.array(self.performance_history['scores'])
        rewards = np.array(self.performance_history['rewards'])
        steps = np.array(self.performance_history['steps'])

        return {
            'episodes': len(scores),
            'avg_score': np.mean(scores),
            'best_score': np.max(scores),
            'avg_reward': np.mean(rewards),
            'avg_steps': np.mean(steps),
            'score_history': scores.tolist()[-10:],  # Last 10 episodes
        }

    def close(self):
        """Close the environment."""
        if self.window is not None:
            pygame.display.quit()
            self.window = None
        if pygame.get_init():
            pygame.quit()
        self.clock = None