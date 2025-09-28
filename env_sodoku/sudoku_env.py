"""
Gymnasium-compatible Sudoku RL Environment.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import pygame
import sys
import os

# Add the current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sudoku_game import SudokuGame
from visualizer import SudokuVisualizer


class SudokuEnv(gym.Env):
    """
    Gymnasium environment for Sudoku solving.

    Action Space: Discrete(810) - flat action space representing (row, col, value)
                  Actions 0-809 map to combinations of:
                  - row: 0-8 (9 possibilities)
                  - col: 0-8 (9 possibilities)
                  - value: 0-9 (10 possibilities, where 0 means clear cell, 1-9 means place number 1-9)

    Observation Space: Box(0, 9, (9, 9)) - 9x9 grid with values 0-9

    Rewards:
    - +10 for correctly filling a cell
    - +100 for completing the puzzle
    - -1 for invalid moves
    - -0.1 for each step (time penalty)
    - -5 for creating conflicts
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 difficulty: str = "medium",
                 render_mode: Optional[str] = None,
                 max_steps: int = 1000):
        super().__init__()

        self.difficulty = difficulty
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Initialize game
        self.game = SudokuGame()

        # Action space: flat encoding of (row, col, value)
        # 9 rows × 9 cols × 10 values (0-9) = 810 actions
        # Value 0 means clear cell, values 1-9 mean place number 1-9
        self.action_space = spaces.Discrete(810)

        # Observation space: 9×9 grid with values 0-9
        self.observation_space = spaces.Box(
            low=0, high=9, shape=(9, 9), dtype=np.int32
        )

        # Initialize rendering with enhanced visualizer
        self.visualizer = None
        self.window = None
        self.clock = None
        self.window_size = (405, 720)  # 9:16 aspect ratio for YouTube Shorts

        # Episode tracking
        self.step_count = 0
        self.total_reward = 0
        self.moves_made = 0
        self.invalid_moves = 0

        # Create initial puzzle
        self.reset()

    def action_to_move(self, action: int) -> Tuple[int, int, int]:
        """Convert flat action to (row, col, value)."""
        # Action encoding: action = row * 90 + col * 10 + value
        # where value 0 means "clear cell", 1-9 means place numbers 1-9
        value = action % 10  # 0-9
        action //= 10
        col = action % 9     # 0-8
        row = action // 9    # 0-8

        return row, col, value

    def move_to_action(self, row: int, col: int, value: int) -> int:
        """Convert (row, col, value) to flat action."""
        return row * 90 + col * 10 + value

    def _get_obs(self) -> np.ndarray:
        """Get current observation."""
        return self.game.grid.copy().astype(np.int32)

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info."""
        return {
            "progress": self.game.get_progress(),
            "conflicts": self.game.get_conflicts(),
            "moves_made": self.moves_made,
            "invalid_moves": self.invalid_moves,
            "is_complete": self.game.is_complete(),
            "step_count": self.step_count,
            "total_reward": self.total_reward,
            "empty_cells": len(self.game.get_empty_cells())
        }

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Create new puzzle
        puzzle, solution = self.game.create_puzzle(self.difficulty)

        # Reset tracking variables
        self.step_count = 0
        self.total_reward = 0
        self.moves_made = 0
        self.invalid_moves = 0

        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.step_count += 1
        reward = 0
        terminated = False
        truncated = False

        # Time penalty
        reward -= 0.1

        # Decode action
        row, col, value = self.action_to_move(action)

        # Track move for visualization
        move_made = False
        is_valid = False
        is_correct = False

        # Check if action is valid
        if row >= 9 or col >= 9 or value > 9:
            reward -= 1
            self.invalid_moves += 1
        elif self.game.initial_grid[row, col] != 0:
            # Trying to modify an initial clue
            reward -= 1
            self.invalid_moves += 1
        else:
            # Record current state for conflict checking
            conflicts_before = self.game.get_conflicts()

            if value == 0:
                # Clear cell (value 0 means remove number)
                if self.game.remove_number(row, col):
                    reward += 1  # Small reward for valid clear
                    self.moves_made += 1
                    move_made = True
                    is_valid = True
                else:
                    reward -= 1  # Can't clear initial clue
                    self.invalid_moves += 1
            else:
                # Place number (value 1-9 represents numbers 1-9)
                actual_value = value  # value 1-9 maps directly to numbers 1-9

                # Skip if trying to place 0 (which should be handled above)
                if actual_value == 0:
                    reward -= 1
                    self.invalid_moves += 1
                elif self.game.make_move(row, col, actual_value):
                    # Valid move
                    self.moves_made += 1
                    move_made = True
                    is_valid = True

                    # Check if this creates conflicts
                    conflicts_after = self.game.get_conflicts()

                    if conflicts_after > conflicts_before:
                        reward -= 5  # Penalty for creating conflicts
                    elif self.game.solution[row, col] == actual_value:
                        reward += 10  # Correct number
                        is_correct = True
                    else:
                        reward += 5   # Valid but incorrect number

                    # Check if puzzle is complete
                    if self.game.is_complete():
                        reward += 100
                        terminated = True
                        if self.visualizer:
                            self.visualizer.notify_completion()
                else:
                    # Invalid move (would create immediate conflict)
                    reward -= 1
                    self.invalid_moves += 1

        # Notify visualizer of move
        if self.visualizer and move_made:
            self.visualizer.notify_move(row, col, is_valid, is_correct)

        # Check for truncation (max steps reached)
        if self.step_count >= self.max_steps:
            truncated = True

        self.total_reward += reward

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render the environment using enhanced visualizer."""
        if self.render_mode is None:
            return

        if self.visualizer is None:
            self.visualizer = SudokuVisualizer(self.window_size)
            self.visualizer.initialize()

        # Prepare game state for visualizer
        game_state = {
            'grid': self.game.grid,
            'initial_grid': self.game.initial_grid,
            'solution': self.game.solution
        }

        # Render using enhanced visualizer
        rgb_array = self.visualizer.render(game_state, self._get_info())

        if self.render_mode == "rgb_array":
            return rgb_array

        return rgb_array


# Register the environment

    def _draw_sudoku_grid(self):
        """Draw the Sudoku grid."""
        # Grid parameters optimized for 9:16 aspect ratio
        grid_size = 360
        grid_offset_x = (self.window_size[0] - grid_size) // 2
        grid_offset_y = 100  # Leave space for progress info
        cell_size = grid_size // 9

        # Colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        GRAY = (128, 128, 128)
        LIGHT_GRAY = (200, 200, 200)
        BLUE = (100, 149, 237)
        RED = (255, 99, 71)
        GREEN = (144, 238, 144)

        # Draw grid background
        grid_rect = pygame.Rect(grid_offset_x, grid_offset_y, grid_size, grid_size)
        pygame.draw.rect(self.window, WHITE, grid_rect)

        # Draw cells
        font = pygame.font.Font(None, 36)

        for row in range(9):
            for col in range(9):
                x = grid_offset_x + col * cell_size
                y = grid_offset_y + row * cell_size

                cell_rect = pygame.Rect(x, y, cell_size, cell_size)

                # Color cell based on type
                if self.game.initial_grid[row, col] != 0:
                    # Initial clue - light blue background
                    pygame.draw.rect(self.window, LIGHT_GRAY, cell_rect)
                elif self.game.grid[row, col] != 0:
                    # User filled - white background
                    if self.game.solution[row, col] == self.game.grid[row, col]:
                        pygame.draw.rect(self.window, GREEN, cell_rect)  # Correct
                    else:
                        pygame.draw.rect(self.window, RED, cell_rect)    # Incorrect

                # Draw cell border
                border_width = 3 if (row % 3 == 0 or col % 3 == 0) else 1
                pygame.draw.rect(self.window, BLACK, cell_rect, border_width)

                # Draw number
                if self.game.grid[row, col] != 0:
                    text = font.render(str(self.game.grid[row, col]), True, BLACK)
                    text_rect = text.get_rect(center=cell_rect.center)
                    self.window.blit(text, text_rect)

    def _draw_progress_info(self):
        """Draw progress information."""
        font_large = pygame.font.Font(None, 32)
        font_medium = pygame.font.Font(None, 24)
        font_small = pygame.font.Font(None, 20)

        BLACK = (0, 0, 0)
        BLUE = (100, 149, 237)

        info = self._get_info()

        # Title
        title = font_large.render("Sudoku RL Agent", True, BLACK)
        title_rect = title.get_rect(center=(self.window_size[0] // 2, 30))
        self.window.blit(title, title_rect)

        # Progress bar
        progress_y = 500
        progress_width = 300
        progress_height = 20
        progress_x = (self.window_size[0] - progress_width) // 2

        # Background
        progress_bg = pygame.Rect(progress_x, progress_y, progress_width, progress_height)
        pygame.draw.rect(self.window, (200, 200, 200), progress_bg)

        # Fill
        fill_width = int(progress_width * info["progress"])
        if fill_width > 0:
            progress_fill = pygame.Rect(progress_x, progress_y, fill_width, progress_height)
            pygame.draw.rect(self.window, BLUE, progress_fill)

        # Border
        pygame.draw.rect(self.window, BLACK, progress_bg, 2)

        # Progress text
        progress_text = font_medium.render(f"Progress: {info['progress']*100:.1f}%", True, BLACK)
        progress_text_rect = progress_text.get_rect(center=(self.window_size[0] // 2, progress_y + 35))
        self.window.blit(progress_text, progress_text_rect)

        # Statistics
        stats_y = 570
        stats = [
            f"Steps: {info['step_count']}/{self.max_steps}",
            f"Moves: {info['moves_made']} | Invalid: {info['invalid_moves']}",
            f"Empty Cells: {info['empty_cells']}/81",
            f"Conflicts: {info['conflicts']}",
            f"Total Reward: {info['total_reward']:.1f}"
        ]

        for i, stat in enumerate(stats):
            text = font_small.render(stat, True, BLACK)
            text_rect = text.get_rect(center=(self.window_size[0] // 2, stats_y + i * 25))
            self.window.blit(text, text_rect)

    def close(self):
        """Close the environment."""
        if self.visualizer is not None:
            self.visualizer.close()
            self.visualizer = None


# Register the environment
gym.register(
    id="Sudoku-v0",
    entry_point="sudoku_env:SudokuEnv",
    max_episode_steps=1000,
)