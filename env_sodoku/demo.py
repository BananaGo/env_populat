"""
Demonstration script for the Sudoku RL Environment.
Shows how to use the environment with different types of agents.
"""

import numpy as np
import gymnasium as gym
from typing import List, Tuple
import random
import time
import sys
import os

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sudoku_env import SudokuEnv


class RandomAgent:
    """Random agent that selects random valid actions."""

    def __init__(self, env: SudokuEnv):
        self.env = env
        self.action_space = env.action_space

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select a random action from valid actions."""
        # Get empty cells and try to fill them
        empty_cells = []
        for row in range(9):
            for col in range(9):
                if observation[row, col] == 0:
                    empty_cells.append((row, col))

        if not empty_cells:
            return random.randint(0, self.action_space.n - 1)

        # Focus on empty cells with random valid numbers
        row, col = random.choice(empty_cells)
        value = random.randint(1, 9)  # 1-9 for placing numbers

        # Convert to action
        action = self.env.move_to_action(row, col, value)
        return action


class GreedyAgent:
    """Greedy agent that tries to make the most logical moves."""

    def __init__(self, env: SudokuEnv):
        self.env = env
        self.action_space = env.action_space

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select action using greedy strategy."""
        # Find the cell with the fewest valid options
        best_cell = None
        min_options = 10

        for row in range(9):
            for col in range(9):
                if observation[row, col] == 0:  # Empty cell
                    valid_numbers = self._get_valid_numbers(observation, row, col)
                    if len(valid_numbers) > 0 and len(valid_numbers) < min_options:
                        best_cell = (row, col, valid_numbers)
                        min_options = len(valid_numbers)

        if best_cell is None:
            # No valid moves, return random action
            return random.randint(0, self.action_space.n - 1)

        row, col, valid_numbers = best_cell
        value = random.choice(valid_numbers)

        return self.env.move_to_action(row, col, value)

    def _get_valid_numbers(self, grid: np.ndarray, row: int, col: int) -> List[int]:
        """Get list of valid numbers for a cell."""
        if grid[row, col] != 0:
            return []

        valid_numbers = []
        for num in range(1, 10):
            if self._is_valid_placement(grid, row, col, num):
                valid_numbers.append(num)
        return valid_numbers

    def _is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in grid[row, :]:
            return False

        # Check column
        if num in grid[:, col]:
            return False

        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        box = grid[box_row:box_row + 3, box_col:box_col + 3]
        if num in box:
            return False

        return True


class HeuristicAgent:
    """Advanced heuristic agent using multiple strategies."""

    def __init__(self, env: SudokuEnv):
        self.env = env
        self.action_space = env.action_space

    def select_action(self, observation: np.ndarray, info: dict) -> int:
        """Select action using advanced heuristics."""
        # Strategy 1: Naked singles (cells with only one possible value)
        action = self._find_naked_single(observation)
        if action is not None:
            return action

        # Strategy 2: Hidden singles (numbers that can only go in one place in a unit)
        action = self._find_hidden_single(observation)
        if action is not None:
            return action

        # Strategy 3: Most constrained cell (minimum remaining values heuristic)
        action = self._find_most_constrained_cell(observation)
        if action is not None:
            return action

        # Fallback: random valid move
        return self._random_valid_move(observation)

    def _find_naked_single(self, grid: np.ndarray) -> int:
        """Find cells that have only one possible value."""
        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:
                    valid_numbers = self._get_valid_numbers(grid, row, col)
                    if len(valid_numbers) == 1:
                        return self.env.move_to_action(row, col, valid_numbers[0])
        return None

    def _find_hidden_single(self, grid: np.ndarray) -> int:
        """Find numbers that can only go in one place in a row/column/box."""
        # Check rows
        for row in range(9):
            for num in range(1, 10):
                if num not in grid[row, :]:
                    possible_cols = []
                    for col in range(9):
                        if grid[row, col] == 0 and self._is_valid_placement(grid, row, col, num):
                            possible_cols.append(col)
                    if len(possible_cols) == 1:
                        return self.env.move_to_action(row, possible_cols[0], num)

        # Check columns
        for col in range(9):
            for num in range(1, 10):
                if num not in grid[:, col]:
                    possible_rows = []
                    for row in range(9):
                        if grid[row, col] == 0 and self._is_valid_placement(grid, row, col, num):
                            possible_rows.append(row)
                    if len(possible_rows) == 1:
                        return self.env.move_to_action(possible_rows[0], col, num)

        # Check 3x3 boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = grid[box_row:box_row + 3, box_col:box_col + 3]
                for num in range(1, 10):
                    if num not in box:
                        possible_positions = []
                        for r in range(box_row, box_row + 3):
                            for c in range(box_col, box_col + 3):
                                if grid[r, c] == 0 and self._is_valid_placement(grid, r, c, num):
                                    possible_positions.append((r, c))
                        if len(possible_positions) == 1:
                            row, col = possible_positions[0]
                            return self.env.move_to_action(row, col, num)

        return None

    def _find_most_constrained_cell(self, grid: np.ndarray) -> int:
        """Find the empty cell with the fewest possible values."""
        best_cell = None
        min_options = 10

        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:
                    valid_numbers = self._get_valid_numbers(grid, row, col)
                    if 0 < len(valid_numbers) < min_options:
                        best_cell = (row, col, valid_numbers)
                        min_options = len(valid_numbers)

        if best_cell:
            row, col, valid_numbers = best_cell
            # Choose the number that appears least frequently in the grid
            number_counts = {}
            for num in valid_numbers:
                number_counts[num] = np.count_nonzero(grid == num)

            best_number = min(valid_numbers, key=lambda x: number_counts[x])
            return self.env.move_to_action(row, col, best_number)

        return None

    def _random_valid_move(self, grid: np.ndarray) -> int:
        """Make a random valid move as fallback."""
        empty_cells = []
        for row in range(9):
            for col in range(9):
                if grid[row, col] == 0:
                    valid_numbers = self._get_valid_numbers(grid, row, col)
                    for num in valid_numbers:
                        empty_cells.append((row, col, num))

        if empty_cells:
            row, col, num = random.choice(empty_cells)
            return self.env.move_to_action(row, col, num)

        return random.randint(0, self.action_space.n - 1)

    def _get_valid_numbers(self, grid: np.ndarray, row: int, col: int) -> List[int]:
        """Get list of valid numbers for a cell."""
        if grid[row, col] != 0:
            return []

        valid_numbers = []
        for num in range(1, 10):
            if self._is_valid_placement(grid, row, col, num):
                valid_numbers.append(num)
        return valid_numbers

    def _is_valid_placement(self, grid: np.ndarray, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        # Check row
        if num in grid[row, :]:
            return False

        # Check column
        if num in grid[:, col]:
            return False

        # Check 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        box = grid[box_row:box_row + 3, box_col:box_col + 3]
        if num in box:
            return False

        return True


def run_demo(agent_type: str = "heuristic", difficulty: str = "medium",
             render: bool = True, max_episodes: int = 5):
    """
    Run demonstration with specified agent type.

    Args:
        agent_type: "random", "greedy", or "heuristic"
        difficulty: "easy", "medium", or "hard"
        render: Whether to render the environment
        max_episodes: Maximum number of episodes to run
    """
    print(f"🎮 Running Sudoku RL Demo")
    print(f"Agent: {agent_type.title()}")
    print(f"Difficulty: {difficulty.title()}")
    print(f"Render: {'Yes' if render else 'No'}")
    print("-" * 50)

    # Create environment
    render_mode = "human" if render else None
    env = SudokuEnv(difficulty=difficulty, render_mode=render_mode, max_steps=500)

    # Create agent
    agents = {
        "random": RandomAgent,
        "greedy": GreedyAgent,
        "heuristic": HeuristicAgent
    }

    if agent_type not in agents:
        print(f"Unknown agent type: {agent_type}")
        return

    agent = agents[agent_type](env)

    # Run episodes
    episode_results = []

    for episode in range(max_episodes):
        print(f"\n📈 Episode {episode + 1}/{max_episodes}")

        observation, info = env.reset()
        total_reward = 0
        steps = 0
        done = False

        start_time = time.time()

        while not done:
            # Select action
            action = agent.select_action(observation, info)

            # Take step
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            done = terminated or truncated

            # Render if enabled
            if render:
                env.render()
                time.sleep(0.05)  # Slow down for better visualization

            # Print progress occasionally
            if steps % 50 == 0:
                progress = info.get('progress', 0) * 100
                print(f"  Step {steps}: Progress {progress:.1f}%, Reward: {total_reward:.2f}")

        episode_time = time.time() - start_time

        # Episode summary
        final_progress = info.get('progress', 0) * 100
        completed = info.get('is_complete', False)
        moves_made = info.get('moves_made', 0)
        invalid_moves = info.get('invalid_moves', 0)

        print(f"✅ Episode {episode + 1} Results:")
        print(f"   Completed: {'Yes' if completed else 'No'}")
        print(f"   Progress: {final_progress:.1f}%")
        print(f"   Steps: {steps}")
        print(f"   Valid moves: {moves_made}")
        print(f"   Invalid moves: {invalid_moves}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Time: {episode_time:.2f}s")

        episode_results.append({
            'episode': episode + 1,
            'completed': completed,
            'progress': final_progress,
            'steps': steps,
            'moves_made': moves_made,
            'invalid_moves': invalid_moves,
            'total_reward': total_reward,
            'time': episode_time
        })

        if render:
            time.sleep(2)  # Pause between episodes

    # Overall statistics
    print("\n" + "="*50)
    print("📊 OVERALL STATISTICS")
    print("="*50)

    completed_episodes = sum(1 for r in episode_results if r['completed'])
    avg_progress = np.mean([r['progress'] for r in episode_results])
    avg_steps = np.mean([r['steps'] for r in episode_results])
    avg_reward = np.mean([r['total_reward'] for r in episode_results])
    avg_time = np.mean([r['time'] for r in episode_results])

    print(f"Completed puzzles: {completed_episodes}/{max_episodes} ({completed_episodes/max_episodes*100:.1f}%)")
    print(f"Average progress: {avg_progress:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Average time: {avg_time:.2f}s")

    # Success rate by difficulty
    if completed_episodes > 0:
        avg_steps_successful = np.mean([r['steps'] for r in episode_results if r['completed']])
        print(f"Average steps (successful): {avg_steps_successful:.1f}")

    env.close()
    return episode_results


def interactive_demo():
    """Run interactive demo where user can choose parameters."""
    print("🎮 Sudoku RL Environment - Interactive Demo")
    print("=" * 50)

    # Get user preferences
    agent_types = ["random", "greedy", "heuristic"]
    difficulties = ["easy", "medium", "hard"]

    print("\nAvailable agents:")
    for i, agent in enumerate(agent_types, 1):
        print(f"  {i}. {agent.title()}")

    while True:
        try:
            agent_choice = int(input("\nSelect agent (1-3): ")) - 1
            if 0 <= agent_choice < len(agent_types):
                break
            print("Invalid choice. Please select 1-3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    print("\nAvailable difficulties:")
    for i, diff in enumerate(difficulties, 1):
        print(f"  {i}. {diff.title()}")

    while True:
        try:
            diff_choice = int(input("\nSelect difficulty (1-3): ")) - 1
            if 0 <= diff_choice < len(difficulties):
                break
            print("Invalid choice. Please select 1-3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        render_choice = input("\nEnable visualization? (y/n): ").lower()
        if render_choice in ['y', 'yes', 'n', 'no']:
            break
        print("Invalid choice. Please enter 'y' or 'n'.")

    while True:
        try:
            episodes = int(input("\nNumber of episodes (1-10): "))
            if 1 <= episodes <= 10:
                break
            print("Invalid choice. Please select 1-10.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Run demo
    run_demo(
        agent_type=agent_types[agent_choice],
        difficulty=difficulties[diff_choice],
        render=render_choice.startswith('y'),
        max_episodes=episodes
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sudoku RL Environment Demo")
    parser.add_argument("--agent", choices=["random", "greedy", "heuristic"],
                       default="heuristic", help="Agent type")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"],
                       default="medium", help="Puzzle difficulty")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--interactive", action="store_true", help="Run interactive demo")

    args = parser.parse_args()

    if args.interactive:
        interactive_demo()
    else:
        run_demo(
            agent_type=args.agent,
            difficulty=args.difficulty,
            render=not args.no_render,
            max_episodes=args.episodes
        )