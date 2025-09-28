"""
Core Sudoku game logic for RL environment.
Handles grid generation, validation, and puzzle creation.
"""

import numpy as np
import random
from typing import Tuple, List, Optional


class SudokuGame:
    """Core Sudoku game logic."""

    def __init__(self):
        self.size = 9
        self.box_size = 3
        self.grid = np.zeros((self.size, self.size), dtype=int)
        self.initial_grid = np.zeros((self.size, self.size), dtype=int)
        self.solution = np.zeros((self.size, self.size), dtype=int)

    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) is valid."""
        if self.grid[row, col] != 0:  # Cell already filled
            return False

        # Check row
        if num in self.grid[row, :]:
            return False

        # Check column
        if num in self.grid[:, col]:
            return False

        # Check 3x3 box
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        box = self.grid[box_row:box_row + self.box_size,
                       box_col:box_col + self.box_size]
        if num in box:
            return False

        return True

    def make_move(self, row: int, col: int, num: int) -> bool:
        """Make a move and return if it was valid."""
        if self.is_valid_move(row, col, num):
            self.grid[row, col] = num
            return True
        return False

    def remove_number(self, row: int, col: int) -> bool:
        """Remove number from cell if it's not part of initial puzzle."""
        if self.initial_grid[row, col] == 0:  # Not an initial clue
            self.grid[row, col] = 0
            return True
        return False

    def is_complete(self) -> bool:
        """Check if the Sudoku is completely solved."""
        return np.all(self.grid != 0) and self.is_valid_state()

    def is_valid_state(self) -> bool:
        """Check if current state is valid (no conflicts)."""
        # Check all rows
        for row in range(self.size):
            non_zero = self.grid[row, :][self.grid[row, :] != 0]
            if len(non_zero) != len(set(non_zero)):
                return False

        # Check all columns
        for col in range(self.size):
            non_zero = self.grid[:, col][self.grid[:, col] != 0]
            if len(non_zero) != len(set(non_zero)):
                return False

        # Check all 3x3 boxes
        for box_row in range(0, self.size, self.box_size):
            for box_col in range(0, self.size, self.box_size):
                box = self.grid[box_row:box_row + self.box_size,
                              box_col:box_col + self.box_size].flatten()
                non_zero = box[box != 0]
                if len(non_zero) != len(set(non_zero)):
                    return False

        return True

    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of empty cells."""
        empty_cells = []
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 0:
                    empty_cells.append((row, col))
        return empty_cells

    def get_valid_numbers(self, row: int, col: int) -> List[int]:
        """Get list of valid numbers for a cell."""
        if self.grid[row, col] != 0:
            return []

        valid_numbers = []
        for num in range(1, 10):
            if self.is_valid_move(row, col, num):
                valid_numbers.append(num)
        return valid_numbers

    def solve(self) -> bool:
        """Solve the Sudoku using backtracking."""
        empty_cells = self.get_empty_cells()
        if not empty_cells:
            return True

        row, col = empty_cells[0]
        valid_numbers = self.get_valid_numbers(row, col)

        for num in valid_numbers:
            self.grid[row, col] = num
            if self.solve():
                return True
            self.grid[row, col] = 0

        return False

    def generate_complete_grid(self) -> np.ndarray:
        """Generate a complete valid Sudoku grid."""
        self.grid = np.zeros((self.size, self.size), dtype=int)

        # Fill diagonal boxes first (they don't affect each other)
        for box in range(0, self.size, self.box_size):
            self._fill_box(box, box)

        # Fill remaining cells
        self.solve()
        return self.grid.copy()

    def _fill_box(self, row_start: int, col_start: int):
        """Fill a 3x3 box with random valid numbers."""
        numbers = list(range(1, 10))
        random.shuffle(numbers)

        for i in range(self.box_size):
            for j in range(self.box_size):
                self.grid[row_start + i, col_start + j] = numbers[i * self.box_size + j]

    def create_puzzle(self, difficulty: str = "medium") -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a Sudoku puzzle by removing numbers from a complete grid.

        Args:
            difficulty: "easy" (40-45 clues), "medium" (30-35 clues), "hard" (25-30 clues)

        Returns:
            Tuple of (puzzle_grid, solution_grid)
        """
        # Generate complete solution
        solution = self.generate_complete_grid()
        puzzle = solution.copy()

        # Determine number of clues to remove based on difficulty
        clue_ranges = {
            "easy": (40, 45),
            "medium": (30, 35),
            "hard": (25, 30)
        }

        min_clues, max_clues = clue_ranges.get(difficulty, (30, 35))
        target_clues = random.randint(min_clues, max_clues)
        cells_to_remove = 81 - target_clues

        # Create list of all cell positions
        positions = [(i, j) for i in range(self.size) for j in range(self.size)]
        random.shuffle(positions)

        # Remove numbers while ensuring puzzle has unique solution
        removed = 0
        for row, col in positions:
            if removed >= cells_to_remove:
                break

            # Temporarily remove number
            original_value = puzzle[row, col]
            puzzle[row, col] = 0

            # Check if puzzle still has unique solution
            temp_game = SudokuGame()
            temp_game.grid = puzzle.copy()
            solutions_count = temp_game._count_solutions()

            if solutions_count == 1:
                removed += 1
            else:
                # Restore number if multiple solutions exist
                puzzle[row, col] = original_value

        # Set up the game state
        self.grid = puzzle.copy()
        self.initial_grid = puzzle.copy()
        self.solution = solution.copy()

        return puzzle, solution

    def _count_solutions(self, max_solutions: int = 2) -> int:
        """Count number of solutions (up to max_solutions)."""
        solutions = [0]  # Use list to allow modification in nested function

        def backtrack():
            if solutions[0] >= max_solutions:
                return

            empty_cells = self.get_empty_cells()
            if not empty_cells:
                solutions[0] += 1
                return

            row, col = empty_cells[0]
            for num in range(1, 10):
                if self.is_valid_move(row, col, num):
                    self.grid[row, col] = num
                    backtrack()
                    self.grid[row, col] = 0

        backtrack()
        return solutions[0]

    def get_progress(self) -> float:
        """Get completion progress as percentage (0-1)."""
        filled_cells = np.count_nonzero(self.grid)
        return filled_cells / (self.size * self.size)

    def get_conflicts(self) -> int:
        """Count number of conflicting cells."""
        conflicts = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] != 0:
                    # Temporarily remove the number and check if it would be valid
                    temp = self.grid[row, col]
                    self.grid[row, col] = 0
                    if not self.is_valid_move(row, col, temp):
                        conflicts += 1
                    self.grid[row, col] = temp
        return conflicts

    def reset(self, puzzle: Optional[np.ndarray] = None):
        """Reset the game to initial state or load new puzzle."""
        if puzzle is not None:
            self.grid = puzzle.copy()
            self.initial_grid = puzzle.copy()
        else:
            self.grid = self.initial_grid.copy()