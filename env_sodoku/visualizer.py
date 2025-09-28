"""
Enhanced YouTube Shorts optimized visualization for Sudoku RL Environment.
"""

import pygame
import numpy as np
import math
from typing import Tuple, Dict, Any, Optional, List
import colorsys


class SudokuVisualizer:
    """Advanced visualization system for Sudoku RL environment optimized for YouTube Shorts."""

    def __init__(self, window_size: Tuple[int, int] = (405, 720)):
        """
        Initialize the visualizer.

        Args:
            window_size: Window dimensions (width, height) optimized for 9:16 aspect ratio
        """
        self.window_size = window_size
        self.window = None
        self.clock = None
        self.font_cache = {}

        # Animation states
        self.animations = []
        self.particle_effects = []
        self.cell_highlights = {}
        self.number_animations = {}

        # Visual parameters
        self.grid_size = min(window_size[0] - 40, 360)  # Leave margins
        self.grid_offset_x = (window_size[0] - self.grid_size) // 2
        self.grid_offset_y = 120  # Space for header
        self.cell_size = self.grid_size // 9

        # Color scheme
        self.colors = {
            'background': (15, 15, 25),           # Dark blue-black
            'grid_bg': (30, 30, 40),              # Dark gray
            'cell_empty': (45, 45, 55),           # Light gray
            'cell_initial': (70, 130, 180),       # Steel blue
            'cell_user_correct': (50, 205, 50),   # Lime green
            'cell_user_wrong': (220, 20, 60),     # Crimson
            'cell_highlight': (255, 215, 0),      # Gold
            'grid_line_thin': (80, 80, 90),       # Light gray
            'grid_line_thick': (200, 200, 210),   # White-ish
            'text_primary': (255, 255, 255),      # White
            'text_secondary': (180, 180, 190),    # Light gray
            'progress_bg': (60, 60, 70),          # Dark gray
            'progress_fill': (100, 149, 237),     # Cornflower blue
            'accent': (255, 69, 0),               # Red-orange
        }

        # Progress visualization
        self.progress_history = []
        self.max_history = 300  # Store last 5 seconds at 60 FPS

    def initialize(self):
        """Initialize pygame and create window."""
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Sudoku RL - YouTube Shorts")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        # Initialize fonts
        self._init_fonts()

    def _init_fonts(self):
        """Initialize font cache."""
        font_sizes = [16, 20, 24, 28, 32, 36, 40, 48]
        for size in font_sizes:
            self.font_cache[size] = pygame.font.Font(None, size)

    def get_font(self, size: int) -> pygame.font.Font:
        """Get font of specified size."""
        if size not in self.font_cache:
            self.font_cache[size] = pygame.font.Font(None, size)
        return self.font_cache[size]

    def add_cell_animation(self, row: int, col: int, animation_type: str, duration: float = 1.0):
        """Add animation for a specific cell."""
        self.animations.append({
            'type': animation_type,
            'row': row,
            'col': col,
            'start_time': pygame.time.get_ticks(),
            'duration': duration * 1000,  # Convert to milliseconds
            'progress': 0.0
        })

    def add_particle_effect(self, x: int, y: int, effect_type: str = 'success'):
        """Add particle effect at specified position."""
        colors = {
            'success': [(50, 205, 50), (144, 238, 144), (255, 255, 255)],
            'error': [(220, 20, 60), (255, 99, 71), (255, 255, 255)],
            'complete': [(255, 215, 0), (255, 140, 0), (255, 255, 255)]
        }

        effect_colors = colors.get(effect_type, colors['success'])

        for _ in range(12):  # Create multiple particles
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 6)

            self.particle_effects.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'color': np.random.choice(effect_colors),
                'life': 1.0,
                'size': np.random.uniform(2, 4),
                'decay': np.random.uniform(0.02, 0.04)
            })

    def update_animations(self):
        """Update all animations and effects."""
        current_time = pygame.time.get_ticks()

        # Update cell animations
        self.animations = [
            anim for anim in self.animations
            if self._update_animation(anim, current_time)
        ]

        # Update particle effects
        self.particle_effects = [
            particle for particle in self.particle_effects
            if self._update_particle(particle)
        ]

    def _update_animation(self, anim: Dict, current_time: int) -> bool:
        """Update single animation. Returns True if animation should continue."""
        elapsed = current_time - anim['start_time']
        anim['progress'] = min(elapsed / anim['duration'], 1.0)

        return anim['progress'] < 1.0

    def _update_particle(self, particle: Dict) -> bool:
        """Update single particle. Returns True if particle should continue."""
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        particle['vy'] += 0.1  # Gravity
        particle['life'] -= particle['decay']

        return particle['life'] > 0

    def render(self, game_state: Dict[str, Any], info: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Render the complete visualization.

        Args:
            game_state: Current game state including grid, initial_grid, solution
            info: Additional info from environment

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        if self.window is None:
            self.initialize()

        # Update progress history
        self._update_progress_history(info.get('progress', 0))

        # Update animations
        self.update_animations()

        # Clear screen with background
        self.window.fill(self.colors['background'])

        # Render components
        self._render_header(info)
        self._render_sudoku_grid(game_state, info)
        self._render_progress_section(info)
        self._render_statistics(info)
        self._render_effects()

        # Handle events (for human rendering)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS for smooth animations

        # Return RGB array if requested
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window)),
            axes=(1, 0, 2)
        )

    def _update_progress_history(self, progress: float):
        """Update progress history for graph visualization."""
        self.progress_history.append(progress)
        if len(self.progress_history) > self.max_history:
            self.progress_history.pop(0)

    def _render_header(self, info: Dict[str, Any]):
        """Render header with title and key metrics."""
        font_title = self.get_font(36)
        font_subtitle = self.get_font(20)

        # Main title
        title_text = font_title.render("🧠 AI SOLVING SUDOKU 🧠", True, self.colors['text_primary'])
        title_rect = title_text.get_rect(center=(self.window_size[0] // 2, 30))
        self.window.blit(title_text, title_rect)

        # Subtitle with episode info
        episode_text = font_subtitle.render(
            f"Step {info.get('step_count', 0)} | Reward: {info.get('total_reward', 0):.1f}",
            True, self.colors['text_secondary']
        )
        episode_rect = episode_text.get_rect(center=(self.window_size[0] // 2, 55))
        self.window.blit(episode_text, episode_rect)

        # Animated separator line
        line_y = 75
        line_width = 200
        line_x = (self.window_size[0] - line_width) // 2

        # Create gradient effect
        for i in range(line_width):
            alpha = int(255 * (1 - abs(i - line_width/2) / (line_width/2)))
            color = (*self.colors['accent'], alpha)

            # Create surface with per-pixel alpha
            line_surf = pygame.Surface((1, 3))
            line_surf.set_alpha(alpha)
            line_surf.fill(self.colors['accent'])
            self.window.blit(line_surf, (line_x + i, line_y))

    def _render_sudoku_grid(self, game_state: Dict[str, Any], info: Dict[str, Any]):
        """Render the main Sudoku grid with animations."""
        grid = game_state.get('grid', np.zeros((9, 9)))
        initial_grid = game_state.get('initial_grid', np.zeros((9, 9)))
        solution = game_state.get('solution', np.zeros((9, 9)))

        # Draw grid background
        grid_rect = pygame.Rect(
            self.grid_offset_x - 5,
            self.grid_offset_y - 5,
            self.grid_size + 10,
            self.grid_size + 10
        )
        pygame.draw.rect(self.window, self.colors['grid_bg'], grid_rect, border_radius=10)

        # Draw cells
        font_number = self.get_font(28)

        for row in range(9):
            for col in range(9):
                self._render_cell(row, col, grid, initial_grid, solution, font_number)

        # Draw grid lines
        self._draw_grid_lines()

        # Draw 3x3 box separators with thicker lines
        for i in range(1, 3):
            # Vertical lines
            x = self.grid_offset_x + i * 3 * self.cell_size
            pygame.draw.line(
                self.window,
                self.colors['grid_line_thick'],
                (x, self.grid_offset_y),
                (x, self.grid_offset_y + self.grid_size),
                3
            )

            # Horizontal lines
            y = self.grid_offset_y + i * 3 * self.cell_size
            pygame.draw.line(
                self.window,
                self.colors['grid_line_thick'],
                (self.grid_offset_x, y),
                (self.grid_offset_x + self.grid_size, y),
                3
            )

    def _render_cell(self, row: int, col: int, grid: np.ndarray,
                    initial_grid: np.ndarray, solution: np.ndarray, font: pygame.font.Font):
        """Render individual cell with animations and effects."""
        x = self.grid_offset_x + col * self.cell_size
        y = self.grid_offset_y + row * self.cell_size

        cell_rect = pygame.Rect(x, y, self.cell_size, self.cell_size)

        # Determine cell type and color
        cell_value = grid[row, col]
        is_initial = initial_grid[row, col] != 0
        is_correct = cell_value != 0 and solution[row, col] == cell_value

        # Base cell color
        if is_initial:
            cell_color = self.colors['cell_initial']
        elif cell_value == 0:
            cell_color = self.colors['cell_empty']
        elif is_correct:
            cell_color = self.colors['cell_user_correct']
        else:
            cell_color = self.colors['cell_user_wrong']

        # Apply animations
        final_color = self._apply_cell_animations(row, col, cell_color)

        # Draw cell
        pygame.draw.rect(self.window, final_color, cell_rect)

        # Draw number if present
        if cell_value != 0:
            # Add subtle glow effect for correct numbers
            if is_correct and not is_initial:
                glow_rect = cell_rect.inflate(4, 4)
                pygame.draw.rect(self.window, (255, 255, 255, 30), glow_rect, border_radius=3)

            text_color = self.colors['text_primary'] if is_initial else (0, 0, 0)
            number_text = font.render(str(cell_value), True, text_color)
            number_rect = number_text.get_rect(center=cell_rect.center)

            # Apply number animations
            animated_rect = self._apply_number_animations(row, col, number_rect)
            self.window.blit(number_text, animated_rect)

        # Draw cell border
        pygame.draw.rect(self.window, self.colors['grid_line_thin'], cell_rect, 1)

    def _apply_cell_animations(self, row: int, col: int, base_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Apply animations to cell color."""
        final_color = base_color

        for anim in self.animations:
            if anim['row'] == row and anim['col'] == col:
                if anim['type'] == 'flash':
                    # Flash effect
                    flash_intensity = abs(math.sin(anim['progress'] * math.pi * 4)) * (1 - anim['progress'])
                    flash_color = tuple(min(255, int(c + flash_intensity * 100)) for c in base_color)
                    final_color = flash_color
                elif anim['type'] == 'pulse':
                    # Pulse effect
                    pulse_intensity = (math.sin(anim['progress'] * math.pi * 2) + 1) * 0.5
                    pulse_color = tuple(min(255, int(c + pulse_intensity * 50)) for c in base_color)
                    final_color = pulse_color

        return final_color

    def _apply_number_animations(self, row: int, col: int, rect: pygame.Rect) -> pygame.Rect:
        """Apply animations to number position/scale."""
        animated_rect = rect.copy()

        for anim in self.animations:
            if anim['row'] == row and anim['col'] == col:
                if anim['type'] == 'pop_in':
                    # Scale animation
                    scale = anim['progress']
                    if scale < 1.0:
                        new_width = int(rect.width * scale)
                        new_height = int(rect.height * scale)
                        animated_rect = pygame.Rect(0, 0, new_width, new_height)
                        animated_rect.center = rect.center

        return animated_rect

    def _draw_grid_lines(self):
        """Draw grid lines."""
        for i in range(10):  # 0 to 9 for 10 lines
            # Vertical lines
            x = self.grid_offset_x + i * self.cell_size
            line_width = 3 if i % 3 == 0 else 1
            color = self.colors['grid_line_thick'] if i % 3 == 0 else self.colors['grid_line_thin']

            pygame.draw.line(
                self.window, color,
                (x, self.grid_offset_y),
                (x, self.grid_offset_y + self.grid_size),
                line_width
            )

            # Horizontal lines
            y = self.grid_offset_y + i * self.cell_size
            pygame.draw.line(
                self.window, color,
                (self.grid_offset_x, y),
                (self.grid_offset_x + self.grid_size, y),
                line_width
            )

    def _render_progress_section(self, info: Dict[str, Any]):
        """Render progress visualization with history graph."""
        progress_y = self.grid_offset_y + self.grid_size + 30

        # Progress bar
        progress = info.get('progress', 0)
        self._render_progress_bar(progress_y, progress)

        # Progress history graph
        if len(self.progress_history) > 1:
            self._render_progress_graph(progress_y + 60)

    def _render_progress_bar(self, y: int, progress: float):
        """Render animated progress bar."""
        bar_width = 300
        bar_height = 25
        bar_x = (self.window_size[0] - bar_width) // 2

        # Background
        bg_rect = pygame.Rect(bar_x - 2, y - 2, bar_width + 4, bar_height + 4)
        pygame.draw.rect(self.window, self.colors['text_primary'], bg_rect, border_radius=15)

        progress_bg = pygame.Rect(bar_x, y, bar_width, bar_height)
        pygame.draw.rect(self.window, self.colors['progress_bg'], progress_bg, border_radius=12)

        # Animated fill
        if progress > 0:
            fill_width = int(bar_width * progress)

            # Create gradient effect
            for i in range(fill_width):
                ratio = i / bar_width
                # Create hue shift from blue to green
                hue = 0.6 - (ratio * 0.4)  # 0.6 (blue) to 0.2 (green)
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                color = tuple(int(c * 255) for c in rgb)

                fill_rect = pygame.Rect(bar_x + i, y, 1, bar_height)
                pygame.draw.rect(self.window, color, fill_rect)

        # Progress text
        font = self.get_font(20)
        progress_text = font.render(f"{progress*100:.1f}% Complete", True, self.colors['text_primary'])
        text_rect = progress_text.get_rect(center=(self.window_size[0] // 2, y + bar_height + 20))
        self.window.blit(progress_text, text_rect)

    def _render_progress_graph(self, y: int):
        """Render mini progress history graph."""
        if not self.progress_history:
            return

        graph_width = 280
        graph_height = 40
        graph_x = (self.window_size[0] - graph_width) // 2

        # Background
        graph_rect = pygame.Rect(graph_x, y, graph_width, graph_height)
        pygame.draw.rect(self.window, self.colors['progress_bg'], graph_rect, border_radius=5)

        # Draw progress line
        if len(self.progress_history) > 1:
            points = []
            for i, progress in enumerate(self.progress_history[-graph_width:]):
                x = graph_x + (i * graph_width) // len(self.progress_history[-graph_width:])
                point_y = y + graph_height - int(progress * graph_height)
                points.append((x, point_y))

            if len(points) > 1:
                pygame.draw.lines(self.window, self.colors['accent'], False, points, 2)

        # Graph border
        pygame.draw.rect(self.window, self.colors['grid_line_thin'], graph_rect, 1)

    def _render_statistics(self, info: Dict[str, Any]):
        """Render statistics panel."""
        stats_y = self.grid_offset_y + self.grid_size + 150
        font = self.get_font(18)

        stats = [
            f"🎯 Moves: {info.get('moves_made', 0)}",
            f"❌ Invalid: {info.get('invalid_moves', 0)}",
            f"📊 Empty: {info.get('empty_cells', 81)}/81",
            f"⚡ Conflicts: {info.get('conflicts', 0)}",
        ]

        # Render stats in 2x2 grid
        for i, stat in enumerate(stats):
            x_offset = (i % 2) * 200 + 10
            y_offset = (i // 2) * 30

            stat_text = font.render(stat, True, self.colors['text_secondary'])
            self.window.blit(stat_text, (x_offset, stats_y + y_offset))

    def _render_effects(self):
        """Render particle effects and other visual effects."""
        for particle in self.particle_effects:
            size = int(particle['size'] * particle['life'])
            if size > 0:
                alpha = int(255 * particle['life'])
                color = (*particle['color'][:3], alpha)

                # Create surface for alpha blending
                particle_surf = pygame.Surface((size * 2, size * 2))
                particle_surf.set_alpha(alpha)
                particle_surf.fill(particle['color'])

                pygame.draw.circle(
                    particle_surf,
                    particle['color'],
                    (size, size),
                    size
                )

                self.window.blit(
                    particle_surf,
                    (int(particle['x'] - size), int(particle['y'] - size))
                )

    def notify_move(self, row: int, col: int, is_valid: bool, is_correct: bool = None):
        """Notify visualizer of a move for appropriate animations."""
        if is_valid:
            self.add_cell_animation(row, col, 'pop_in', 0.3)
            if is_correct:
                cell_x = self.grid_offset_x + col * self.cell_size + self.cell_size // 2
                cell_y = self.grid_offset_y + row * self.cell_size + self.cell_size // 2
                self.add_particle_effect(cell_x, cell_y, 'success')
            else:
                self.add_cell_animation(row, col, 'pulse', 0.5)
        else:
            self.add_cell_animation(row, col, 'flash', 0.3)
            cell_x = self.grid_offset_x + col * self.cell_size + self.cell_size // 2
            cell_y = self.grid_offset_y + row * self.cell_size + self.cell_size // 2
            self.add_particle_effect(cell_x, cell_y, 'error')

    def notify_completion(self):
        """Notify visualizer of puzzle completion."""
        # Add celebration effects
        center_x = self.window_size[0] // 2
        center_y = self.grid_offset_y + self.grid_size // 2

        for _ in range(30):  # Lots of particles for celebration
            self.add_particle_effect(
                center_x + np.random.randint(-100, 100),
                center_y + np.random.randint(-100, 100),
                'complete'
            )

    def close(self):
        """Clean up resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None