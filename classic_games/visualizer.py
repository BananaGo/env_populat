"""
Advanced Game Visualizer for YouTube Shorts optimization.
"""

import pygame
import numpy as np
import math
from typing import Dict, Any, List, Tuple, Optional
import colorsys


class GameVisualizer:
    """Enhanced visualization system optimized for YouTube Shorts format."""

    def __init__(self, window_size: Tuple[int, int] = (405, 720)):
        """Initialize the visualizer with YouTube Shorts dimensions."""
        self.window_size = window_size
        self.font_cache = {}

        # Animation states
        self.particles = []
        self.screen_shake = {'intensity': 0, 'duration': 0, 'timer': 0}
        self.flash_effects = []
        self.floating_text = []

        # Visual effects settings
        self.particle_systems = {
            'explosion': {'count': 20, 'colors': [(255, 100, 0), (255, 200, 0), (255, 255, 255)]},
            'collect': {'count': 10, 'colors': [(50, 255, 50), (100, 255, 100), (255, 255, 255)]},
            'death': {'count': 15, 'colors': [(255, 50, 50), (200, 0, 0), (100, 0, 0)]},
            'score': {'count': 8, 'colors': [(255, 215, 0), (255, 255, 0), (255, 255, 255)]}
        }

        # Color schemes
        self.colors = {
            'neon_blue': (0, 191, 255),
            'neon_green': (57, 255, 20),
            'neon_pink': (255, 20, 147),
            'neon_purple': (186, 85, 211),
            'neon_orange': (255, 165, 0),
            'electric_blue': (125, 249, 255),
            'cyber_yellow': (255, 255, 0),
            'matrix_green': (0, 255, 65),
        }

    def get_font(self, size: int, bold: bool = False) -> pygame.font.Font:
        """Get cached font."""
        key = (size, bold)
        if key not in self.font_cache:
            self.font_cache[key] = pygame.font.Font(None, size)
            if bold:
                self.font_cache[key].set_bold(True)
        return self.font_cache[key]

    def add_particle_effect(self, x: int, y: int, effect_type: str = 'explosion'):
        """Add particle effect at position."""
        if effect_type not in self.particle_systems:
            effect_type = 'explosion'

        config = self.particle_systems[effect_type]

        for _ in range(config['count']):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 8)

            self.particles.append({
                'x': x,
                'y': y,
                'vx': math.cos(angle) * speed,
                'vy': math.sin(angle) * speed,
                'color': np.random.choice(config['colors']),
                'life': 1.0,
                'size': np.random.uniform(2, 6),
                'decay': np.random.uniform(0.02, 0.05)
            })

    def add_screen_shake(self, intensity: float = 5.0, duration: float = 0.3):
        """Add screen shake effect."""
        self.screen_shake = {
            'intensity': intensity,
            'duration': duration * 1000,  # Convert to milliseconds
            'timer': pygame.time.get_ticks()
        }

    def add_flash_effect(self, color: Tuple[int, int, int] = (255, 255, 255), duration: float = 0.2):
        """Add screen flash effect."""
        self.flash_effects.append({
            'color': color,
            'duration': duration * 1000,
            'timer': pygame.time.get_ticks(),
            'alpha': 100
        })

    def add_floating_text(self, text: str, x: int, y: int,
                         color: Tuple[int, int, int] = (255, 255, 255),
                         size: int = 24, duration: float = 2.0):
        """Add floating text effect."""
        self.floating_text.append({
            'text': text,
            'x': x,
            'y': y,
            'original_y': y,
            'color': color,
            'size': size,
            'life': 1.0,
            'duration': duration * 1000,
            'timer': pygame.time.get_ticks(),
            'alpha': 255
        })

    def update_effects(self):
        """Update all visual effects."""
        current_time = pygame.time.get_ticks()

        # Update particles
        self.particles = [p for p in self.particles if self._update_particle(p)]

        # Update screen shake
        if self.screen_shake['intensity'] > 0:
            elapsed = current_time - self.screen_shake['timer']
            if elapsed >= self.screen_shake['duration']:
                self.screen_shake['intensity'] = 0

        # Update flash effects
        self.flash_effects = [f for f in self.flash_effects if self._update_flash(f, current_time)]

        # Update floating text
        self.floating_text = [t for t in self.floating_text if self._update_floating_text(t, current_time)]

    def _update_particle(self, particle: Dict) -> bool:
        """Update single particle. Returns True if particle should continue."""
        particle['x'] += particle['vx']
        particle['y'] += particle['vy']
        particle['vy'] += 0.2  # Gravity
        particle['vx'] *= 0.98  # Air resistance
        particle['life'] -= particle['decay']

        return particle['life'] > 0

    def _update_flash(self, flash: Dict, current_time: int) -> bool:
        """Update flash effect. Returns True if effect should continue."""
        elapsed = current_time - flash['timer']
        progress = min(elapsed / flash['duration'], 1.0)
        flash['alpha'] = int(100 * (1 - progress))

        return progress < 1.0

    def _update_floating_text(self, text: Dict, current_time: int) -> bool:
        """Update floating text. Returns True if text should continue."""
        elapsed = current_time - text['timer']
        progress = min(elapsed / text['duration'], 1.0)

        # Float upward
        text['y'] = text['original_y'] - (progress * 50)

        # Fade out
        text['alpha'] = int(255 * (1 - progress))
        text['life'] = 1 - progress

        return progress < 1.0

    def render_effects(self, surface: pygame.Surface):
        """Render all visual effects to surface."""
        # Apply screen shake
        shake_offset = self._get_shake_offset()

        # Render particles
        for particle in self.particles:
            if particle['life'] > 0:
                size = max(1, int(particle['size'] * particle['life']))
                alpha = int(255 * particle['life'])

                # Create particle surface with alpha
                particle_surf = pygame.Surface((size * 2, size * 2))
                particle_surf.set_alpha(alpha)
                particle_surf.fill(particle['color'])

                pygame.draw.circle(
                    particle_surf,
                    particle['color'],
                    (size, size),
                    size
                )

                pos = (
                    int(particle['x'] - size + shake_offset[0]),
                    int(particle['y'] - size + shake_offset[1])
                )
                surface.blit(particle_surf, pos)

        # Render floating text
        for text in self.floating_text:
            if text['life'] > 0:
                font = self.get_font(text['size'], bold=True)
                text_surf = font.render(text['text'], True, text['color'])
                text_surf.set_alpha(text['alpha'])

                pos = (
                    int(text['x'] + shake_offset[0]),
                    int(text['y'] + shake_offset[1])
                )
                surface.blit(text_surf, pos)

        # Render flash effects
        for flash in self.flash_effects:
            if flash['alpha'] > 0:
                flash_surf = pygame.Surface(surface.get_size())
                flash_surf.set_alpha(flash['alpha'])
                flash_surf.fill(flash['color'])
                surface.blit(flash_surf, (0, 0))

    def _get_shake_offset(self) -> Tuple[int, int]:
        """Get current screen shake offset."""
        if self.screen_shake['intensity'] <= 0:
            return (0, 0)

        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.screen_shake['timer']
        progress = min(elapsed / self.screen_shake['duration'], 1.0)

        # Decrease intensity over time
        current_intensity = self.screen_shake['intensity'] * (1 - progress)

        # Random shake offset
        offset_x = np.random.randint(-int(current_intensity), int(current_intensity) + 1)
        offset_y = np.random.randint(-int(current_intensity), int(current_intensity) + 1)

        return (offset_x, offset_y)

    def render_neon_text(self, surface: pygame.Surface, text: str, pos: Tuple[int, int],
                        size: int = 32, color: Tuple[int, int, int] = None):
        """Render text with neon glow effect."""
        if color is None:
            color = self.colors['neon_blue']

        font = self.get_font(size, bold=True)

        # Create glow layers
        glow_colors = [
            (*color, 30),   # Outer glow
            (*color, 60),   # Middle glow
            (*color, 100),  # Inner glow
            (*color, 255)   # Main text
        ]

        glow_sizes = [6, 4, 2, 0]  # Glow blur sizes

        for glow_color, glow_size in zip(glow_colors, glow_sizes):
            text_surf = font.render(text, True, glow_color[:3])

            if len(glow_color) > 3:  # Has alpha
                text_surf.set_alpha(glow_color[3])

            # Apply glow by blitting multiple times with slight offsets
            if glow_size > 0:
                for dx in range(-glow_size, glow_size + 1):
                    for dy in range(-glow_size, glow_size + 1):
                        if dx == 0 and dy == 0:
                            continue
                        surface.blit(text_surf, (pos[0] + dx, pos[1] + dy))

            # Main text
            surface.blit(text_surf, pos)

    def render_progress_bar(self, surface: pygame.Surface, progress: float,
                           pos: Tuple[int, int], size: Tuple[int, int],
                           animated: bool = True):
        """Render animated progress bar with effects."""
        x, y = pos
        width, height = size

        # Background
        bg_rect = pygame.Rect(x - 2, y - 2, width + 4, height + 4)
        pygame.draw.rect(surface, (255, 255, 255), bg_rect, border_radius=height//2 + 2)

        progress_bg = pygame.Rect(x, y, width, height)
        pygame.draw.rect(surface, (30, 30, 40), progress_bg, border_radius=height//2)

        # Animated fill
        if progress > 0:
            fill_width = int(width * progress)

            if animated:
                # Create gradient effect
                for i in range(fill_width):
                    ratio = i / width
                    # Animate hue over time
                    time_offset = pygame.time.get_ticks() * 0.001
                    hue = (0.6 - ratio * 0.4 + time_offset) % 1.0
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    color = tuple(int(c * 255) for c in rgb)

                    fill_rect = pygame.Rect(x + i, y, 1, height)
                    pygame.draw.rect(surface, color, fill_rect)
            else:
                # Static color fill
                fill_rect = pygame.Rect(x, y, fill_width, height)
                pygame.draw.rect(surface, self.colors['neon_blue'], fill_rect, border_radius=height//2)

        # Glow effect
        if progress > 0.5:  # Only glow when significant progress
            glow_rect = pygame.Rect(x - 1, y - 1, fill_width + 2, height + 2)
            glow_surf = pygame.Surface((fill_width + 2, height + 2))
            glow_surf.set_alpha(60)
            glow_surf.fill(self.colors['electric_blue'])
            surface.blit(glow_surf, (x - 1, y - 1))

    def render_score_display(self, surface: pygame.Surface, score: int,
                           high_score: int = None, pos: Tuple[int, int] = None):
        """Render animated score display."""
        if pos is None:
            pos = (surface.get_width() // 2, 120)

        # Main score
        score_text = f"SCORE: {score:,}"
        self.render_neon_text(surface, score_text, pos, 28, self.colors['cyber_yellow'])

        # High score if provided
        if high_score is not None and high_score > 0:
            high_score_text = f"BEST: {high_score:,}"
            high_pos = (pos[0], pos[1] + 35)
            self.render_neon_text(surface, high_score_text, high_pos, 18, self.colors['neon_green'])

        # Add pulsing effect for new high scores
        if high_score is not None and score >= high_score and score > 0:
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.01)) * 0.3 + 0.7
            glow_color = tuple(int(c * pulse) for c in self.colors['neon_pink'])

            # New high score indicator
            new_best_text = "NEW BEST!"
            indicator_pos = (pos[0], pos[1] + 60)
            self.render_neon_text(surface, new_best_text, indicator_pos, 16, glow_color)

    def render_game_over_overlay(self, surface: pygame.Surface, score: int, best_score: int = None):
        """Render game over screen with effects."""
        # Semi-transparent overlay
        overlay = pygame.Surface(surface.get_size())
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (0, 0))

        center_x = surface.get_width() // 2
        center_y = surface.get_height() // 2

        # Game Over text with pulsing effect
        pulse = abs(math.sin(pygame.time.get_ticks() * 0.005)) * 0.5 + 0.5
        pulse_color = tuple(int(c * pulse) for c in self.colors['neon_pink'])

        self.render_neon_text(surface, "GAME OVER", (center_x, center_y - 60), 36, pulse_color)

        # Final score
        score_text = f"FINAL SCORE: {score:,}"
        self.render_neon_text(surface, score_text, (center_x, center_y), 24, self.colors['cyber_yellow'])

        # Best score comparison
        if best_score is not None:
            if score >= best_score:
                best_text = "NEW RECORD!"
                color = self.colors['neon_green']
            else:
                best_text = f"BEST: {best_score:,}"
                color = self.colors['neon_blue']

            self.render_neon_text(surface, best_text, (center_x, center_y + 40), 20, color)

        # Restart instruction
        restart_text = "Press SPACE to restart"
        self.render_neon_text(surface, restart_text, (center_x, center_y + 100), 16, self.colors['matrix_green'])