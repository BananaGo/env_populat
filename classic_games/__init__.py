"""
Classic Games RL Environment Package

YouTube Shorts optimized implementations of classic arcade games
from the PyGame Learning Environment, built for reinforcement learning.
"""

from .base_game import BaseGameEnv
from .visualizer import GameVisualizer
from .recorder import GameRecorder

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "BaseGameEnv",
    "GameVisualizer",
    "GameRecorder",
    "CatcherEnv",
    "FlappyBirdEnv",
    "PixelcopterEnv",
    "PongEnv",
    "SnakeEnv"
]