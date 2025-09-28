import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.PngImagePlugin")

import numpy as np
from ple import PLE

from ple.games.flappybird import FlappyBird
from ple.games.catcher import Catcher
from ple.games.pixelcopter import Pixelcopter
from ple.games.pong import Pong
from ple.games.snake import Snake

def env_raw():
     env = FlappyBird()
     ple_env = PLE(env, fps=30, display_screen=True)
     return ple_env

def env_raw_flappybird():
     env = FlappyBird()
     ple_env = PLE(env, fps=30, display_screen=True)
     return ple_env

def env_raw_catcher():
     env = Catcher()
     ple_env = PLE(env, fps=30, display_screen=True)
     return ple_env

def env_raw_pixelcopter():
     env = Pixelcopter()
     ple_env = PLE(env, fps=30, display_screen=True)
     return ple_env

def env_raw_pong():
     env = Pong()
     ple_env = PLE(env, fps=30, display_screen=True)
     return ple_env

def env_raw_snake():
     env = Snake()
     ple_env = PLE(env, fps=30, display_screen=True)
     return ple_env

def get_env_collection():
     """
     Returns a dictionary of environment constructors for all supported games.
     """
     return {
          "flappybird": env_raw_flappybird,
          "catcher": env_raw_catcher,
          "pixelcopter": env_raw_pixelcopter,
          "pong": env_raw_pong,
          "snake": env_raw_snake,
     }
