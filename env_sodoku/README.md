# Sudoku RL Environment 🧠🔢

A sophisticated Sudoku Reinforcement Learning environment built with Gymnasium, featuring advanced visualization optimized for YouTube Shorts (9:16 aspect ratio).

## Features ✨

- **Gymnasium Compatible**: Full integration with OpenAI Gymnasium framework
- **Flat Action Space**: 729 discrete actions representing (row, col, value) combinations
- **YouTube Shorts Ready**: 9:16 aspect ratio visualization with smooth animations
- **Advanced Visualization**:
  - Real-time progress tracking with history graphs
  - Particle effects for valid/invalid moves
  - Color-coded cells (initial clues, correct, incorrect)
  - Animated progress bars and statistics
- **Multiple Difficulty Levels**: Easy, Medium, Hard puzzle generation
- **Smart Agents**: Random, Greedy, and Heuristic solving strategies
- **Performance Metrics**: Comprehensive tracking of moves, rewards, conflicts

## Installation 🚀

1. **Clone/Navigate to the environment directory**:
```bash
cd /Users/thiesmohlenhof/python/env_populat/env_sodoku
```

2. **Install dependencies** (if not already installed):
```bash
pip install gymnasium numpy pygame matplotlib opencv-python
```

## Quick Start 🎮

### Basic Usage

```python
from sudoku_env import SudokuEnv
import numpy as np

# Create environment
env = SudokuEnv(difficulty="medium", render_mode="human")

# Reset environment
observation, info = env.reset()

# Take random actions
for step in range(100):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    env.render()  # Show visualization

    if terminated or truncated:
        print(f"Episode finished! Progress: {info['progress']*100:.1f}%")
        break

env.close()
```

### Run Demo Script

```bash
# Run with heuristic agent (recommended)
python demo.py --agent heuristic --difficulty medium --episodes 3

# Run with different configurations
python demo.py --agent random --difficulty easy --episodes 5
python demo.py --agent greedy --difficulty hard --episodes 2 --no-render

# Interactive mode
python demo.py --interactive
```

## Environment Details 📋

### Action Space
- **Type**: `Discrete(729)`
- **Encoding**: `action = row * 81 + col * 9 + value`
- **Values**:
  - `row`: 0-8 (grid rows)
  - `col`: 0-8 (grid columns)
  - `value`: 0-8 (0=clear cell, 1-8 represent numbers 2-9)

### Observation Space
- **Type**: `Box(0, 9, (9, 9))`
- **Format**: 9x9 numpy array with values 0-9

### Rewards
- `+10`: Correct number placement
- `+5`: Valid but incorrect placement
- `+1`: Valid cell clearing
- `+100`: Puzzle completion
- `-1`: Invalid moves
- `-5`: Creating conflicts
- `-0.1`: Time penalty per step

### Info Dictionary
```python
{
    'progress': 0.45,          # Completion percentage (0-1)
    'conflicts': 2,            # Number of conflicting cells
    'moves_made': 23,          # Valid moves count
    'invalid_moves': 5,        # Invalid moves count
    'is_complete': False,      # Puzzle solved status
    'step_count': 156,         # Total steps taken
    'total_reward': 45.2,      # Cumulative reward
    'empty_cells': 35          # Remaining empty cells
}
```

## Agent Strategies 🤖

### 1. Random Agent
- Selects random actions from available moves
- Good baseline for comparison

### 2. Greedy Agent
- Chooses cells with fewest possible values (Most Constrained Variable)
- Better than random but not optimal

### 3. Heuristic Agent (Recommended)
- **Naked Singles**: Cells with only one possible value
- **Hidden Singles**: Numbers that can only go in one place
- **Most Constrained Variable**: Prefer cells with fewer options
- **Least Constraining Value**: Choose numbers that appear less frequently

## Visualization Features 🎨

### YouTube Shorts Optimization
- **Aspect Ratio**: 9:16 (405x720 pixels)
- **60 FPS**: Smooth animations
- **Dark Theme**: High contrast for mobile viewing
- **Large Text**: Readable on small screens

### Visual Elements
- **Grid Animation**: Smooth cell transitions
- **Progress Bar**: Real-time completion tracking with gradient
- **Progress Graph**: Historical progress visualization
- **Particle Effects**: Success/error feedback
- **Color Coding**:
  - 🔵 Initial clues (steel blue)
  - 🟢 Correct placements (green)
  - 🔴 Incorrect placements (red)
  - ⚫ Empty cells (dark gray)

### Statistics Display
- Current step and reward
- Valid/invalid move counts
- Remaining empty cells
- Conflict detection
- Real-time progress percentage

## Advanced Usage 🔧

### Custom Agent Implementation

```python
class MyAgent:
    def __init__(self, env):
        self.env = env

    def select_action(self, observation, info):
        # Your strategy here
        # observation: 9x9 numpy array
        # info: environment info dict

        # Example: Focus on empty cells
        empty_cells = []
        for row in range(9):
            for col in range(9):
                if observation[row, col] == 0:
                    empty_cells.append((row, col))

        if empty_cells:
            row, col = random.choice(empty_cells)
            value = random.randint(1, 9)
            return self.env.move_to_action(row, col, value)

        return self.env.action_space.sample()

# Use custom agent
env = SudokuEnv(render_mode="human")
agent = MyAgent(env)

observation, info = env.reset()
while True:
    action = agent.select_action(observation, info)
    observation, reward, done, truncated, info = env.step(action)
    env.render()
    if done or truncated:
        break
```

### Saving Gameplay Videos

```python
import cv2
import numpy as np

env = SudokuEnv(render_mode="rgb_array")
frames = []

observation, info = env.reset()
for step in range(500):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Capture frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    if terminated or truncated:
        break

# Save as video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('sudoku_gameplay.mp4', fourcc, 30.0, (405, 720))

for frame in frames:
    # Convert RGB to BGR for OpenCV
    bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(bgr_frame)

out.release()
env.close()
```

## File Structure 📁

```
env_sudoku/
├── __init__.py           # Package initialization
├── sudoku_game.py        # Core Sudoku game logic
├── sudoku_env.py         # Gymnasium environment
├── visualizer.py         # Enhanced visualization system
├── demo.py               # Demonstration script
└── README.md             # This file
```

## Performance Tips 🚀

1. **Use Heuristic Agent**: Significantly better success rate than random
2. **Start with Easy**: Build up difficulty gradually
3. **Monitor Conflicts**: High conflict count indicates poor strategy
4. **Track Progress**: Use progress graph to identify stagnation
5. **Adjust Max Steps**: Increase for harder puzzles

## Troubleshooting 🔧

### Common Issues

**Pygame not rendering**:
```bash
# On macOS, you might need:
export PYGAME_HIDE_SUPPORT_PROMPT=1
python demo.py
```

**ImportError**:
```bash
# Make sure you're in the correct directory
cd /Users/thiesmohlenhof/python/env_populat/env_sodoku
python demo.py
```

**Low performance**:
- Disable rendering for training: `render_mode=None`
- Reduce max_steps for faster episodes
- Use rgb_array mode for video capture only

## Future Enhancements 🚧

- [ ] Deep RL agent integration (DQN, PPO, A2C)
- [ ] Multi-puzzle batch training
- [ ] Difficulty auto-adjustment
- [ ] Sound effects for better engagement
- [ ] Tournament mode with multiple agents
- [ ] Web interface for online play

## Contributing 🤝

Feel free to extend the environment with:
- New solving algorithms
- Different puzzle variants (6x6, 12x12)
- Enhanced visualization effects
- Performance optimizations
- Bug fixes and improvements

## License 📄

This project is open source and available under the MIT License.

---

**Happy Sudoku Solving! 🎯**

*Perfect for creating engaging AI content for social media!* 📱✨