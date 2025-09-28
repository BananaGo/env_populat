"""
Simple example script to quickly test the Sudoku RL Environment.
"""

from sudoku_env import SudokuEnv
import random

def quick_test():
    """Quick test of the environment with a simple random agent."""
    print("🎮 Quick Test: Sudoku RL Environment")
    print("=" * 40)

    # Create environment with rendering
    env = SudokuEnv(difficulty="easy", render_mode="human", max_steps=200)

    observation, info = env.reset()
    print(f"Initial progress: {info['progress']*100:.1f}%")
    print(f"Empty cells: {info['empty_cells']}")

    step = 0
    while step < 200:
        # Simple strategy: find empty cell and try a random valid number
        action_made = False

        for row in range(9):
            for col in range(9):
                if observation[row, col] == 0:  # Empty cell
                    # Try a random number
                    value = random.randint(1, 9)
                    action = env.move_to_action(row, col, value)

                    observation, reward, terminated, truncated, info = env.step(action)
                    step += 1

                    # Render the environment
                    env.render()

                    if reward > 0:  # Valid move
                        print(f"Step {step}: Placed {value} at ({row},{col}), Reward: {reward:.2f}, Progress: {info['progress']*100:.1f}%")

                    if terminated:
                        print(f"🎉 Puzzle completed in {step} steps!")
                        env.close()
                        return

                    if truncated:
                        print(f"Episode truncated after {step} steps")
                        env.close()
                        return

                    action_made = True
                    break

            if action_made:
                break

        if not action_made:
            print("No more moves possible")
            break

    print(f"Finished with {info['progress']*100:.1f}% completion")
    env.close()

if __name__ == "__main__":
    quick_test()