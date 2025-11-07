"""
Play trained Lunar Lander model with visual rendering.

This script loads the best trained model and plays episodes with visualization,
providing a way to observe the agent's learned behavior.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import time
from pathlib import Path
from typing import Optional

from dqn_agent import DQNAgent
from constants import (
    ENVIRONMENT_NAME,
    MODEL_BEST_FILENAME,
    PERFECT_LANDING_SCORE,
    DECENT_LANDING_SCORE,
    MAX_EPISODE_STEPS
)


def play_best_model(
    model_path: str | Path = f"models/{MODEL_BEST_FILENAME}",
    episodes: int = 5,
    delay: float = 0.02
) -> None:
    """
    Play the best trained model with visual rendering.

    Args:
        model_path: Path to the saved model (default: models/moonlander_best.pth)
        episodes: Number of episodes to play (default: 5)
        delay: Delay between frames in seconds for better visualization (default: 0.02)

    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If there's an error loading or running the model
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Model file not found: {model_path}\n"
            f"Please train the model first by running: python train.py"
        )

    try:
        # Create environment with rendering
        env = gym.make(ENVIRONMENT_NAME, render_mode='human')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        # Load agent
        agent = DQNAgent(state_size, action_size)
        agent.load(str(model_path))
        agent.epsilon = 0  # No exploration, only exploitation

        print(f"🎮 Playing {episodes} episodes with the best model...")
        print(f"📂 Model: {model_path}")
        print("🎯 Actions: 0=Do nothing, 1=Fire left, 2=Fire main, 3=Fire right")
        print("🏁 Goal: Land between the flags with minimal fuel usage\n")

        total_scores = []

        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            step_count = 0

            print(f"Episode {episode + 1}/{episodes}")

            while True:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                state = next_state
                total_reward += reward
                step_count += 1

                # Add small delay for better visualization
                time.sleep(delay)

                if done:
                    break

                # Safety limit
                if step_count > MAX_EPISODE_STEPS:
                    break

            total_scores.append(total_reward)

            # Classify performance
            if terminated:
                if total_reward >= PERFECT_LANDING_SCORE:
                    status = "✅ Successful landing!"
                    print(f"{status} Score: {total_reward:.2f}")
                elif total_reward >= DECENT_LANDING_SCORE:
                    status = "🟡 Decent landing."
                    print(f"{status} Score: {total_reward:.2f}")
                else:
                    status = "❌ Crashed or poor landing."
                    print(f"{status} Score: {total_reward:.2f}")
            else:
                status = "⏰ Episode timed out."
                print(f"{status} Score: {total_reward:.2f}")

            print(f"Steps taken: {step_count}\n")

        env.close()

        # Print summary
        print("=" * 60)
        print("=== Performance Summary ===")
        print("=" * 60)
        print(f"Average Score: {np.mean(total_scores):.2f}")
        print(f"Best Score: {max(total_scores):.2f}")
        print(f"Worst Score: {min(total_scores):.2f}")
        print(f"Score Std Dev: {np.std(total_scores):.2f}")

        # Success rate
        success_count = len([s for s in total_scores if s >= PERFECT_LANDING_SCORE])
        success_rate = (success_count / len(total_scores)) * 100
        print(f"Success Rate: {success_count}/{len(total_scores)} ({success_rate:.1f}%)")

    except FileNotFoundError:
        raise
    except Exception as e:
        raise Exception(f"❌ Error during playback: {e}") from e


def main() -> None:
    """Main entry point for playing the best model."""
    import sys

    # Parse command line arguments
    model_path = f"models/{MODEL_BEST_FILENAME}"
    episodes = 5
    delay = 0.02

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            episodes = int(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid episode count '{sys.argv[2]}', using default {episodes}")
    if len(sys.argv) > 3:
        try:
            delay = float(sys.argv[3])
        except ValueError:
            print(f"Warning: Invalid delay '{sys.argv[3]}', using default {delay}")

    try:
        play_best_model(model_path, episodes, delay)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
