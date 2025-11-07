"""
Play/Visualize Trained GA Controller

Loads a trained genetic algorithm controller and visualizes it in the environment.

Usage:
    python play_ga.py --model ga_models/ga_best.npy --episodes 5
"""

import argparse
import time
import numpy as np
import gymnasium as gym

from linear_controller import LinearController
from ga_evaluator import GAEvaluator


def play_controller(controller, num_episodes=5, render=True, delay=0.02):
    """
    Play episodes with a controller and render them.

    Args:
        controller: Controller to use
        num_episodes: Number of episodes to play
        render: Whether to render visually
        delay: Delay between frames (seconds)
    """
    # Create environment with rendering
    if render:
        env = gym.make('LunarLander-v2', render_mode='human')
    else:
        env = gym.make('LunarLander-v2')

    print("\n" + "="*70)
    print("PLAYING TRAINED CONTROLLER")
    print("="*70 + "\n")

    total_reward = 0
    landings = 0

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"Episode {episode + 1}/{num_episodes}")

        while not done:
            # Get action from controller
            action = controller.get_action(state)

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            state = next_state

            # Small delay for visualization
            if render:
                time.sleep(delay)

        # Check outcome
        if terminated and reward > 0:
            outcome = "LANDED! ✓"
            landings += 1
        elif terminated and reward < 0:
            outcome = "CRASHED ✗"
        else:
            outcome = "TIMEOUT"

        total_reward += episode_reward

        print(f"  Reward: {episode_reward:7.2f} | Steps: {episode_length:3d} | {outcome}\n")

    # Summary
    avg_reward = total_reward / num_episodes
    landing_rate = landings / num_episodes * 100

    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Episodes:        {num_episodes}")
    print(f"Average Reward:  {avg_reward:.2f}")
    print(f"Landing Rate:    {landing_rate:.1f}% ({landings}/{num_episodes})")
    print("="*70 + "\n")

    env.close()


def evaluate_controller(controller, num_episodes=100):
    """
    Evaluate controller without rendering (faster).

    Args:
        controller: Controller to evaluate
        num_episodes: Number of episodes
    """
    print(f"\nEvaluating controller over {num_episodes} episodes (no rendering)...")

    evaluator = GAEvaluator(num_episodes=num_episodes, render=False)
    fitness, metrics = evaluator.evaluate_controller(controller, verbose=False)
    evaluator.close()

    print("\n" + "="*70)
    print(f"EVALUATION RESULTS ({num_episodes} episodes)")
    print("="*70)
    print(f"Fitness:          {fitness:.2f}")
    print(f"Average Reward:   {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Landing Rate:     {metrics['landing_rate']*100:.1f}%")
    print(f"  Landings:       {metrics['landing_count']}")
    print(f"  Crashes:        {metrics['crash_count']}")
    print(f"  Timeouts:       {metrics['timeout_count']}")
    print(f"Average Length:   {metrics['avg_length']:.1f} steps")
    print(f"Reward Range:     [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Play trained GA controller")
    parser.add_argument('--model', type=str, default='ga_models/ga_best.npy',
                        help='Path to trained controller (.npy file)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to play')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster)')
    parser.add_argument('--delay', type=float, default=0.02,
                        help='Delay between frames (seconds)')
    parser.add_argument('--evaluate', type=int, default=0,
                        help='Run evaluation over N episodes (no rendering)')
    parser.add_argument('--describe', action='store_true',
                        help='Show controller parameter analysis')

    args = parser.parse_args()

    # Load controller
    print(f"Loading controller from {args.model}...")
    controller = LinearController()
    controller.load(args.model)

    # Show description if requested
    if args.describe:
        controller.describe()

    # Evaluate if requested
    if args.evaluate > 0:
        evaluate_controller(controller, num_episodes=args.evaluate)

    # Play episodes
    if not args.no_render or args.evaluate == 0:
        play_controller(
            controller,
            num_episodes=args.episodes,
            render=not args.no_render,
            delay=args.delay
        )


if __name__ == "__main__":
    main()
