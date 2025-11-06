"""
Compare GA and DQN Approaches

Loads trained models from both approaches and compares their performance.

Usage:
    python compare_ga_dqn.py --ga-model ga_models/ga_best.npy --dqn-model moonlander_best.pth --episodes 100
"""

import argparse
import numpy as np
import torch
import gymnasium as gym

from linear_controller import LinearController
from ga_evaluator import GAEvaluator
from dqn_agent import DQNAgent


def evaluate_ga_controller(model_path, num_episodes=100):
    """Evaluate GA controller."""
    print(f"\nEvaluating GA controller: {model_path}")

    controller = LinearController()
    controller.load(model_path)

    evaluator = GAEvaluator(num_episodes=num_episodes, render=False)
    fitness, metrics = evaluator.evaluate_controller(controller, verbose=False)
    evaluator.close()

    return {
        'type': 'GA',
        'model': model_path,
        'fitness': fitness,
        'avg_reward': metrics['avg_reward'],
        'std_reward': metrics['std_reward'],
        'landing_rate': metrics['landing_rate'],
        'landing_count': metrics['landing_count'],
        'crash_count': metrics['crash_count'],
        'timeout_count': metrics['timeout_count'],
        'avg_length': metrics['avg_length'],
        'min_reward': metrics['min_reward'],
        'max_reward': metrics['max_reward'],
        'num_parameters': 36
    }


def evaluate_dqn_agent(model_path, num_episodes=100):
    """Evaluate DQN agent."""
    print(f"\nEvaluating DQN agent: {model_path}")

    # Load DQN agent
    env = gym.make('LunarLander-v2')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    agent.q_network.load_state_dict(torch.load(model_path))
    agent.q_network.eval()

    # Count parameters
    num_params = sum(p.numel() for p in agent.q_network.parameters())

    # Evaluate
    total_reward = 0
    landing_count = 0
    crash_count = 0
    timeout_count = 0
    episode_rewards = []
    episode_lengths = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0

        for step in range(1000):
            # Get action from DQN (greedy)
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = agent.q_network(state_tensor)
                action = q_values.argmax().item()

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            state = next_state

            if done:
                if terminated and reward > 0:
                    landing_count += 1
                elif terminated and reward < 0:
                    crash_count += 1
                else:
                    timeout_count += 1
                break

        total_reward += episode_reward
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    env.close()

    # Compute metrics
    avg_reward = total_reward / num_episodes
    landing_rate = landing_count / num_episodes
    std_reward = np.std(episode_rewards)

    return {
        'type': 'DQN',
        'model': model_path,
        'fitness': avg_reward + (landing_rate * 100),
        'avg_reward': avg_reward,
        'std_reward': std_reward,
        'landing_rate': landing_rate,
        'landing_count': landing_count,
        'crash_count': crash_count,
        'timeout_count': timeout_count,
        'avg_length': np.mean(episode_lengths),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'num_parameters': num_params
    }


def print_comparison(results_list):
    """Print comparison table."""
    print("\n" + "="*90)
    print("MODEL COMPARISON")
    print("="*90)

    # Header
    print(f"{'Model':<15} {'Type':<6} {'Params':<8} {'Fitness':<10} {'Avg Reward':<12} "
          f"{'Landing %':<12} {'Avg Steps':<10}")
    print("-"*90)

    # Results
    for res in results_list:
        print(f"{res['type']:<15} {res['type']:<6} {res['num_parameters']:<8} "
              f"{res['fitness']:<10.2f} {res['avg_reward']:<12.2f} "
              f"{res['landing_rate']*100:<12.1f} {res['avg_length']:<10.1f}")

    print("="*90)

    # Detailed comparison
    print("\nDETAILED RESULTS:")
    print("-"*90)

    for i, res in enumerate(results_list):
        print(f"\n{i+1}. {res['type']} Model ({res['model']})")
        print(f"   Parameters:       {res['num_parameters']:,}")
        print(f"   Fitness:          {res['fitness']:.2f}")
        print(f"   Average Reward:   {res['avg_reward']:.2f} ± {res['std_reward']:.2f}")
        print(f"   Landing Rate:     {res['landing_rate']*100:.1f}% ({res['landing_count']} landings)")
        print(f"   Crash Rate:       {res['crash_count']} crashes")
        print(f"   Timeout Rate:     {res['timeout_count']} timeouts")
        print(f"   Average Length:   {res['avg_length']:.1f} steps")
        print(f"   Reward Range:     [{res['min_reward']:.2f}, {res['max_reward']:.2f}]")

    # Winner analysis
    print("\n" + "="*90)
    print("WINNER ANALYSIS")
    print("="*90)

    best_fitness = max(res['fitness'] for res in results_list)
    best_reward = max(res['avg_reward'] for res in results_list)
    best_landing = max(res['landing_rate'] for res in results_list)
    fewest_params = min(res['num_parameters'] for res in results_list)

    for res in results_list:
        print(f"\n{res['type']}:")
        wins = []
        if res['fitness'] == best_fitness:
            wins.append("Best Fitness")
        if res['avg_reward'] == best_reward:
            wins.append("Best Reward")
        if res['landing_rate'] == best_landing:
            wins.append("Best Landing Rate")
        if res['num_parameters'] == fewest_params:
            wins.append("Fewest Parameters")

        if wins:
            print(f"  Wins: {', '.join(wins)}")
        else:
            print("  No category wins")

    print("="*90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare GA and DQN models")
    parser.add_argument('--ga-model', type=str, default='ga_models/ga_best.npy',
                        help='Path to GA model (.npy)')
    parser.add_argument('--dqn-model', type=str, default='moonlander_best.pth',
                        help='Path to DQN model (.pth)')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes for evaluation')
    parser.add_argument('--ga-only', action='store_true',
                        help='Only evaluate GA model')
    parser.add_argument('--dqn-only', action='store_true',
                        help='Only evaluate DQN model')

    args = parser.parse_args()

    results = []

    # Evaluate GA
    if not args.dqn_only:
        try:
            ga_results = evaluate_ga_controller(args.ga_model, args.episodes)
            results.append(ga_results)
        except Exception as e:
            print(f"Error evaluating GA model: {e}")

    # Evaluate DQN
    if not args.ga_only:
        try:
            dqn_results = evaluate_dqn_agent(args.dqn_model, args.episodes)
            results.append(dqn_results)
        except Exception as e:
            print(f"Error evaluating DQN model: {e}")

    # Print comparison
    if len(results) > 0:
        print_comparison(results)
    else:
        print("No models were successfully evaluated.")


if __name__ == "__main__":
    main()
