"""
Fitness Evaluation for Genetic Algorithm

Evaluates controllers by running them in the Lunar Lander environment
and computing fitness metrics.
"""

import numpy as np
import gymnasium as gym
from typing import Tuple, Dict


class GAEvaluator:
    """
    Evaluates controller fitness in the Lunar Lander environment.
    """

    def __init__(self, num_episodes=10, max_steps=1000, render=False):
        """
        Initialize evaluator.

        Args:
            num_episodes: Number of episodes to average over for fitness
            max_steps: Maximum steps per episode
            render: Whether to render the environment (for visualization)
        """
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.render = render

        # Create environment
        if render:
            self.env = gym.make('LunarLander-v2', render_mode='human')
        else:
            self.env = gym.make('LunarLander-v2')

    def evaluate_controller(self, controller, verbose=False) -> Tuple[float, Dict]:
        """
        Evaluate a controller's fitness.

        Args:
            controller: Controller with get_action(state) method
            verbose: Print detailed episode info

        Returns:
            fitness: Single fitness value (higher is better)
            metrics: Dictionary with detailed metrics
        """
        total_reward = 0.0
        landing_count = 0
        crash_count = 0
        timeout_count = 0
        episode_rewards = []
        episode_lengths = []

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0

            for step in range(self.max_steps):
                # Get action from controller
                action = controller.get_action(state)

                # Take step in environment
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1
                state = next_state

                if done:
                    # Classify episode outcome
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

            if verbose:
                outcome = "LANDED" if (terminated and reward > 0) else "CRASHED" if (terminated and reward < 0) else "TIMEOUT"
                print(f"  Episode {episode+1}/{self.num_episodes}: Reward={episode_reward:.2f}, Steps={episode_length}, Outcome={outcome}")

        # Compute metrics
        avg_reward = total_reward / self.num_episodes
        landing_rate = landing_count / self.num_episodes
        crash_rate = crash_count / self.num_episodes
        timeout_rate = timeout_count / self.num_episodes

        metrics = {
            'avg_reward': avg_reward,
            'landing_rate': landing_rate,
            'landing_count': landing_count,
            'crash_rate': crash_rate,
            'crash_count': crash_count,
            'timeout_rate': timeout_rate,
            'timeout_count': timeout_count,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'avg_length': np.mean(episode_lengths)
        }

        # Compute composite fitness
        # We want to maximize both reward and landing rate
        # Landing rate is scaled by 100 to make it comparable to reward
        fitness = avg_reward + (landing_rate * 100)

        # Alternative fitness functions (can experiment):
        # fitness = avg_reward  # Reward only
        # fitness = landing_rate * 200 + avg_reward * 0.5  # Prioritize landing
        # fitness = avg_reward * (1 + landing_rate)  # Multiplicative bonus

        return fitness, metrics

    def evaluate_parameters(self, parameters, controller_class, verbose=False):
        """
        Evaluate a parameter vector directly.

        Useful for CMA-ES which works with parameter vectors.

        Args:
            parameters: Parameter array for controller
            controller_class: Controller class to instantiate
            verbose: Print detailed info

        Returns:
            fitness: Fitness value
            metrics: Detailed metrics
        """
        # Create controller from parameters
        controller = controller_class()
        controller.set_parameters(parameters)

        # Evaluate
        return self.evaluate_controller(controller, verbose=verbose)

    def close(self):
        """Close the environment."""
        self.env.close()


def parallel_evaluate(parameters_list, controller_class, num_episodes=10, num_workers=4):
    """
    Evaluate multiple parameter sets in parallel.

    Args:
        parameters_list: List of parameter arrays
        controller_class: Controller class
        num_episodes: Episodes per evaluation
        num_workers: Number of parallel workers

    Returns:
        fitness_list: List of fitness values
        metrics_list: List of metric dictionaries
    """
    from multiprocessing import Pool
    import functools

    def evaluate_single(params):
        evaluator = GAEvaluator(num_episodes=num_episodes, render=False)
        fitness, metrics = evaluator.evaluate_parameters(params, controller_class, verbose=False)
        evaluator.close()
        return fitness, metrics

    # Use multiprocessing pool for parallel evaluation
    with Pool(num_workers) as pool:
        results = pool.map(evaluate_single, parameters_list)

    fitness_list = [r[0] for r in results]
    metrics_list = [r[1] for r in results]

    return fitness_list, metrics_list


if __name__ == "__main__":
    # Test the evaluator
    from linear_controller import LinearController

    print("Testing GAEvaluator...")

    # Create a random controller
    controller = LinearController()

    # Create evaluator
    evaluator = GAEvaluator(num_episodes=5, render=False)

    # Evaluate
    print("\nEvaluating random controller over 5 episodes...")
    fitness, metrics = evaluator.evaluate_controller(controller, verbose=True)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Fitness:          {fitness:.2f}")
    print(f"Average Reward:   {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Landing Rate:     {metrics['landing_rate']*100:.1f}%")
    print(f"Crash Rate:       {metrics['crash_rate']*100:.1f}%")
    print(f"Timeout Rate:     {metrics['timeout_rate']*100:.1f}%")
    print(f"Average Length:   {metrics['avg_length']:.1f} steps")
    print(f"Reward Range:     [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
    print("="*60)

    evaluator.close()

    # Test parallel evaluation
    print("\nTesting parallel evaluation with 4 random controllers...")
    params_list = [LinearController().get_parameters() for _ in range(4)]
    fitness_list, metrics_list = parallel_evaluate(params_list, LinearController, num_episodes=3, num_workers=2)

    print("\nParallel evaluation results:")
    for i, (fit, met) in enumerate(zip(fitness_list, metrics_list)):
        print(f"  Controller {i+1}: Fitness={fit:.2f}, Landing Rate={met['landing_rate']*100:.1f}%")
