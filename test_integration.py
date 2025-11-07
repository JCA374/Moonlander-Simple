"""
Integration Test for GA System

Tests the complete GA training pipeline end-to-end with a very short training run.
"""

import os
import numpy as np
from linear_controller import LinearController
from cma_es_optimizer import CMAESOptimizer
from test_ga_components import MockLunarLanderEnv


def test_short_training_run():
    """Run a very short GA training to test the full pipeline."""
    print("="*70)
    print("INTEGRATION TEST: SHORT GA TRAINING RUN")
    print("="*70)
    print()

    # Parameters for short test
    num_generations = 5
    population_size = 10
    episodes_per_eval = 3

    print(f"Settings:")
    print(f"  Generations:      {num_generations}")
    print(f"  Population:       {population_size}")
    print(f"  Episodes/eval:    {episodes_per_eval}")
    print()

    # Initialize controller
    initial_controller = LinearController()
    initial_params = initial_controller.get_parameters()

    # Initialize optimizer
    optimizer = CMAESOptimizer(
        initial_params=initial_params,
        sigma=0.5,
        population_size=population_size
    )

    print(f"Optimizer initialized")
    print(f"  Using CMA-ES: {optimizer.use_cma}")
    print(f"  Population size: {optimizer.population_size}")
    print()

    # Create mock environment
    env = MockLunarLanderEnv()

    # Training loop
    best_fitness = -np.inf
    print("Starting training...")
    print("-"*70)

    for generation in range(num_generations):
        # Get candidates
        candidates = optimizer.ask()

        # Evaluate each candidate
        fitness_list = []
        for params in candidates:
            # Create controller from params
            controller = LinearController()
            controller.set_parameters(params)

            # Evaluate over multiple episodes
            total_reward = 0
            landing_count = 0

            for _ in range(episodes_per_eval):
                state, _ = env.reset()
                episode_reward = 0

                for step in range(200):  # Max 200 steps for test
                    action = controller.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    state = next_state

                    if done:
                        if terminated and reward > 0:
                            landing_count += 1
                        break

                total_reward += episode_reward

            # Compute fitness
            avg_reward = total_reward / episodes_per_eval
            landing_rate = landing_count / episodes_per_eval
            fitness = avg_reward + (landing_rate * 100)
            fitness_list.append(fitness)

        # Update optimizer
        optimizer.tell(fitness_list)

        # Track best
        gen_best = max(fitness_list)
        if gen_best > best_fitness:
            best_fitness = gen_best

        # Print progress
        mean_fitness = np.mean(fitness_list)
        print(f"Gen {generation:2d}: Best={gen_best:7.2f}, Mean={mean_fitness:7.2f}, "
              f"Overall Best={best_fitness:7.2f}")

    print("-"*70)
    print()

    # Get best solution
    best_params, final_best_fitness = optimizer.get_best()
    print(f"Final Results:")
    print(f"  Best fitness: {best_fitness:.2f}")
    if final_best_fitness is not None:
        print(f"  Returned fitness: {final_best_fitness:.2f}")
    print(f"  Parameter count: {len(best_params)}")
    print()

    # Test the best controller
    print("Testing best controller over 10 episodes...")
    best_controller = LinearController()
    best_controller.set_parameters(best_params)

    total_reward = 0
    landing_count = 0

    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(200):
            action = best_controller.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                if terminated and reward > 0:
                    landing_count += 1
                break

        total_reward += episode_reward

    avg_reward = total_reward / 10
    landing_rate = landing_count / 10
    print(f"  Average reward:   {avg_reward:.2f}")
    print(f"  Landing rate:     {landing_rate*100:.1f}%")
    print()

    env.close()

    # Save test model
    test_dir = "test_output"
    os.makedirs(test_dir, exist_ok=True)
    test_model_path = os.path.join(test_dir, "test_ga_model.npy")
    best_controller.save(test_model_path)
    print(f"Test model saved to {test_model_path}")

    # Load and verify
    loaded_controller = LinearController()
    loaded_controller.load(test_model_path)
    loaded_params = loaded_controller.get_parameters()

    assert np.allclose(best_params, loaded_params), "Save/load verification failed!"
    print("✓ Save/load verification passed")
    print()

    # Clean up
    os.remove(test_model_path)
    os.rmdir(test_dir)

    print("="*70)
    print("✓ INTEGRATION TEST PASSED!")
    print("="*70)
    print()
    print("Note: This test uses a mock environment.")
    print("Real training with LunarLander-v2 will require Box2D.")


if __name__ == "__main__":
    test_short_training_run()
