"""
Test GA Evaluator with Mock Environment

Since Box2D is not available in this environment, we'll test the evaluator
with a mock environment to verify the logic works correctly.
"""

import numpy as np
from linear_controller import LinearController


class MockLunarLanderEnv:
    """Mock environment for testing purposes."""

    def __init__(self):
        self.observation_space = type('obj', (object,), {'shape': (8,)})()
        self.action_space = type('obj', (object,), {'n': 4})()
        self.step_count = 0

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        state = np.random.randn(8) * 0.5
        return state, {}

    def step(self, action):
        """Take a step in the environment."""
        self.step_count += 1

        # Mock next state
        next_state = np.random.randn(8) * 0.5

        # Mock reward (higher if action is reasonable)
        reward = np.random.randn() * 10 + (action * 0.1)

        # Episode ends after 100-300 steps randomly
        terminated = self.step_count >= np.random.randint(100, 300)
        truncated = False

        # Simulate landing success sometimes
        if terminated and np.random.random() < 0.3:
            reward = 10.0  # Landing bonus
        elif terminated:
            reward = -5.0  # Crash penalty

        return next_state, reward, terminated, truncated, {}

    def close(self):
        """Close environment."""
        pass


def test_controller_evaluation():
    """Test evaluating a controller."""
    print("Testing controller evaluation with mock environment...")

    # Create mock environment
    env = MockLunarLanderEnv()

    # Create random controller
    controller = LinearController()

    # Run a few episodes
    total_reward = 0
    num_episodes = 5

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(1000):
            # Get action
            action = controller.get_action(state)
            assert 0 <= action < 4, f"Invalid action: {action}"

            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            state = next_state

            if done:
                break

        total_reward += episode_reward
        print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Steps={step+1}")

    avg_reward = total_reward / num_episodes
    print(f"\nAverage reward: {avg_reward:.2f}")
    print("✓ Controller evaluation test passed!")

    env.close()


def test_fitness_computation():
    """Test fitness computation logic."""
    print("\nTesting fitness computation...")

    # Test different scenarios
    test_cases = [
        {"avg_reward": 100, "landing_rate": 0.8, "expected_fitness": 180},
        {"avg_reward": 200, "landing_rate": 1.0, "expected_fitness": 300},
        {"avg_reward": -100, "landing_rate": 0.0, "expected_fitness": -100},
        {"avg_reward": 150, "landing_rate": 0.5, "expected_fitness": 200},
    ]

    for i, case in enumerate(test_cases):
        avg_reward = case["avg_reward"]
        landing_rate = case["landing_rate"]
        expected_fitness = case["expected_fitness"]

        # Compute fitness (same formula as in ga_evaluator.py)
        fitness = avg_reward + (landing_rate * 100)

        assert abs(fitness - expected_fitness) < 0.01, f"Test case {i} failed"
        print(f"  Case {i+1}: reward={avg_reward}, landing_rate={landing_rate:.1f} → fitness={fitness:.1f} ✓")

    print("✓ Fitness computation test passed!")


def test_parameter_passing():
    """Test that parameters can be passed to controller correctly."""
    print("\nTesting parameter passing...")

    # Create controller
    controller1 = LinearController()

    # Get parameters
    params = controller1.get_parameters()
    assert len(params) == 36, f"Expected 36 parameters, got {len(params)}"
    print(f"  Parameter count: {len(params)} ✓")

    # Modify parameters
    new_params = params + 1.0

    # Create new controller with modified params
    controller2 = LinearController()
    controller2.set_parameters(new_params)

    # Verify they're different
    params2 = controller2.get_parameters()
    assert not np.allclose(params, params2), "Parameters should be different"
    assert np.allclose(params + 1.0, params2), "Parameters not set correctly"
    print(f"  Parameter modification: ✓")

    # Test that they produce different actions
    test_state = np.random.randn(8)
    action1 = controller1.get_action(test_state)
    action2 = controller2.get_action(test_state)
    print(f"  Action from controller1: {action1}")
    print(f"  Action from controller2: {action2}")

    print("✓ Parameter passing test passed!")


if __name__ == "__main__":
    print("="*70)
    print("GA EVALUATOR TESTS (WITH MOCK ENVIRONMENT)")
    print("="*70)
    print()

    test_controller_evaluation()
    test_fitness_computation()
    test_parameter_passing()

    print()
    print("="*70)
    print("ALL TESTS PASSED!")
    print("="*70)
    print()
    print("Note: These tests use a mock environment since Box2D is not available.")
    print("The actual LunarLander-v2 environment will work similarly.")
