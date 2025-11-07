"""
Linear Controller for Lunar Lander

A simple interpretable controller that uses a linear mapping from state to action.
State space: 8 dimensions (position, velocity, angle, leg contacts)
Action space: 4 discrete actions (nothing, left engine, main engine, right engine)

Total parameters: 8 states × 4 actions + 4 biases = 36 parameters
"""

import numpy as np


class LinearController:
    """
    Linear policy for Lunar Lander.

    Action selection: action = argmax(state @ weights + bias)
    where weights is an 8x4 matrix and bias is a 4-element vector.
    """

    def __init__(self, weights=None, bias=None):
        """
        Initialize linear controller.

        Args:
            weights: 8x4 weight matrix (or flattened 32-element array)
            bias: 4-element bias vector
        """
        if weights is None:
            # Initialize with small random values
            self.weights = np.random.randn(8, 4) * 0.1
        else:
            if len(weights) == 32:
                # Reshape from flattened form
                self.weights = np.array(weights).reshape(8, 4)
            else:
                self.weights = np.array(weights)

        if bias is None:
            self.bias = np.zeros(4)
        else:
            self.bias = np.array(bias)

    def get_action(self, state):
        """
        Select action based on current state.

        Args:
            state: 8-element numpy array or list

        Returns:
            action: integer in [0, 3]
        """
        state = np.array(state)
        # Compute action scores
        action_scores = state @ self.weights + self.bias
        # Select action with highest score
        action = np.argmax(action_scores)
        return int(action)

    def get_parameters(self):
        """
        Get all parameters as a single flat array.

        Returns:
            36-element array: [weights (32), bias (4)]
        """
        weights_flat = self.weights.flatten()
        return np.concatenate([weights_flat, self.bias])

    def set_parameters(self, params):
        """
        Set all parameters from a flat array.

        Args:
            params: 36-element array [weights (32), bias (4)]
        """
        params = np.array(params)
        self.weights = params[:32].reshape(8, 4)
        self.bias = params[32:]

    def get_num_parameters(self):
        """Return total number of parameters."""
        return 36

    def save(self, filepath):
        """Save controller parameters to file."""
        params = self.get_parameters()
        np.save(filepath, params)
        print(f"Controller saved to {filepath}")

    def load(self, filepath):
        """Load controller parameters from file."""
        params = np.load(filepath)
        self.set_parameters(params)
        print(f"Controller loaded from {filepath}")

    def describe(self):
        """
        Print a human-readable description of the controller.

        This helps interpret what the controller has learned.
        """
        state_names = [
            'X position',
            'Y position (altitude)',
            'Horizontal velocity',
            'Vertical velocity',
            'Angle',
            'Angular velocity',
            'Left leg contact',
            'Right leg contact'
        ]

        action_names = [
            'Do nothing',
            'Fire left engine',
            'Fire main engine',
            'Fire right engine'
        ]

        print("\n" + "="*70)
        print("LINEAR CONTROLLER ANALYSIS")
        print("="*70)

        print("\nBias values (baseline action preferences):")
        for i, action_name in enumerate(action_names):
            print(f"  {action_name:20s}: {self.bias[i]:+.3f}")

        print("\nWeight matrix (how each state influences each action):")
        print(f"{'State':<25} {'Nothing':>10} {'Left':>10} {'Main':>10} {'Right':>10}")
        print("-" * 70)

        for i, state_name in enumerate(state_names):
            print(f"{state_name:<25}", end='')
            for j in range(4):
                print(f"{self.weights[i, j]:>10.3f}", end='')
            print()

        print("\nMost influential weights:")
        # Find top 10 absolute weights
        flat_weights = self.weights.flatten()
        abs_weights = np.abs(flat_weights)
        top_indices = np.argsort(abs_weights)[-10:][::-1]

        for idx in top_indices:
            state_idx = idx // 4
            action_idx = idx % 4
            weight_value = flat_weights[idx]
            print(f"  {state_names[state_idx]:25s} → {action_names[action_idx]:20s}: {weight_value:+.3f}")

        print("="*70 + "\n")


if __name__ == "__main__":
    # Test the controller
    print("Testing LinearController...")

    # Create a controller
    controller = LinearController()

    # Test with a sample state
    sample_state = [0.0, 0.5, 0.1, -0.3, 0.0, 0.0, 0.0, 0.0]
    action = controller.get_action(sample_state)
    print(f"Sample state: {sample_state}")
    print(f"Selected action: {action}")

    # Test parameter get/set
    params = controller.get_parameters()
    print(f"\nTotal parameters: {len(params)}")

    # Test save/load
    controller.save("test_controller.npy")
    controller2 = LinearController()
    controller2.load("test_controller.npy")

    # Verify they produce the same action
    action2 = controller2.get_action(sample_state)
    assert action == action2, "Save/load failed!"
    print("Save/load test passed!")

    # Show description
    controller.describe()
