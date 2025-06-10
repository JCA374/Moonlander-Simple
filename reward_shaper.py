import numpy as np


class RewardShaper:
    """
    Minimal reward shaper that adds gentle guidance without overwhelming the original reward signal.
    Designed to prevent the engine cutoff problem while maintaining reasonable scores.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None

    def reset(self):
        self.prev_phi = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        # Start with original reward - this is the foundation
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]
        leg1, leg2 = state[6], state[7]

        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # =============================================================
        # 1. MINIMAL ENGINE SAFETY (fix the cutoff problem)
        # =============================================================

        # Only intervene in critical situations
        falling_fast = vy < -0.6  # Very fast descent
        very_low = altitude < 0.2  # Very close to ground
        not_landed = not (leg1 or leg2)

        # Small penalty for turning off engines when crashing
        if action == 0 and falling_fast and very_low and not_landed:
            shaped_reward -= 1.0  # Gentle nudge, not harsh penalty

        # Small reward for using engines to prevent crash
        if action == 2 and falling_fast and very_low and not_landed:
            shaped_reward += 0.5  # Slight encouragement

        # =============================================================
        # 2. BASIC POTENTIAL-BASED SHAPING
        # =============================================================

        # Simple potential function: closer to pad and lower is better
        phi = -distance_to_pad * 0.5 - altitude * 0.5

        # Add small bonus for being in good landing position
        if distance_to_pad < 0.3 and altitude < 0.3:
            phi += 1.0

        # Apply potential-based shaping
        if self.prev_phi is not None:
            potential_diff = self.gamma * phi - self.prev_phi
            # Clip to prevent huge swings
            potential_diff = np.clip(potential_diff, -2.0, 2.0)
            shaped_reward += potential_diff

        self.prev_phi = phi

        # =============================================================
        # 3. GENTLE SPEED GUIDANCE
        # =============================================================

        # Only apply when very close to ground
        if altitude < 0.2 and speed > 0.8:
            shaped_reward -= 0.5  # Gentle penalty for being too fast
        elif altitude < 0.2 and speed < 0.4:
            shaped_reward += 0.5  # Gentle reward for good speed

        # =============================================================
        # 4. SMALL TERMINAL BONUSES
        # =============================================================

        if done:
            if terminated and reward > 0:
                # Small bonus for successful landing
                if speed < 0.4:
                    shaped_reward += 5.0  # Gentle landing bonus
                else:
                    shaped_reward += 2.0  # Any successful landing
            elif terminated and reward < 0:
                # Check if it was an engine cutoff crash
                if falling_fast and action == 0:
                    shaped_reward -= 2.0  # Learn not to cut engines when falling
            elif not terminated:
                # Timeout - small penalty
                shaped_reward -= 1.0

        # =============================================================
        # 5. CLAMP TOTAL SHAPING
        # =============================================================

        # Ensure shaping doesn't overwhelm original reward
        # Limit shaping to be at most ±10 per step (except terminals)
        if not done:
            shaping_amount = shaped_reward - reward
            if abs(shaping_amount) > 10:
                shaped_reward = reward + np.sign(shaping_amount) * 10

        return shaped_reward


# Test function
def test_reward_shaper():
    """Test that rewards stay reasonable"""

    shaper = RewardShaper()
    shaper.reset()

    test_cases = [
        # (state, action, original_reward, done, description)
        ([0.0, 0.2, 0.0, -0.7, 0.0, 0.0, 0, 0], 0, -
         0.5, False, "Falling fast, engines off"),
        ([0.0, 0.2, 0.0, -0.7, 0.0, 0.0, 0, 0],
         2, -0.5, False, "Falling fast, engines on"),
        ([0.0, 1.5, 0.0, -0.3, 0.0, 0.0, 0, 0],
         0, -0.3, False, "High altitude, coasting"),
        ([0.0, 0.1, 0.0, -0.2, 0.0, 0.0, 1, 1],
         0, 100.0, True, "Successful landing"),
        ([1.0, 0.1, 0.0, -0.8, 0.5, 0.0, 0, 0], 0, -100.0, True, "Crash landing"),
    ]

    print("Testing reward shaper...")
    print("=" * 60)

    for state, action, orig_reward, done, desc in test_cases:
        shaped = shaper.shape_reward(
            state, action, orig_reward, done, 100, terminated=done)
        shaping = shaped - orig_reward

        print(f"{desc}:")
        print(f"  Original: {orig_reward:>7.1f}")
        print(f"  Shaped:   {shaped:>7.1f}")
        print(f"  Shaping:  {shaping:>7.1f}")
        print()

        # Reset for terminal states
        if done:
            shaper.reset()


if __name__ == "__main__":
    test_reward_shaper()
