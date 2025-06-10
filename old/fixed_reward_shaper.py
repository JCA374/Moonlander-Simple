import numpy as np

from improved_reward_shaper import ImprovedRewardShaper


class FixedRewardShaper:
    """
    Conservative reward shaper that preserves original environment signals
    while providing gentle guidance. Key principle: never overwhelm original rewards.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.hover_steps = 0
        self.last_distance = None

    def reset(self):
        """Call at the start of each episode"""
        self.prev_phi = None
        self.hover_steps = 0
        self.last_distance = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Conservative shaping that preserves environment reward structure
        
        Key changes from original:
        1. Much smaller shaping magnitudes (max ±20 vs ±150)
        2. Preserve terminal reward dominance
        3. Focus on guidance, not overwhelming the signal
        """
        # Start with original reward - this is the foundation
        shaped_reward = reward

        # Extract state variables
        x, y = state[0], state[1]
        vel_x, vel_y = state[2], state[3]
        angle, ang_vel = state[4], state[5]
        leg1, leg2 = state[6], state[7]

        # Basic metrics
        distance_to_pad = np.hypot(x, y)
        speed = np.hypot(vel_x, vel_y)

        # 1. GENTLE potential-based shaping (max impact: ±5 per step)
        # Encourage being closer to pad and lower altitude
        phi = -distance_to_pad * 2.0 - max(0, y) * 1.0

        # Add bonus for good orientation when close
        if distance_to_pad < 0.5:
            phi += (1.0 - abs(angle)) * 2.0  # Reward being upright

        # Calculate potential difference
        if self.prev_phi is not None:
            potential_shaping = self.gamma * phi - self.prev_phi
            shaped_reward += potential_shaping
        self.prev_phi = phi

        # 2. GENTLE behavior shaping (max ±3 per step)
        behavior_bonus = 0.0

        # Reward controlled descent
        if vel_y < -0.1 and abs(angle) < 0.3:
            behavior_bonus += 1.0

        # Reward being positioned over pad
        if abs(x) < 0.4:
            behavior_bonus += 0.5

        # Small penalty for excessive speed near ground
        if y < 0.5 and speed > 1.0:
            behavior_bonus -= 2.0

        # 3. ANTI-HOVERING (max -5 per step, only when clearly hovering)
        hover_penalty = 0.0
        if distance_to_pad < 0.8 and y > 0.2:  # Close to pad but still high
            if speed < 0.15 and abs(vel_y) < 0.1:  # Nearly stationary
                self.hover_steps += 1
                if self.hover_steps > 30:  # Allow some hovering for control
                    hover_penalty = -min(5.0, (self.hover_steps - 30) * 0.2)
            else:
                self.hover_steps = 0
        else:
            self.hover_steps = 0

        # 4. PRESERVE terminal rewards (this is crucial!)
        terminal_bonus = 0.0
        if done:
            if terminated:
                if reward > 0:
                    # SUCCESS: Add modest bonus, don't overwhelm the +100-300 from environment
                    terminal_bonus = 20.0  # Small compared to environment's +100-300
                else:
                    # CRASH: Environment already gives -100, add small extra penalty
                    terminal_bonus = -10.0
            else:
                # TIMEOUT: Moderate penalty to encourage decisive action
                terminal_bonus = -15.0

        # 5. Combine all components
        total_shaping = behavior_bonus + hover_penalty + terminal_bonus
        shaped_reward += total_shaping

        # 6. SAFETY CLAMP: Ensure we never completely reverse the reward signal
        # If original reward was strongly positive/negative, preserve that
        if abs(reward) > 50:  # Strong environment signal
            # Limit shaping to ±30% of original magnitude
            max_shaping = abs(reward) * 0.3
            if abs(total_shaping) > max_shaping:
                total_shaping = np.sign(total_shaping) * max_shaping
                shaped_reward = reward + total_shaping

        return shaped_reward


class MinimalRewardShaper:
    """
    Even more conservative shaper - just potential-based guidance
    Use this if FixedRewardShaper still causes issues
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None

    def reset(self):
        self.prev_phi = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """Minimal shaping - just potential-based guidance toward pad"""
        shaped_reward = reward

        # Very simple potential: encourage being close to pad center and low
        x, y = state[0], state[1]
        distance = np.hypot(x, y)
        phi = -distance * 1.0 - max(0, y - 0.1) * 0.5

        if self.prev_phi is not None:
            potential_shaping = self.gamma * phi - self.prev_phi
            shaped_reward += potential_shaping
        self.prev_phi = phi

        # Only add terminal bonus for timeouts
        if done and not terminated:  # Timeout
            shaped_reward -= 10.0

        return shaped_reward


class NoRewardShaper:
    """
    No shaping at all - for testing if shaping is the problem
    """

    def __init__(self, gamma=0.99):
        pass

    def reset(self):
        pass

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        return reward  # Return original reward unchanged

# Test function to compare shapers


def test_reward_shapers():
    """Test different shaping approaches on key scenarios"""

    # Test scenarios: [x, y, vx, vy, angle, ang_vel, leg1, leg2]
    scenarios = [
        ("Successful Landing", [0.0, 0.05, -0.05, -
         0.1, 0.0, 0.0, 1, 1], 200, True, False),
        ("Crash Landing", [0.0, 0.05, -0.3, -
         1.5, 0.5, 0.0, 0, 0], -100, True, False),
        ("Timeout", [1.5, 2.0, 0.0, 0.0, 0.0, 0.0, 0, 0], 0, False, True),
        ("Good Approach", [0.1, 0.3, -0.1, -0.2,
         0.1, 0.0, 0, 0], -0.5, False, False),
        ("Hovering", [0.2, 0.8, 0.01, 0.01,
         0.0, 0.0, 0, 0], -0.3, False, False),
    ]

    shapers = {
        "Original": ImprovedRewardShaper(),
        "Fixed": FixedRewardShaper(),
        "Minimal": MinimalRewardShaper(),
        "None": NoRewardShaper()
    }

    print("Reward Shaper Comparison:")
    print("=" * 60)

    for scenario_name, state, reward, terminated, truncated in scenarios:
        print(f"\n{scenario_name}:")
        print(f"  Environment reward: {reward}")

        for shaper_name, shaper in shapers.items():
            shaper.reset()
            shaped = shaper.shape_reward(
                state, 0, reward, terminated or truncated, 100, terminated, truncated)
            difference = shaped - reward
            print(f"  {shaper_name:12s}: {shaped:7.1f} (Δ{difference:+6.1f})")


if __name__ == "__main__":
    test_reward_shapers()
