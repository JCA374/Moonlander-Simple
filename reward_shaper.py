import numpy as np


class RewardShaper:
    """
    Simple potential-based shaper for LunarLander-v2:
     - Encourages moving toward pad center and downward
     - Small step cost to discourage hovering
     - Graduated speed penalties near ground
     - Clear terminal bonuses
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None

    def reset(self, init_state=None):
        """Reset shaper, optionally with initial state to avoid step-0 jump"""
        if init_state is not None:
            x, y, _, _, angle = init_state[:5]
            altitude = max(0.0, y)
            # Initialize with potential of starting state
            self.prev_phi = -1.0 * abs(x) - 1.5 * altitude - 0.5 * abs(angle)
        else:
            self.prev_phi = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        state: [x, y, vx, vy, angle, ang_vel, left_contact, right_contact]
        """
        x, y, vx, vy, angle = state[:5]
        altitude = max(0.0, y)
        speed = np.hypot(vx, vy)

        # 1) Potential-based shaping: guide toward (x=0, y=0) and upright
        phi = -1.0 * abs(x) - 1.5 * altitude - 0.5 * abs(angle)

        if self.prev_phi is not None:
            delta = self.gamma * phi - self.prev_phi
            shaping = np.clip(delta, -3.0, 3.0)  # Prevent dramatic jumps
        else:
            shaping = 0.0
        self.prev_phi = phi

        # 2) Small living penalty
        living_penalty = -0.05  # Slightly higher to encourage decisive action

        # 3) Graduated speed control based on altitude
        speed_penalty = 0.0
        if altitude < 0.5:  # Near ground
            if altitude < 0.2 and speed > 0.3:  # Very close and too fast
                speed_penalty = -5.0 * (speed - 0.3)
            elif altitude < 0.4 and speed > 0.5:  # Close and too fast
                speed_penalty = -2.0 * (speed - 0.5)

        # 4) Good approach bonus (replaces need for complex terminal logic)
        approach_bonus = 0.0
        if altitude < 0.3 and abs(x) < 0.4 and speed < 0.4:
            approach_bonus = 2.0  # Reward being in good landing configuration

        shaped = reward + shaping + living_penalty + speed_penalty + approach_bonus

        # 5) Terminal rewards (only if original game didn't already give big reward)
        if done:
            if terminated:
                if reward > 0:  # Successful landing
                    # Bonus based on landing quality
                    if speed < 0.3:
                        shaped += 50.0  # Gentle landing bonus
                    else:
                        shaped += 20.0  # Rough landing bonus
                else:  # Crash
                    shaped -= 20.0  # Additional crash penalty
            else:  # Timeout
                shaped -= 50.0  # Stronger timeout penalty

        return shaped
