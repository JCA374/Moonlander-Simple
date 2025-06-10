import numpy as np


class ImprovedRewardShaper:
    """
    Balanced reward shaper that encourages landing without overwhelming penalties.
    Focus on positive reinforcement rather than excessive punishment.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.low_velocity_steps = 0
        self.near_pad_steps = 0
        self.descent_bonus_given = False

    def reset(self):
        """
        Call at the start of each episode to clear history.
        """
        self.prev_phi = None
        self.low_velocity_steps = 0
        self.near_pad_steps = 0
        self.descent_bonus_given = False

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Args:
            state: array-like [x, y, vel_x, vel_y, angle, ang_vel, leg1, leg2]
            action: int (0=nothing, 1=left engine, 2=main engine, 3=right engine)
            reward: float, original env reward
            done: bool, whether episode terminated
            step: int, time step in episode
            terminated: bool, whether episode ended due to crash/success
            truncated: bool, whether episode ended due to timeout
        Returns:
            float: shaped reward
        """
        # 1) Start with base environment reward
        r = reward

        # 2) Potential-based shaping - encourage moving toward pad center
        x, y = state[0], state[1]
        vel_x, vel_y = state[2], state[3]

        # Distance to landing pad
        dist = np.hypot(x, y)

        # Potential function: reward being close to pad and being low
        # but only if moving reasonably (not hovering)
        speed = np.hypot(vel_x, vel_y)

        # Basic potential
        phi = -dist * 2.0  # Doubled weight on position

        # Add height component to encourage descent
        if y > 0:
            phi -= y * 1.5  # Encourage being lower

        # Bonus for being aligned over the pad when low
        if abs(x) < 0.4 and y < 0.5:
            phi += 5.0  # Nice bonus for being well-positioned

        # Calculate potential-based shaping
        if self.prev_phi is None:
            F = 0.0
        else:
            F = self.gamma * phi - self.prev_phi

        # 3) Stronger hovering discouragement and velocity control
        hover_penalty = 0.0
        velocity_penalty = 0.0  # Initialize here

        # Penalize hovering near pad
        if dist < 0.5 and y > 0.1:  # Close to pad but still high
            if speed < 0.1 and vel_y > -0.1:  # Not moving down
                self.low_velocity_steps += 1
                if self.low_velocity_steps > 20:  # Reduced tolerance
                    hover_penalty = -0.5 * \
                        (self.low_velocity_steps - 20)  # Stronger penalty
            else:
                self.low_velocity_steps = 0

        # Penalize excessive velocity when close to ground
        if y < 0.5:  # Close to ground
            if speed > 1.0:  # Moving too fast
                # Strong penalty for high speed near ground
                velocity_penalty = -2.0 * (speed - 1.0)

        # 4) Positive reinforcement for good behavior
        behavior_bonus = 0.0

        # Reward controlled descent
        if vel_y < -0.1 and abs(state[4]) < 0.2:  # Descending with good angle
            behavior_bonus += 0.5

        # Reward being over the pad
        if abs(x) < 0.4:
            behavior_bonus += 0.3

        # Extra bonus for slow, controlled approach when very close
        if y < 0.3 and dist < 0.4 and speed < 0.5:
            behavior_bonus += 2.0  # Doubled reward for careful final approach

        # Reward slowing down when approaching ground
        if y < 0.5 and speed < 0.3:
            behavior_bonus += 1.5  # Big bonus for being slow near ground

        # Small bonus for efficient fuel use (not using engine when not needed)
        if action == 0 and y > 1.0:  # Coasting when high
            behavior_bonus += 0.1

        # 5) Terminal bonuses/penalties (balanced)
        terminal_bonus = 0.0

        if done:
            if terminated and reward > 0:
                # Successful landing! (by game's definition)
                base_bonus = 150.0  # Increased bonus

                # Speed bonus for quick landing
                if step < 200:
                    quick_bonus = (200 - step) * 0.5  # Up to +100
                else:
                    quick_bonus = 0

                # Clean landing bonus - based on final velocity, not just legs
                clean_bonus = 0.0
                if speed < 0.5:  # Very soft landing
                    clean_bonus = 100.0
                elif speed < 1.0:  # Acceptable landing
                    clean_bonus = 50.0

                terminal_bonus = base_bonus + quick_bonus + clean_bonus

            elif truncated:
                # Timeout - strong penalty to encourage decisive action
                terminal_bonus = -100.0

            elif terminated and reward < 0:
                # Crash - original game already gives -100
                terminal_bonus = 0.0  # Don't double-penalize

        # 6) Combine everything
        shaped = r + F + hover_penalty + velocity_penalty + behavior_bonus + terminal_bonus

        # Update state
        self.prev_phi = phi

        return shaped
