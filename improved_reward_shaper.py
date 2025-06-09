import numpy as np


class ImprovedRewardShaper:
    """
    Potential-based reward shaper with small continuous velocity and angle penalties.
    Guarantees policy invariance for the main shaping term.
    """

    def __init__(self, gamma=0.99, vel_scale=0.01, ang_scale=0.01):
        self.gamma = gamma
        self.vel_scale = vel_scale
        self.ang_scale = ang_scale
        self.prev_phi = None
        self.low_velocity_steps = 0

    def reset(self):
        """
        Call at the start of each episode to clear history.
        """
        self.prev_phi = None
        self.low_velocity_steps = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Args:
            state: array-like of length ≥5 [x, y, vel_x, vel_y, angle, ...]
            action: int (not used for shaping here)
            reward: float, original env reward r(s,a,s')
            done: bool, whether episode terminated (not used)
            step: int, time step in episode (not used)
            terminated: bool, whether episode ended due to crash/success
            truncated: bool, whether episode ended due to timeout
        Returns:
            float: shaped reward r + F + penalties
        """
        # 1) Base environment reward
        r = reward

        # 2) Potential-based shaping term
        #    Φ(s) = -distance to pad center
        x, y = state[0], state[1]
        dist = np.hypot(x, y)
        phi = -dist

        if self.prev_phi is None:
            F = 0.0
        else:
            F = self.gamma * phi - self.prev_phi

        # 3) Hovering penalty - detect when agent is moving slowly for too long
        vel_x, vel_y = state[2], state[3]
        speed = np.hypot(vel_x, vel_y)
        
        # Count steps with low velocity (hovering)
        if speed < 0.1:  # Very low velocity threshold
            self.low_velocity_steps += 1
        else:
            self.low_velocity_steps = 0  # Reset counter if moving
            
        # Apply increasing penalty for extended hovering
        hover_penalty = 0.0
        if self.low_velocity_steps > 50:  # After 50 steps of hovering
            hover_penalty = -0.5 * (self.low_velocity_steps - 50)  # Increasing penalty
            
        vel_pen = hover_penalty
        ang_pen = 0.0

        # 4) Apply terminal bonuses/penalties
        terminal_bonus = 0.0
        if done and terminated and reward > 0:
            # Successful landing bonus + quick landing bonus
            base_bonus = 200.0
            # Bonus for landing quickly (fewer steps = more bonus)
            quick_bonus = max(0, (500 - step) * 0.5)  # Up to +250 for very quick landings
            terminal_bonus = base_bonus + quick_bonus
        elif done and truncated and not terminated:
            # Timeout penalty - worse than crash penalty
            terminal_bonus = -50.0
        elif done and terminated and reward < 0:
            # Crash penalty (LunarLander gives -100 for crash)
            terminal_bonus = -25.0

        # 5) Combine and update
        shaped = r + F + vel_pen + ang_pen + terminal_bonus
        self.prev_phi = phi

        return shaped
