import numpy as np


class TargetedRewardShaper:
    """
    Reward shaper specifically designed to fix the "learned to avoid crashing 
    but never land" problem. Focuses on the key missing behaviors.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.prev_state = None
        self.hover_count = 0
        self.descent_momentum = 0

    def reset(self):
        self.prev_phi = None
        self.prev_state = None
        self.hover_count = 0
        self.descent_momentum = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Targeted shaping to encourage actual landing behavior
        """
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle, ang_vel = state[4], state[5]
        leg1, leg2 = state[6], state[7]

        distance = np.hypot(x, y)
        speed = np.hypot(vx, vy)

        # =================================================================
        # 1. POTENTIAL-BASED SHAPING: Guide toward successful landing zone
        # =================================================================

        # Core potential: distance to pad + height penalty
        phi = -distance * 3.0 - max(0, y) * 2.0

        # Bonus for being in the "success zone" (close + low + slow)
        if distance < 0.5 and y < 0.3:
            phi += 10.0  # Strong bonus for being in landing zone

        # Extra bonus for proper orientation in landing zone
        if distance < 0.5 and y < 0.3 and abs(angle) < 0.2:
            phi += 5.0

        # Apply potential-based shaping
        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            shaped_reward += potential_bonus
        self.prev_phi = phi

        # =================================================================
        # 2. DESCENT ENCOURAGEMENT: Reward moving toward the ground
        # =================================================================

        descent_bonus = 0.0

        # Reward active descent (negative vy)
        if vy < -0.1:  # Moving downward
            descent_bonus += 2.0
            self.descent_momentum += 1
        elif vy > 0.1:  # Moving upward (bad!)
            descent_bonus -= 1.0
            self.descent_momentum = 0
        else:
            self.descent_momentum *= 0.9  # Decay momentum

        # Bonus for sustained descent
        if self.descent_momentum > 10:
            descent_bonus += 1.0

        # =================================================================
        # 3. GROUND INTERACTION REWARD: Encourage getting close to ground
        # =================================================================

        ground_bonus = 0.0

        # Big bonus for getting very close to ground
        if y < 0.2:
            ground_bonus += 5.0
        elif y < 0.5:
            ground_bonus += 2.0

        # Bonus for touching ground with legs (even if not perfect landing)
        if leg1 or leg2:
            ground_bonus += 3.0

        if leg1 and leg2:
            ground_bonus += 5.0  # Both legs touching

        # =================================================================
        # 4. ANTI-HOVER SYSTEM: Punish staying in one place
        # =================================================================

        hover_penalty = 0.0

        # Detect hovering (low speed + not on ground)
        if speed < 0.2 and y > 0.1:
            self.hover_count += 1
            if self.hover_count > 20:  # After 20 steps of hovering
                hover_penalty = -0.5 * (self.hover_count - 20)
        else:
            self.hover_count = 0

        # =================================================================
        # 5. ACTION ENCOURAGEMENT: Fix action bias problems
        # =================================================================

        action_bonus = 0.0

        # Encourage using main engine when high up (overcome "do nothing" bias)
        if action == 2 and y > 1.0:  # Main engine when high
            action_bonus += 0.5

        # Encourage using side engines for control when close to pad
        if distance > 0.3 and (action == 1 or action == 3):  # Side engines
            action_bonus += 0.3

        # Small penalty for "do nothing" when clearly action is needed
        if action == 0 and (y > 1.5 or abs(x) > 1.0):
            action_bonus -= 0.2

        # =================================================================
        # 6. TERMINAL BONUSES: Encourage actually completing the task
        # =================================================================

        terminal_bonus = 0.0

        if done:
            if terminated:
                if reward > 0:
                    # Successful landing - moderate bonus to preserve env signal
                    terminal_bonus = 30.0
                else:
                    # Crashed - small additional penalty
                    terminal_bonus = -5.0
            else:
                # Timeout - strong penalty for indecisiveness
                terminal_bonus = -20.0

        # =================================================================
        # 7. COMBINE ALL COMPONENTS
        # =================================================================

        total_shaping = (descent_bonus + ground_bonus + hover_penalty +
                         action_bonus + terminal_bonus)

        shaped_reward += total_shaping

        # Update state tracking
        self.prev_state = state.copy()

        return shaped_reward


class ProgressiveRewardShaper:
    """
    Reward shaper that adapts its strategy based on training progress.
    Starts with basic guidance, adds complexity as agent improves.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.episode_count = 0
        self.phase = "exploration"  # exploration -> descent -> landing

    def reset(self):
        self.prev_phi = None
        self.episode_count += 1

        # Update training phase based on episode count
        if self.episode_count < 500:
            self.phase = "exploration"
        elif self.episode_count < 1500:
            self.phase = "descent"
        else:
            self.phase = "landing"

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """Progressive shaping that evolves with training"""

        shaped_reward = reward
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]

        distance = np.hypot(x, y)

        # PHASE 1: Exploration (first 500 episodes)
        # Focus: Get agent to explore different areas, avoid immediate crashes
        if self.phase == "exploration":
            # Simple potential to encourage movement toward pad
            phi = -distance * 1.0

            if self.prev_phi is not None:
                shaped_reward += self.gamma * phi - self.prev_phi
            self.prev_phi = phi

            # Encourage action diversity
            if action != 0:  # Not "do nothing"
                shaped_reward += 0.1

            # Mild timeout penalty
            if done and not terminated:
                shaped_reward -= 5.0

        # PHASE 2: Descent (episodes 500-1500)
        # Focus: Teach agent to descend toward pad
        elif self.phase == "descent":
            # Stronger potential including height
            phi = -distance * 2.0 - max(0, y) * 1.0

            if self.prev_phi is not None:
                shaped_reward += self.gamma * phi - self.prev_phi
            self.prev_phi = phi

            # Reward downward movement
            if vy < -0.1:
                shaped_reward += 1.0

            # Penalize hovering
            if abs(vx) < 0.1 and abs(vy) < 0.1 and y > 0.2:
                shaped_reward -= 0.3

            # Stronger timeout penalty
            if done and not terminated:
                shaped_reward -= 10.0

        # PHASE 3: Landing (episodes 1500+)
        # Focus: Perfect the landing technique
        else:
            # Full potential with landing zone bonus
            phi = -distance * 3.0 - max(0, y) * 2.0
            if distance < 0.5 and y < 0.3:
                phi += 8.0

            if self.prev_phi is not None:
                shaped_reward += self.gamma * phi - self.prev_phi
            self.prev_phi = phi

            # Landing technique rewards
            if y < 0.5:
                speed = np.hypot(vx, vy)
                if speed < 0.5:  # Slow approach
                    shaped_reward += 2.0
                elif speed > 1.0:  # Too fast
                    shaped_reward -= 2.0

            # Ground contact bonus
            leg1, leg2 = state[6], state[7]
            if leg1 or leg2:
                shaped_reward += 2.0

            # Terminal bonuses
            if done:
                if terminated and reward > 0:
                    shaped_reward += 20.0  # Success bonus
                elif not terminated:
                    shaped_reward -= 15.0  # Timeout penalty

        return shaped_reward

# Factory function to easily switch between shapers


def create_reward_shaper(shaper_type="targeted"):
    """Create the appropriate reward shaper"""

    if shaper_type == "targeted":
        return TargetedRewardShaper()
    elif shaper_type == "progressive":
        return ProgressiveRewardShaper()
    elif shaper_type == "minimal":
        return MinimalRewardShaper()
    elif shaper_type == "none":
        return NoRewardShaper()
    else:
        raise ValueError(f"Unknown shaper type: {shaper_type}")

# Import the minimal and no-shaping classes from previous artifact


class MinimalRewardShaper:
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None

    def reset(self):
        self.prev_phi = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward
        x, y = state[0], state[1]
        distance = np.hypot(x, y)
        phi = -distance * 1.0 - max(0, y - 0.1) * 0.5

        if self.prev_phi is not None:
            potential_shaping = self.gamma * phi - self.prev_phi
            shaped_reward += potential_shaping
        self.prev_phi = phi

        if done and not terminated:
            shaped_reward -= 10.0

        return shaped_reward


class NoRewardShaper:
    def __init__(self, gamma=0.99):
        pass

    def reset(self):
        pass

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        return reward
