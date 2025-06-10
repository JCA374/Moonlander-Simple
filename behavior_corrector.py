import numpy as np


class BehaviorCorrectorShaper:
    """
    Targeted fix for identified problems: main engine overuse and speed control
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.main_engine_streak = 0
        self.speed_violations = 0

    def reset(self):
        self.prev_phi = None
        self.main_engine_streak = 0
        self.speed_violations = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]

        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # =============================================================
        # FIX 1: AGGRESSIVE MAIN ENGINE CORRECTION
        # =============================================================

        engine_penalty = 0.0

        # Track main engine overuse
        if action == 2:  # Main engine
            self.main_engine_streak += 1
        else:
            self.main_engine_streak = 0

        # Penalize excessive main engine use
        if self.main_engine_streak > 3:
            engine_penalty -= 2.0 * (self.main_engine_streak - 3)

        # STRONG penalty for main engine when slow and low
        if action == 2 and speed < 0.4 and altitude < 0.5:
            engine_penalty -= 5.0  # You're already slow enough!

        # Reward "do nothing" when positioned well and slow
        if action == 0 and speed < 0.4 and distance_to_pad < 0.3:
            engine_penalty += 2.0

        # =============================================================
        # FIX 2: STRICT SPEED CONTROL FOR FINAL APPROACH
        # =============================================================

        speed_control = 0.0

        # Altitude-based speed requirements (STRICT)
        if altitude < 0.3:  # Very close to ground
            if speed > 0.3:
                speed_control -= 10.0  # MASSIVE penalty
                self.speed_violations += 1
            elif speed < 0.2:
                speed_control += 5.0   # Big reward for being gentle

        elif altitude < 0.6:  # Approaching
            if speed > 0.5:
                speed_control -= 5.0
            elif speed < 0.3:
                speed_control += 3.0

        # =============================================================
        # FIX 3: REWARD GENTLE TECHNIQUE
        # =============================================================

        technique_bonus = 0.0

        # Perfect final approach (close, slow, centered)
        if altitude < 0.4 and speed < 0.3 and distance_to_pad < 0.2:
            technique_bonus += 8.0  # HUGE bonus for perfect setup

        # Reward slowing down when approaching
        if altitude < 0.8 and -0.3 < vy < -0.1:  # Controlled descent
            technique_bonus += 2.0

        # =============================================================
        # FIX 4: BASIC GUIDANCE (KEEP EXISTING STRENGTHS)
        # =============================================================

        # Gentle potential-based guidance
        phi = -distance_to_pad * 1.0 - altitude * 0.5

        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            shaped_reward += potential_bonus
        self.prev_phi = phi

        # =============================================================
        # FIX 5: TERMINAL REWARDS FOR GENTLENESS
        # =============================================================

        terminal_bonus = 0.0

        if done and terminated:
            if reward > 0:
                # Success - reward based on gentleness
                if speed < 0.25:
                    terminal_bonus = 100.0  # Perfect gentle landing
                elif speed < 0.4:
                    terminal_bonus = 50.0   # Good landing
                else:
                    terminal_bonus = 20.0   # Rough but successful
            else:
                # Failure analysis
                if speed > 0.6:
                    terminal_bonus = -20.0  # Speed-related crash
                elif distance_to_pad > 0.4:
                    terminal_bonus = -10.0  # Position-related crash
                else:
                    terminal_bonus = -5.0   # Other crash (angle, etc.)
        elif done and not terminated:
            terminal_bonus = -15.0  # Timeout penalty

        # =============================================================
        # COMBINE ALL CORRECTIONS
        # =============================================================

        total_correction = engine_penalty + \
            speed_control + technique_bonus + terminal_bonus
        shaped_reward += total_correction

        return shaped_reward
