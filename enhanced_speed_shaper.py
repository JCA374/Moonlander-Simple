import numpy as np


class EnhancedSpeedControlShaper:
    """
    Aggressive speed control fix based on analysis showing:
    - Main engine overuse (49.8%)
    - Too fast landings (50% above safe speed)
    - Good positioning but poor speed control
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.main_engine_streak = 0
        self.speed_violations = 0
        self.prev_speed = None
        self.gentle_steps = 0

    def reset(self):
        self.prev_phi = None
        self.main_engine_streak = 0
        self.speed_violations = 0
        self.prev_speed = None
        self.gentle_steps = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]

        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # =============================================================
        # 1. AGGRESSIVE MAIN ENGINE CORRECTION (Target 49.8% -> 25%)
        # =============================================================

        engine_penalty = 0.0

        # Track consecutive main engine use
        if action == 2:  # Main engine
            self.main_engine_streak += 1
        else:
            self.main_engine_streak = 0

        # SEVERE penalties for main engine overuse
        if self.main_engine_streak > 2:  # Allow only 2 consecutive
            engine_penalty -= 3.0 * (self.main_engine_streak - 2)

        # MASSIVE penalty for main engine when already slow
        if action == 2:
            if speed < 0.4 and altitude < 0.8:
                engine_penalty -= 8.0  # "You're already slow enough!"
            elif speed < 0.6 and altitude < 0.4:
                engine_penalty -= 12.0  # "WAY too much thrust near ground!"

        # HUGE reward for NOT using main engine when positioned well
        if action == 0 and speed < 0.5 and distance_to_pad < 0.3 and altitude < 0.6:
            engine_penalty += 5.0  # "Perfect! Coast to landing"

        # =============================================================
        # 2. ULTRA-STRICT SPEED LIMITS (Target avg speed 0.65 -> 0.35)
        # =============================================================

        speed_control = 0.0

        # Altitude-based speed requirements (MUCH stricter)
        if altitude < 0.2:  # Very close to ground
            max_safe = 0.25
            if speed <= max_safe:
                speed_control += 10.0  # MASSIVE reward
                self.gentle_steps += 1
            elif speed > max_safe * 1.5:  # 0.375+
                speed_control -= 15.0  # MASSIVE penalty
                self.speed_violations += 1

        elif altitude < 0.4:  # Close to ground
            max_safe = 0.35
            if speed <= max_safe:
                speed_control += 6.0
                self.gentle_steps += 1
            elif speed > max_safe * 1.5:  # 0.525+
                speed_control -= 10.0

        elif altitude < 0.6:  # Approaching
            max_safe = 0.45
            if speed <= max_safe:
                speed_control += 3.0
                self.gentle_steps += 1
            elif speed > max_safe * 1.5:  # 0.675+
                speed_control -= 6.0
        else:
            self.gentle_steps = 0

        # Bonus for sustained gentle approach
        if self.gentle_steps > 10:
            speed_control += 2.0

        # Reward speed reduction (deceleration)
        if self.prev_speed is not None:
            speed_change = speed - self.prev_speed
            if speed_change < -0.02 and altitude < 1.0:  # Slowing down
                speed_control += 3.0
            elif speed_change > 0.02 and altitude < 0.8:  # Speeding up when low
                speed_control -= 4.0

        self.prev_speed = speed

        # =============================================================
        # 3. DIVE BOMB PREVENTION (Address high downward velocity)
        # =============================================================

        dive_prevention = 0.0

        # Detect dangerous descent rates
        if altitude < 0.6 and vy < -0.6:  # Fast downward when low
            dive_prevention -= 12.0  # SEVERE penalty

        if altitude < 0.4 and vy < -0.4:  # Moderate descent when very low
            dive_prevention -= 6.0

        # Reward controlled descent
        if altitude < 0.8 and -0.3 < vy < -0.1:  # Perfect descent rate
            dive_prevention += 4.0

        # =============================================================
        # 4. PERFECT LANDING ZONE REWARDS
        # =============================================================

        perfect_approach = 0.0

        # Define "perfect landing zone" - where agent should be rewarded massively
        in_perfect_zone = (
            distance_to_pad < 0.2 and  # Centered over pad
            altitude < 0.4 and         # Low enough
            speed < 0.35 and           # Slow enough
            abs(angle) < 0.2           # Upright enough
        )

        if in_perfect_zone:
            perfect_approach += 8.0  # HUGE bonus for being in the sweet spot

            # Extra bonus for gentle descent in perfect zone
            if -0.2 < vy < -0.05:
                perfect_approach += 5.0

        # =============================================================
        # 5. BASIC GUIDANCE (Keep existing strengths)
        # =============================================================

        # Gentle potential-based guidance
        phi = -distance_to_pad * 1.5 - altitude * 0.8

        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            shaped_reward += potential_bonus
        self.prev_phi = phi

        # =============================================================
        # 6. TERMINAL ANALYSIS - REWARD GENTLENESS MASSIVELY
        # =============================================================

        terminal_bonus = 0.0

        if done:
            if terminated:
                if reward > 0:
                    # Success - tier rewards based on gentleness
                    if speed < 0.25:
                        terminal_bonus = 150.0  # MASSIVE bonus for ultra-gentle
                    elif speed < 0.35:
                        terminal_bonus = 100.0  # Great bonus for gentle
                    elif speed < 0.5:
                        terminal_bonus = 50.0   # Good bonus for acceptable
                    else:
                        terminal_bonus = 10.0   # Small bonus for rough success
                else:
                    # Analyze crash reasons
                    if speed > 0.7:
                        terminal_bonus = -25.0  # Speed-related crash
                    elif speed > 0.5:
                        terminal_bonus = -15.0  # Moderately speed-related
                    else:
                        terminal_bonus = -5.0   # Other crash reason
            else:
                # Timeout
                terminal_bonus = -20.0

        # =============================================================
        # 7. COMBINE ALL COMPONENTS
        # =============================================================

        total_shaping = (engine_penalty + speed_control + dive_prevention +
                         perfect_approach + terminal_bonus)

        shaped_reward += total_shaping


        return shaped_reward


class ConservativeSpeedShaper:
    """
    Even more conservative version if the above doesn't work
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None

    def reset(self):
        self.prev_phi = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]

        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # EXTREME speed penalties
        speed_bonus = 0.0

        if altitude < 0.5:
            if speed < 0.3:
                speed_bonus = 15.0  # MASSIVE reward for being slow
            elif speed > 0.4:
                speed_bonus = -20.0  # MASSIVE penalty for any speed

        # MASSIVE main engine penalties
        if action == 2 and speed < 0.5:
            speed_bonus -= 10.0

        # Basic guidance
        phi = -distance_to_pad * 2.0 - altitude * 1.0
        if self.prev_phi is not None:
            shaped_reward += self.gamma * phi - self.prev_phi
        self.prev_phi = phi

        # Only reward ultra-gentle success
        if done and terminated and reward > 0:
            if speed < 0.2:
                shaped_reward += 200.0
            elif speed > 0.4:
                shaped_reward -= 50.0  # Penalize even successful rough landings

        shaped_reward += speed_bonus
        return shaped_reward
