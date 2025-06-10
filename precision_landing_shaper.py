import numpy as np


class GentleLandingShaper:
    """
    Reward shaper specifically designed to fix the "coming in too fast" problem.
    Focus: Teach the agent that SLOW and CONTROLLED is better than FAST.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.prev_speed = None
        self.prev_altitude = None
        self.speed_violations = 0
        self.good_approach_steps = 0
        self.dive_bomb_penalty_applied = False

    def reset(self):
        self.prev_phi = None
        self.prev_speed = None
        self.prev_altitude = None
        self.speed_violations = 0
        self.good_approach_steps = 0
        self.dive_bomb_penalty_applied = False

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Aggressive speed control to prevent dive bombing and encourage gentle landings
        """
        shaped_reward = reward

        # Extract state variables
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]

        distance_to_pad = abs(x)
        altitude = max(0, y)
        speed = np.hypot(vx, vy)
        vertical_speed = abs(vy)

        # =================================================================
        # 1. AGGRESSIVE SPEED CONTROL - Core fix for dive bombing
        # =================================================================

        speed_control_bonus = 0.0

        # Altitude-based speed limits (MUCH stricter than before)
        if altitude < 0.3:  # Very close to ground
            max_safe_speed = 0.25  # VERY slow required
            if speed <= max_safe_speed:
                speed_control_bonus += 8.0  # HUGE reward for being slow
            elif speed > max_safe_speed * 2:  # Way too fast
                speed_control_bonus -= 15.0  # MASSIVE penalty
                self.speed_violations += 1
        elif altitude < 0.6:  # Moderately close
            max_safe_speed = 0.4
            if speed <= max_safe_speed:
                speed_control_bonus += 4.0
            elif speed > max_safe_speed * 1.5:
                speed_control_bonus -= 8.0
        elif altitude < 1.0:  # Getting close
            max_safe_speed = 0.6
            if speed <= max_safe_speed:
                speed_control_bonus += 2.0
            elif speed > max_safe_speed * 1.5:
                speed_control_bonus -= 4.0

        # Track speed reduction (reward slowing down)
        if self.prev_speed is not None:
            speed_change = speed - self.prev_speed
            if speed_change < -0.03:  # Slowing down significantly
                speed_control_bonus += 3.0
            elif speed_change > 0.03 and altitude < 1.0:  # Speeding up when low
                speed_control_bonus -= 3.0

        self.prev_speed = speed

        # =================================================================
        # 2. DIVE BOMB DETECTION AND PREVENTION
        # =================================================================

        dive_bomb_penalty = 0.0

        # Detect dangerous dive bombing pattern
        if altitude < 0.5 and vertical_speed > 0.8:  # High downward speed when low
            dive_bomb_penalty = -20.0  # SEVERE penalty
            self.dive_bomb_penalty_applied = True

        # Detect "free fall" pattern (no engine control)
        if altitude < 1.0 and vertical_speed > 0.6 and action == 0:  # Doing nothing while falling fast
            dive_bomb_penalty -= 10.0

        # =================================================================
        # 3. GENTLE APPROACH REWARDS
        # =================================================================

        gentle_approach_bonus = 0.0

        # Perfect gentle approach criteria
        is_gentle_approach = (
            altitude < 0.8 and
            speed < 0.5 and
            abs(angle) < 0.3 and
            distance_to_pad < 0.4
        )

        if is_gentle_approach:
            gentle_approach_bonus += 5.0
            self.good_approach_steps += 1

            # Sustained gentle approach bonus
            if self.good_approach_steps > 10:
                gentle_approach_bonus += 2.0  # Extra for sustained good behavior
        else:
            self.good_approach_steps = 0

        # Reward controlled vertical descent (not too fast, not too slow)
        if altitude < 0.8:
            if -0.4 < vy < -0.1:  # Gentle downward motion
                gentle_approach_bonus += 3.0
            elif vy < -0.6:  # Too fast downward
                gentle_approach_bonus -= 5.0
            elif vy > 0.1:  # Going upward (wasteful)
                gentle_approach_bonus -= 2.0

        # =================================================================
        # 4. ENGINE USAGE GUIDANCE
        # =================================================================

        engine_bonus = 0.0

        # Encourage main engine for deceleration when moving too fast
        if action == 2 and speed > 0.6 and altitude < 1.0:
            engine_bonus += 2.0  # Good use of main engine for braking

        # Discourage main engine when already slow enough
        if action == 2 and speed < 0.3 and altitude < 0.5:
            engine_bonus -= 3.0  # Don't over-thrust when slow

        # Encourage "do nothing" when speed and position are good
        if action == 0 and speed < 0.4 and distance_to_pad < 0.3 and altitude < 0.8:
            engine_bonus += 1.5  # Good coasting

        # =================================================================
        # 5. BASIC POSITIONAL GUIDANCE (GENTLE)
        # =================================================================

        # Simple potential-based guidance (much gentler than before)
        phi = -distance_to_pad * 1.5 - altitude * 0.5

        # Bonus for being in the landing zone
        if distance_to_pad < 0.3 and altitude < 0.4:
            phi += 5.0

        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            shaped_reward += potential_bonus
        self.prev_phi = phi

        # =================================================================
        # 6. TERMINAL REWARDS - PRIORITIZE GENTLE TECHNIQUE
        # =================================================================

        terminal_bonus = 0.0

        if done:
            if terminated:
                if reward > 0:
                    # Success! Give bonus based on gentleness
                    if speed < 0.25:
                        terminal_bonus = 100.0  # HUGE bonus for very gentle landing
                    elif speed < 0.4:
                        terminal_bonus = 50.0   # Good bonus for gentle landing
                    else:
                        terminal_bonus = 20.0   # Small bonus for rough but successful landing
                else:
                    # Crash - analyze why and penalize accordingly
                    if speed > 1.0:
                        terminal_bonus = -30.0  # Big penalty for speed-related crash
                    elif distance_to_pad > 0.5:
                        terminal_bonus = -15.0  # Penalty for missing pad
                    else:
                        terminal_bonus = -5.0   # Small penalty for other issues
            else:
                # Timeout - moderate penalty
                terminal_bonus = -10.0

        # =================================================================
        # 7. COMBINE ALL COMPONENTS
        # =================================================================

        total_bonus = (speed_control_bonus + dive_bomb_penalty + gentle_approach_bonus +
                       engine_bonus + terminal_bonus)

        shaped_reward += total_bonus

        # Debug logging for critical moments (optional - remove if too verbose)
        if done and terminated:
            print(f"    Landing attempt: Speed={speed:.3f}, Altitude={altitude:.3f}, "
                  f"Distance={distance_to_pad:.3f}, Success={reward > 0}")

        return shaped_reward


class UltraConservativeShaper:
    """
    Even more conservative approach - minimize all speed-related risks
    Use this if the GentleLandingShaper still allows too much speed
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.consecutive_slow_steps = 0

    def reset(self):
        self.prev_phi = None
        self.consecutive_slow_steps = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]

        speed = np.hypot(vx, vy)
        distance_to_pad = abs(x)
        altitude = max(0, y)

        # EXTREME speed control
        speed_bonus = 0.0

        # ANY speed above limits gets penalized harshly
        if altitude < 0.5:
            if speed < 0.2:
                speed_bonus = 10.0  # Massive reward for being very slow
                self.consecutive_slow_steps += 1
            elif speed > 0.3:
                speed_bonus = -25.0  # Massive penalty for any significant speed
        elif altitude < 1.0:
            if speed < 0.4:
                speed_bonus = 5.0
                self.consecutive_slow_steps += 1
            elif speed > 0.6:
                speed_bonus = -15.0
        else:
            self.consecutive_slow_steps = 0

        # Bonus for sustained slow approach
        if self.consecutive_slow_steps > 20:
            speed_bonus += 5.0

        # Simple guidance
        phi = -distance_to_pad * 2.0 - altitude * 1.0
        if self.prev_phi is not None:
            shaped_reward += self.gamma * phi - self.prev_phi
        self.prev_phi = phi

        # Terminal: ONLY reward ultra-gentle landings
        if done and terminated and reward > 0:
            if speed < 0.15:
                shaped_reward += 200.0  # Enormous bonus for ultra-gentle
            elif speed < 0.3:
                shaped_reward += 50.0
            else:
                shaped_reward += 10.0  # Minimal bonus for rough landing

        shaped_reward += speed_bonus
        return shaped_reward


def create_gentle_shaper(shaper_type="gentle"):
    """Factory function for gentle landing shapers"""
    if shaper_type == "gentle":
        return GentleLandingShaper()
    elif shaper_type == "ultra_conservative":
        return UltraConservativeShaper()
    else:
        raise ValueError(f"Unknown shaper: {shaper_type}")
