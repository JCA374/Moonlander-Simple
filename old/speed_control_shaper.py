import numpy as np


class SpeedControlShaper:
    """
    Reward shaper specifically designed to fix the "dive bombing" problem.
    Focuses on teaching speed control and gentle approaches.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.prev_speed = None
        self.consecutive_main_engine = 0
        self.speed_violations = 0

    def reset(self):
        self.prev_phi = None
        self.prev_speed = None
        self.consecutive_main_engine = 0
        self.speed_violations = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Target the specific problems: too much main engine, too fast approaches
        """
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]

        distance = np.hypot(x, y)
        speed = np.hypot(vx, vy)

        # =================================================================
        # 1. SPEED CONTROL SYSTEM - Core fix for dive bombing
        # =================================================================

        speed_bonus = 0.0

        # Height-dependent speed limits (slower as you get lower)
        if y < 0.5:  # Very close to ground
            safe_speed = 0.3
            if speed <= safe_speed:
                speed_bonus += 3.0  # Big reward for being slow near ground
            elif speed > safe_speed * 2:  # Way too fast
                speed_bonus -= 5.0  # Strong penalty
                self.speed_violations += 1
        elif y < 1.0:  # Moderately close
            safe_speed = 0.6
            if speed <= safe_speed:
                speed_bonus += 1.5
            elif speed > safe_speed * 2:
                speed_bonus -= 2.0

        # Reward slowing down (deceleration)
        if self.prev_speed is not None:
            speed_change = speed - self.prev_speed
            if speed_change < -0.05:  # Slowing down
                speed_bonus += 1.0
            elif speed_change > 0.05 and y < 1.0:  # Speeding up when low
                speed_bonus -= 1.0

        self.prev_speed = speed

        # =================================================================
        # 2. ENGINE USAGE CORRECTION - Fix main engine overuse
        # =================================================================

        engine_bonus = 0.0

        # Track consecutive main engine usage
        if action == 2:  # Main engine
            self.consecutive_main_engine += 1
        else:
            self.consecutive_main_engine = 0

        # Penalize excessive main engine use
        if self.consecutive_main_engine > 5:
            engine_bonus -= 0.5 * (self.consecutive_main_engine - 5)

        # Encourage "do nothing" when speed is already good
        if action == 0 and speed < 0.4 and y < 1.0:
            engine_bonus += 0.5  # Reward coasting when slow

        # Reward side engines for fine control
        if (action == 1 or action == 3) and distance > 0.2:
            engine_bonus += 0.3

        # Penalize main engine when already slow and low
        if action == 2 and speed < 0.3 and y < 0.5:
            engine_bonus -= 1.0  # You're already slow enough!

        # =================================================================
        # 3. GENTLE APPROACH ENCOURAGEMENT
        # =================================================================

        approach_bonus = 0.0

        # Reward controlled vertical descent
        if vy < -0.05 and vy > -0.4:  # Gentle downward motion
            approach_bonus += 1.0
        elif vy < -0.8:  # Too fast downward
            approach_bonus -= 2.0

        # Reward being positioned over the pad with low speed
        if abs(x) < 0.4 and speed < 0.5:
            approach_bonus += 2.0

        # Big bonus for the ideal "landing zone" state
        if (abs(x) < 0.3 and y < 0.3 and speed < 0.4 and
                abs(angle) < 0.2 and vy < 0 and vy > -0.5):
            approach_bonus += 5.0  # Perfect landing approach!

        # =================================================================
        # 4. BASIC POTENTIAL-BASED GUIDANCE
        # =================================================================

        # Simple potential to guide toward pad
        phi = -distance * 2.0 - max(0, y - 0.1) * 1.0

        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            shaped_reward += potential_bonus
        self.prev_phi = phi

        # =================================================================
        # 5. TERMINAL REWARDS - Encourage completion
        # =================================================================

        terminal_bonus = 0.0

        if done:
            if terminated:
                if reward > 0:
                    # Success! Moderate bonus to preserve env signal
                    terminal_bonus = 25.0
                    # Extra bonus for gentle landing
                    if speed < 0.3:
                        terminal_bonus += 15.0  # Gentle landing bonus
                else:
                    # Crash - analyze why
                    if speed > 0.6:
                        terminal_bonus = -10.0  # Crashed due to speed
                    else:
                        terminal_bonus = -2.0   # Other crash reason
            else:
                # Timeout
                terminal_bonus = -15.0

        # =================================================================
        # 6. COMBINE ALL COMPONENTS
        # =================================================================

        total_shaping = speed_bonus + engine_bonus + approach_bonus + terminal_bonus
        shaped_reward += total_shaping

        # Debug info (optional - remove if too verbose)
        # if step % 50 == 0 and not done:  # Print occasionally
        #     print(f"  Step {step}: Speed={speed:.2f}, Y={y:.2f}, "
        #           f"Action={action}, SpeedBonus={speed_bonus:.1f}")

        return shaped_reward


class AntiDiveBombShaper:
    """
    Even more aggressive fix specifically for the dive bombing problem.
    Use this if SpeedControlShaper doesn't work.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.altitude_speed_history = []

    def reset(self):
        self.prev_phi = None
        self.altitude_speed_history = []

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]

        distance = np.hypot(x, y)
        speed = np.hypot(vx, vy)

        # Track altitude-speed relationship
        self.altitude_speed_history.append((y, speed))
        if len(self.altitude_speed_history) > 20:
            self.altitude_speed_history.pop(0)

        # HARSH penalties for dangerous approach patterns
        danger_penalty = 0.0

        # Detect dive bombing: high speed + low altitude
        if y < 0.5 and speed > 0.7:
            danger_penalty -= 10.0  # VERY strong penalty

        # Detect "last second braking" pattern
        if len(self.altitude_speed_history) >= 10:
            recent_speeds = [s for _, s in self.altitude_speed_history[-10:]]
            if max(recent_speeds) > 1.0 and y < 0.3:
                danger_penalty -= 5.0  # Penalty for recent high speed when low

        # MAJOR rewards for good technique
        good_technique_bonus = 0.0

        # Perfect approach: low altitude + low speed + centered
        if y < 0.4 and speed < 0.4 and abs(x) < 0.3:
            good_technique_bonus += 8.0

        # Sustained good approach
        if len(self.altitude_speed_history) >= 5:
            recent_good = all(
                s < 0.6 for _, s in self.altitude_speed_history[-5:])
            if recent_good and y < 0.6:
                good_technique_bonus += 3.0

        # Simple potential guidance
        phi = -distance * 1.5 - max(0, y) * 1.0
        if self.prev_phi is not None:
            shaped_reward += self.gamma * phi - self.prev_phi
        self.prev_phi = phi

        # Terminal analysis
        if done and terminated:
            if reward > 0:
                # Success - big bonus for gentle technique
                if speed < 0.3:
                    shaped_reward += 50.0  # Huge bonus for gentle success
                else:
                    shaped_reward += 20.0  # Smaller bonus for rough success
            else:
                # Failure - analyze if it was due to speed
                if speed > 0.8:
                    shaped_reward -= 20.0  # Big penalty for speed-related crash

        shaped_reward += danger_penalty + good_technique_bonus
        return shaped_reward


def create_speed_shaper(shaper_type="speed_control"):
    """Factory function for speed-focused shapers"""
    if shaper_type == "speed_control":
        return SpeedControlShaper()
    elif shaper_type == "anti_dive_bomb":
        return AntiDiveBombShaper()
    else:
        raise ValueError(f"Unknown shaper: {shaper_type}")
