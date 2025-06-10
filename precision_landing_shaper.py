import numpy as np


class PrecisionLandingShaper:
    """
    Focused reward shaper to fix the specific "lands outside pad" problem.
    Addresses the core issues: horizontal guidance and landing completion.
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.prev_distance = None
        self.hover_count = 0
        self.ground_contact_bonus_given = False

    def reset(self):
        self.prev_phi = None
        self.prev_distance = None
        self.hover_count = 0
        self.ground_contact_bonus_given = False

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Precision landing shaper focused on the specific problems identified:
        1. Poor horizontal control (X-position)
        2. Speed control near ground  
        3. Action bias (encourage side engines)
        """
        shaped_reward = reward

        # Extract state
        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]
        leg1, leg2 = state[6], state[7]

        # Core metrics
        # Only horizontal distance matters for landing pad
        distance_to_pad = abs(x)
        altitude = max(0, y)
        speed = np.hypot(vx, vy)
        horizontal_speed = abs(vx)

        # =================================================================
        # 1. PRECISION HORIZONTAL GUIDANCE - Fix the main problem
        # =================================================================

        # Strong potential-based shaping for X-position
        # This creates a "funnel" effect toward the pad center
        horizontal_potential = -distance_to_pad * 5.0  # Strong horizontal guidance

        # Extra bonus for being very close to center
        if distance_to_pad < 0.3:
            horizontal_potential += 10.0  # Big bonus for being centered
        elif distance_to_pad < 0.1:
            horizontal_potential += 20.0  # Huge bonus for perfect positioning

        # =================================================================
        # 2. ACTION GUIDANCE - Fix the action bias
        # =================================================================

        action_bonus = 0.0

        # Encourage side engines when off-center (fix underuse of side engines)
        if distance_to_pad > 0.2:
            if action == 1 and x > 0:  # Left engine when too far right
                action_bonus += 2.0
            elif action == 3 and x < 0:  # Right engine when too far left
                action_bonus += 2.0

        # Discourage main engine when well-positioned and low
        if distance_to_pad < 0.3 and y < 0.5 and action == 2:
            action_bonus -= 1.0  # Don't blast main engine when positioned well

        # Reward "do nothing" when perfectly positioned
        if distance_to_pad < 0.1 and speed < 0.3 and action == 0:
            action_bonus += 1.0

        # =================================================================
        # 3. SPEED CONTROL - Fix the "too fast" problem
        # =================================================================

        speed_bonus = 0.0

        # Height-dependent speed limits
        if y < 0.3:  # Very close to ground
            target_speed = 0.3
            if speed <= target_speed:
                speed_bonus += 5.0  # Big reward for being slow near ground
            elif speed > target_speed * 2:
                speed_bonus -= 8.0  # Strong penalty for being too fast
        elif y < 0.6:  # Moderately close
            target_speed = 0.6
            if speed <= target_speed:
                speed_bonus += 2.0
            elif speed > target_speed * 1.5:
                speed_bonus -= 3.0

        # Special bonus for very gentle horizontal approach
        if y < 0.4 and horizontal_speed < 0.2:
            speed_bonus += 3.0

        # =================================================================
        # 4. GROUND CONTACT ENCOURAGEMENT - Get agent to actually land
        # =================================================================

        ground_bonus = 0.0

        # Progressive rewards for getting closer to successful landing
        if leg1 or leg2:  # Any leg contact
            ground_bonus += 5.0
            if not self.ground_contact_bonus_given:
                ground_bonus += 10.0  # First contact bonus
                self.ground_contact_bonus_given = True

        if leg1 and leg2:  # Both legs
            ground_bonus += 10.0

        # =================================================================
        # 5. ANTI-HOVER SYSTEM - Force decisive action
        # =================================================================

        hover_penalty = 0.0

        # Detect hovering: low speed while not on ground
        if speed < 0.15 and y > 0.15:
            self.hover_count += 1
            if self.hover_count > 25:
                hover_penalty = -1.0 * (self.hover_count - 25)
        else:
            self.hover_count = 0

        # =================================================================
        # 6. APPROACH PROGRESS TRACKING
        # =================================================================

        progress_bonus = 0.0

        # Reward improving horizontal position
        if self.prev_distance is not None:
            if distance_to_pad < self.prev_distance:  # Getting closer
                progress_bonus += 1.0
            elif distance_to_pad > self.prev_distance:  # Getting farther
                progress_bonus -= 0.5

        self.prev_distance = distance_to_pad

        # =================================================================
        # 7. TERMINAL REWARDS - Preserve but enhance environment signals
        # =================================================================

        terminal_bonus = 0.0

        if done:
            if terminated:
                if reward > 0:
                    # Successful landing - moderate bonus
                    terminal_bonus = 30.0
                    # Extra bonus for precision
                    if distance_to_pad < 0.1:
                        terminal_bonus += 20.0  # Precision landing bonus
                else:
                    # Crash - analyze the failure
                    if distance_to_pad > 0.5:
                        terminal_bonus = -15.0  # Penalty for missing pad
                    elif speed > 0.8:
                        terminal_bonus = -10.0  # Penalty for speed
                    else:
                        terminal_bonus = -5.0   # Other crash
            else:
                # Timeout - moderate penalty
                terminal_bonus = -12.0

        # =================================================================
        # 8. COMBINE ALL COMPONENTS
        # =================================================================

        # Apply potential-based shaping
        phi = horizontal_potential
        if self.prev_phi is not None:
            potential_shaping = self.gamma * phi - self.prev_phi
            shaped_reward += potential_shaping
        self.prev_phi = phi

        # Add all other bonuses
        total_bonus = (action_bonus + speed_bonus + ground_bonus +
                       hover_penalty + progress_bonus + terminal_bonus)
        shaped_reward += total_bonus

        return shaped_reward


class DiagnosticShaper:
    """
    Minimal shaper with detailed logging to understand what's happening
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.step_count = 0

    def reset(self):
        self.prev_phi = None
        self.step_count = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        self.step_count += 1

        # Very simple shaping - just guide toward pad
        x, y = state[0], state[1]
        distance = abs(x)

        phi = -distance * 2.0

        if self.prev_phi is not None:
            shaping = self.gamma * phi - self.prev_phi
        else:
            shaping = 0

        self.prev_phi = phi

        shaped_reward = reward + shaping

        # Log key moments
        if done:
            leg1, leg2 = state[6], state[7]
            speed = np.hypot(state[2], state[3])
            print(f"  Episode end: X={x:.3f}, Speed={speed:.3f}, Legs={leg1}{leg2}, "
                  f"Original={reward:.1f}, Shaped={shaped_reward:.1f}")

        return shaped_reward


def create_precision_shaper(shaper_type="precision"):
    """Factory function for precision landing shapers"""
    if shaper_type == "precision":
        return PrecisionLandingShaper()
    elif shaper_type == "diagnostic":
        return DiagnosticShaper()
    else:
        raise ValueError(f"Unknown shaper: {shaper_type}")


# Test the shaper on your typical failure scenarios
if __name__ == "__main__":
    print("Testing Precision Landing Shaper on typical failure scenarios:")
    print("=" * 60)

    shaper = PrecisionLandingShaper()

    # Scenario 1: Landing outside pad (your main problem)
    print("\n1. Landing outside pad (X=1.0, typical failure):")
    shaper.reset()
    state = [1.0, 0.1, -0.2, -0.3, 0.1, 0.0, 1, 1]  # Outside pad
    shaped = shaper.shape_reward(state, 0, -100, True, 200, True, False)
    print(f"   Original: -100, Shaped: {shaped:.1f}")

    # Scenario 2: Perfect landing (what we want)
    print("\n2. Perfect landing (X=0.0, centered):")
    shaper.reset()
    state = [0.0, 0.05, -0.05, -0.1, 0.0, 0.0, 1, 1]  # Centered
    shaped = shaper.shape_reward(state, 0, 200, True, 180, True, False)
    print(f"   Original: +200, Shaped: {shaped:.1f}")

    # Scenario 3: Too fast approach
    print("\n3. Too fast approach (your speed problem):")
    shaper.reset()
    state = [0.1, 0.2, -0.3, -0.8, 0.2, 0.0, 0, 0]  # Fast approach
    shaped = shaper.shape_reward(state, 2, -0.5, False, 150, False, False)
    print(f"   Original: -0.5, Shaped: {shaped:.1f}")

    print("\nThe shaper should:")
    print("- Heavily penalize landing outside pad (scenario 1)")
    print("- Reward perfect centered landings (scenario 2)")
    print("- Penalize excessive speed when low (scenario 3)")
