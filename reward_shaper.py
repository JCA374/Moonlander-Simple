import numpy as np


class RewardShaper:
    """
    Fixed reward shaper that prevents premature engine cutoff near ground.
    Key changes:
    - Rewards engine use when needed for safe landing
    - Penalizes "do nothing" when falling too fast
    - Encourages controlled powered descent
    """

    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.prev_phi = None
        self.main_engine_streak = 0
        self.speed_violations = 0
        self.prev_speed = None
        self.gentle_steps = 0
        self.falling_fast_steps = 0

    def reset(self):
        self.prev_phi = None
        self.main_engine_streak = 0
        self.speed_violations = 0
        self.prev_speed = None
        self.gentle_steps = 0
        self.falling_fast_steps = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]
        leg1, leg2 = state[6], state[7]

        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # =============================================================
        # 1. CRITICAL FIX: PREVENT PREMATURE ENGINE CUTOFF
        # =============================================================

        engine_safety_bonus = 0.0

        # Detect dangerous situations where engines are needed
        falling_fast = vy < -0.5  # Falling quickly
        very_low = altitude < 0.3  # Close to ground
        not_on_ground = not (leg1 or leg2)  # Not touching ground

        # PENALIZE doing nothing when falling fast near ground
        if action == 0 and falling_fast and very_low and not_on_ground:
            engine_safety_bonus -= 15.0  # HUGE penalty for turning off engines when needed
            self.falling_fast_steps += 1
        else:
            self.falling_fast_steps = 0

        # REWARD using engines to control descent when needed
        if action == 2 and falling_fast and altitude < 0.5:
            engine_safety_bonus += 5.0  # Good! Using engines to slow down

        # REWARD appropriate engine use for final touchdown
        if action == 2 and altitude < 0.2 and vy < -0.3 and not_on_ground:
            engine_safety_bonus += 8.0  # Excellent! Cushioning the landing

        # =============================================================
        # 2. BALANCED ENGINE CONTROL (Less aggressive than before)
        # =============================================================

        engine_penalty = 0.0

        # Track consecutive main engine use
        if action == 2:
            self.main_engine_streak += 1
        else:
            self.main_engine_streak = 0

        # Moderate penalties for excessive main engine use (reduced from before)
        if self.main_engine_streak > 5:  # Allow more consecutive use
            engine_penalty -= 1.0 * (self.main_engine_streak - 5)

        # Only penalize main engine when truly unnecessary
        if action == 2:
            # If already very slow AND not falling fast
            if speed < 0.3 and vy > -0.2 and altitude > 0.5:
                engine_penalty -= 3.0  # Wasting fuel when coasting would work

        # Reward efficient "do nothing" only when safe
        if action == 0:
            # Only reward coasting when not in danger
            if speed < 0.5 and vy > -0.4 and distance_to_pad < 0.3 and altitude > 0.3:
                engine_penalty += 2.0  # Good efficiency when safe

        # =============================================================
        # 3. ADAPTIVE SPEED CONTROL (Context-aware)
        # =============================================================

        speed_control = 0.0

        # Speed limits that consider vertical velocity
        if altitude < 0.2:  # Very close to ground
            max_safe_speed = 0.35
            max_safe_vy = -0.3  # Don't fall too fast

            if speed <= max_safe_speed and vy > max_safe_vy:
                speed_control += 10.0  # Perfect control
                self.gentle_steps += 1
            elif speed > max_safe_speed * 1.5 or vy < -0.5:
                speed_control -= 10.0  # Too fast!
                self.speed_violations += 1

        elif altitude < 0.4:  # Close to ground
            max_safe_speed = 0.45
            max_safe_vy = -0.4

            if speed <= max_safe_speed and vy > max_safe_vy:
                speed_control += 5.0
                self.gentle_steps += 1
            elif speed > max_safe_speed * 1.5 or vy < -0.6:
                speed_control -= 6.0

        elif altitude < 0.6:  # Approaching
            max_safe_speed = 0.6
            if speed <= max_safe_speed:
                speed_control += 2.0
                self.gentle_steps += 1
            elif speed > max_safe_speed * 1.5:
                speed_control -= 3.0
        else:
            self.gentle_steps = 0

        # Bonus for sustained controlled approach
        if self.gentle_steps > 10:
            speed_control += 3.0

        # Track speed changes
        if self.prev_speed is not None:
            speed_change = speed - self.prev_speed
            # Reward slowing down when needed
            if speed_change < -0.02 and altitude < 1.0 and speed > 0.4:
                speed_control += 2.0
            # Penalize speeding up when already low
            elif speed_change > 0.05 and altitude < 0.5:
                speed_control -= 3.0

        self.prev_speed = speed

        # =============================================================
        # 4. CONTROLLED DESCENT REWARDS
        # =============================================================

        descent_control = 0.0

        # Reward appropriate descent rates based on altitude
        if altitude < 0.3:
            # Very low - need gentle descent or hovering
            if -0.25 < vy < 0.1:  # Allow slight upward for control
                descent_control += 6.0
            elif vy < -0.5:  # Falling too fast!
                descent_control -= 8.0

        elif altitude < 0.6:
            # Medium altitude - controlled descent
            if -0.4 < vy < -0.1:
                descent_control += 4.0
            elif vy < -0.7:
                descent_control -= 5.0

        else:
            # High altitude - can descend faster
            if -0.6 < vy < -0.2:
                descent_control += 2.0

        # =============================================================
        # 5. LANDING ZONE REWARDS
        # =============================================================

        landing_zone_bonus = 0.0

        # Define safe landing conditions
        in_good_position = distance_to_pad < 0.3
        good_angle = abs(angle) < 0.2
        controlled_speed = speed < 0.4
        safe_descent = vy > -0.4

        # Reward being in the landing zone with good conditions
        if in_good_position and altitude < 0.4:
            landing_zone_bonus += 5.0

            if controlled_speed and safe_descent:
                landing_zone_bonus += 5.0  # Excellent approach

            if good_angle:
                landing_zone_bonus += 3.0  # Good orientation

        # =============================================================
        # 6. BASIC GUIDANCE
        # =============================================================

        # Potential-based guidance toward pad
        phi = -distance_to_pad * 2.0 - altitude * 1.0

        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            shaped_reward += potential_bonus
        self.prev_phi = phi

        # =============================================================
        # 7. TERMINAL REWARDS
        # =============================================================

        terminal_bonus = 0.0

        if done:
            if terminated:
                if reward > 0:
                    # Success - reward based on landing quality
                    if speed < 0.3 and abs(vy) < 0.3:
                        terminal_bonus = 150.0  # Perfect gentle landing
                    elif speed < 0.4:
                        terminal_bonus = 100.0  # Good landing
                    elif speed < 0.6:
                        terminal_bonus = 50.0   # Acceptable landing
                    else:
                        terminal_bonus = 20.0   # Rough but successful
                else:
                    # Crash analysis
                    if self.falling_fast_steps > 5:
                        terminal_bonus = -30.0  # Crashed due to engine cutoff
                    elif speed > 0.7:
                        terminal_bonus = -20.0  # Speed-related crash
                    elif abs(vy) > 0.8:
                        terminal_bonus = -25.0  # Vertical speed crash
                    else:
                        terminal_bonus = -10.0  # Other crash
            else:
                # Timeout
                terminal_bonus = -15.0

        # =============================================================
        # 8. COMBINE ALL COMPONENTS
        # =============================================================

        total_shaping = (engine_safety_bonus + engine_penalty + speed_control +
                         descent_control + landing_zone_bonus + terminal_bonus)

        shaped_reward += total_shaping

        # =============================================================
        # 9. FUEL COMPENSATION
        # =============================================================

        fuel_compensation = 0.0
        if action != 0:  # Any engine use
            # Slightly higher compensation to encourage necessary engine use
            fuel_compensation += 0.1

        shaped_reward += fuel_compensation

        return shaped_reward


# Optional: Test function to verify behavior
def test_critical_scenarios():
    """Test the reward shaper on critical scenarios"""

    shaper = RewardShaper()
    shaper.reset()

    scenarios = [
        # Scenario: [x, y, vx, vy, angle, ang_vel, leg1, leg2], action, description
        ([0.0, 0.2, 0.0, -0.6, 0.0, 0.0, 0, 0], 0,
         "Falling fast near ground, engines OFF"),
        ([0.0, 0.2, 0.0, -0.6, 0.0, 0.0, 0, 0], 2,
         "Falling fast near ground, main engine ON"),
        ([0.0, 0.15, 0.0, -0.3, 0.0, 0.0, 0, 0], 2, "Cushioning landing"),
        ([0.0, 0.8, 0.0, -0.2, 0.0, 0.0, 0, 0], 0, "Safe coasting at altitude"),
        ([0.0, 0.1, 0.0, -0.8, 0.0, 0.0, 0, 0],
         0, "DANGER: Too fast, no engine!"),
    ]

    print("Testing critical scenarios:")
    print("=" * 60)

    for state, action, desc in scenarios:
        # Reset for each scenario to avoid carry-over
        shaper.reset()
        shaped = shaper.shape_reward(state, action, 0, False, 100)

        # Color code based on reward
        if shaped > 5:
            prefix = "✅"
        elif shaped < -5:
            prefix = "❌"
        else:
            prefix = "⚠️ "

        print(f"{prefix} {desc}")
        print(
            f"   State: y={state[1]:.2f}, vy={state[3]:.2f}, action={action}")
        print(f"   Shaped reward: {shaped:.2f}")
        print()


if __name__ == "__main__":
    test_critical_scenarios()
