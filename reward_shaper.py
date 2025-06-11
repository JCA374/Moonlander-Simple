import numpy as np


class RewardShaper:
    """
    Modular anti-oscillation reward shaper with proper scaling and ablation support.
    
    Each shaping component can be enabled/disabled for ablation studies.
    All penalties/bonuses are scaled to work with LunarLander's ±100-200 reward range.
    """

    def __init__(self, gamma=0.99, enable_oscillation_penalty=True,
                 enable_commitment_bonus=True, enable_speed_control=True,
                 enable_engine_correction=True, enable_potential_guidance=True,
                 enable_approach_tracking=False):
        self.gamma = gamma

        # Ablation control flags
        self.enable_oscillation_penalty = enable_oscillation_penalty
        self.enable_commitment_bonus = enable_commitment_bonus
        self.enable_speed_control = enable_speed_control
        self.enable_engine_correction = enable_engine_correction
        self.enable_potential_guidance = enable_potential_guidance
        self.enable_approach_tracking = enable_approach_tracking

        # Tunable parameters (extracted from hard-coded constants)
        self.osc_penalty_coeff = 1.5
        self.commitment_descent_bonus = 2.0
        self.commitment_vx_penalty = 1.0
        self.commitment_upward_penalty = 2.0
        self.speed_good_bonus = 2.0
        self.speed_bad_penalty = 3.0
        self.engine_main_penalty = 2.0
        self.engine_side_bonus = 0.5
        self.engine_coast_bonus = 1.0
        self.hover_penalty_max = 3.0
        self.hover_threshold_steps = 20
        self.terminal_success_base = 20.0
        self.terminal_efficiency_bonus = 15.0
        self.terminal_gentle_bonus = 10.0
        self.terminal_failure_penalty = 5.0
        self.terminal_timeout_penalty = 10.0
        self.terminal_osc_threshold = 15
        self.terminal_osc_penalty = 5.0

        # Core tracking
        self.prev_phi = None

        # Oscillation detection (simplified)
        self.velocity_history = []
        self.direction_changes = 0  # Total flips for entire episode
        self.hover_count = 0

        # Approach tracking (now properly implemented)
        self.consecutive_approach_steps = 0
        self.last_distance = None

    def reset(self):
        """Reset all episode-specific tracking - CALL THIS AT START OF EACH EPISODE!"""
        self.prev_phi = None
        self.velocity_history = []
        self.direction_changes = 0
        self.hover_count = 0
        self.consecutive_approach_steps = 0
        self.last_distance = None

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Apply modular reward shaping with proper scaling.
        All bonuses/penalties scaled to 1-5 range to avoid overwhelming environment signal.
        """
        shaped_reward = reward

        x, y = state[0], state[1]
        vx, vy = state[2], state[3]
        angle = state[4]

        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # Update velocity history for oscillation detection
        self.velocity_history.append(vx)
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)

        # =============================================================
        # 1. OSCILLATION PENALTY (Simplified and properly scaled)
        # =============================================================

        if self.enable_oscillation_penalty and len(self.velocity_history) >= 6:
            # Count velocity direction changes (simpler than position peaks)
            vx_sign_changes = 0
            for i in range(1, len(self.velocity_history)):
                if (np.sign(self.velocity_history[i]) != np.sign(self.velocity_history[i-1])
                        and abs(self.velocity_history[i]) > 0.05):  # Ignore tiny movements
                    vx_sign_changes += 1
                    self.direction_changes += 1  # Track total for episode

            # Apply penalty based on recent oscillations
            if vx_sign_changes >= 2:  # Multiple direction changes in recent history
                oscillation_penalty = -self.osc_penalty_coeff * vx_sign_changes
                shaped_reward += oscillation_penalty

        # =============================================================
        # 2. APPROACH TRACKING (Now properly implemented)
        # =============================================================

        if self.enable_approach_tracking:
            approach_bonus = 0.0

            # Track if agent is making progress toward landing
            if self.last_distance is not None:
                distance_change = self.last_distance - distance_to_pad

                if distance_change > 0.01:  # Moving closer to pad
                    self.consecutive_approach_steps += 1
                    approach_bonus += 0.5

                    # Bonus for sustained approach
                    if self.consecutive_approach_steps > 10:
                        approach_bonus += 1.0

                elif distance_change < -0.01:  # Moving away from pad
                    self.consecutive_approach_steps = 0
                    if altitude < 1.0:  # Penalize moving away when low
                        approach_bonus -= 1.0
                else:
                    # Not making much progress - slow decay
                    self.consecutive_approach_steps = max(
                        0, self.consecutive_approach_steps - 1)

            self.last_distance = distance_to_pad
            shaped_reward += approach_bonus

        # =============================================================
        # 3. COMMITMENT BONUS (When close to pad, reward descent)
        # =============================================================

        if self.enable_commitment_bonus:
            commitment_bonus = 0.0

            # Only apply when in "commitment zone"
            if distance_to_pad < 0.4 and altitude < 0.8:
                # Reward downward movement
                if vy < -0.1:
                    commitment_bonus += self.commitment_descent_bonus

                    # Extra bonus for sustained descent when very close
                    if altitude < 0.4 and vy < -0.05:
                        commitment_bonus += self.commitment_descent_bonus * 0.75

                # Small penalty for horizontal movement in commitment zone
                if abs(vx) > 0.2:
                    commitment_bonus -= self.commitment_vx_penalty

                # Penalty for going up when should be landing
                if vy > 0.1:
                    commitment_bonus -= self.commitment_upward_penalty

            shaped_reward += commitment_bonus

        # =============================================================
        # 4. SPEED CONTROL (Scaled down from original)
        # =============================================================

        if self.enable_speed_control:
            speed_control = 0.0

            # Altitude-based speed requirements (much gentler scaling)
            if altitude < 0.3:
                if speed <= 0.3:
                    speed_control += self.speed_good_bonus
                elif speed > 0.6:
                    speed_control -= self.speed_bad_penalty
            elif altitude < 0.6:
                if speed <= 0.5:
                    speed_control += self.speed_good_bonus * 0.5
                elif speed > 0.8:
                    speed_control -= self.speed_bad_penalty * 0.67

            shaped_reward += speed_control

        # =============================================================
        # 5. ENGINE CORRECTION (Simplified)
        # =============================================================

        if self.enable_engine_correction:
            engine_bonus = 0.0

            # Penalize main engine when already slow and low
            if action == 2 and speed < 0.4 and altitude < 0.6:
                engine_bonus -= self.engine_main_penalty

            # Reward side engines for control
            elif (action == 1 or action == 3) and distance_to_pad > 0.1:
                engine_bonus += self.engine_side_bonus

            # Reward coasting when well-positioned
            elif action == 0 and distance_to_pad < 0.3 and speed < 0.4:
                engine_bonus += self.engine_coast_bonus

            shaped_reward += engine_bonus

        # =============================================================
        # 6. POTENTIAL-BASED GUIDANCE (Fixed)
        # =============================================================

        if self.enable_potential_guidance:
            # Strictly decreasing potential as we approach goal
            # φ = -(distance + height penalty)
            phi = -(distance_to_pad * 2.0 + max(0, altitude - 0.1) * 1.0)

            # No bonus terms that could flip the sign

            if self.prev_phi is not None:
                potential_bonus = self.gamma * phi - self.prev_phi
                shaped_reward += potential_bonus
            self.prev_phi = phi

        # =============================================================
        # 7. ANTI-HOVERING (Improved reset logic)
        # =============================================================

        # Improved hovering detection with aggressive reset
        if speed < 0.15 and altitude > 0.2:
            self.hover_count += 1
            if self.hover_count > self.hover_threshold_steps:
                hover_penalty = -min(self.hover_penalty_max,
                                     (self.hover_count - self.hover_threshold_steps) * 0.1)
                shaped_reward += hover_penalty
        else:
            # Aggressive reset - any good movement clears the slate
            self.hover_count = 0

        # =============================================================
        # 8. TERMINAL REWARDS (Moderate scaling)
        # =============================================================

        if done:
            terminal_bonus = 0.0

            if terminated:
                if reward > 0:
                    # Success bonuses (scaled down)
                    terminal_bonus = self.terminal_success_base

                    # Efficiency bonus for quick landing
                    if step < 300:
                        terminal_bonus += self.terminal_efficiency_bonus

                    # Gentleness bonus
                    if speed < 0.3:
                        terminal_bonus += self.terminal_gentle_bonus

                else:
                    # Failure analysis (gentle penalties)
                    if speed > 0.7:
                        terminal_bonus = -self.terminal_failure_penalty
                    else:
                        terminal_bonus = -self.terminal_failure_penalty * 0.4
            else:
                # Timeout penalty
                terminal_bonus = -self.terminal_timeout_penalty

                # Extra penalty for excessive oscillation (consistent scale with detection)
                if self.direction_changes > self.terminal_osc_threshold:
                    terminal_bonus -= self.terminal_osc_penalty

            shaped_reward += terminal_bonus

        # =============================================================
        # 9. FUEL COMPENSATION (Unchanged)
        # =============================================================

        if action != 0:
            shaped_reward += 0.05

        # Apply safety clamps to prevent runaway shaping
        max_total_shaping = 15.0  # Prevent overwhelming environment signal
        if abs(shaped_reward - reward) > max_total_shaping:
            excess = shaped_reward - reward
            clamped_excess = np.sign(excess) * max_total_shaping
            shaped_reward = reward + clamped_excess

        return shaped_reward

    def get_debug_info(self):
        """Return current shaping state for debugging"""
        return {
            'direction_changes': self.direction_changes,
            'hover_count': self.hover_count,
            'consecutive_approach_steps': self.consecutive_approach_steps,
            'velocity_history_length': len(self.velocity_history),
            'last_distance': self.last_distance
        }

    def update_parameters(self, **kwargs):
        """Update tunable parameters for easy hyperparameter sweeps"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Parameter '{key}' not found")
