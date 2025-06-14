import numpy as np


class RewardShaper:
    """
    Enhanced reward shaper that fixes the "side engine overuse during landing" problem.
    
    Key fixes:
    1. Landing zone detection - when very close to pad, discourage side engines
    2. "Settling bonus" - reward doing nothing when well-positioned and slow
    3. Progressive side engine discouragement as agent gets closer to landing
    4. Enhanced landing completion rewards
    """

    def __init__(self, gamma=0.99, enable_oscillation_penalty=True,
                 enable_commitment_bonus=True, enable_speed_control=True,
                 enable_engine_correction=True, enable_potential_guidance=True,
                 enable_approach_tracking=False, enable_horizontal_precision=True,
                 enable_landing_zone_control=True):
        self.gamma = gamma

        # Ablation control flags
        self.enable_oscillation_penalty = enable_oscillation_penalty
        self.enable_commitment_bonus = enable_commitment_bonus
        self.enable_speed_control = enable_speed_control
        self.enable_engine_correction = enable_engine_correction
        self.enable_potential_guidance = enable_potential_guidance
        self.enable_approach_tracking = enable_approach_tracking
        self.enable_horizontal_precision = enable_horizontal_precision
        self.enable_landing_zone_control = enable_landing_zone_control  # NEW

        # Enhanced horizontal guidance parameters
        self.osc_penalty_coeff = 1.5
        self.commitment_descent_bonus = 2.0
        self.commitment_vx_penalty = 2.5
        self.commitment_upward_penalty = 2.0
        self.speed_good_bonus = 2.0
        self.speed_bad_penalty = 3.0
        self.engine_main_penalty = 2.0
        self.engine_side_bonus = 2.0
        self.engine_coast_bonus = 1.0
        self.hover_penalty_max = 3.0
        self.hover_threshold_steps = 20
        self.horizontal_guidance_weight = 12.0
        self.horizontal_penalty_coeff = 3.0
        self.horizontal_penalty_altitude = 0.4

        # NEW: Landing zone control parameters
        # Slightly tighter than 0.2 to avoid teetering
        self.landing_zone_altitude = 0.18
        # Slightly tighter than 0.15 to avoid edge gaming
        self.landing_zone_distance = 0.12
        # Slightly slower than 0.4 for better control
        self.landing_zone_speed = 0.35
        # Bonus for doing nothing when positioned well
        self.settling_bonus = 3.0
        # Penalty for side engines in landing zone
        self.side_engine_landing_penalty = 4.0
        # Gradually increase penalty as getting closer
        self.progressive_side_penalty = True
        self.progressive_penalty_cap = 3.0     # Cap on progressive penalty growth
        # Reset overuse counter after this many non-side actions
        self.overuse_reset_threshold = 3

        # Terminal bonuses
        self.terminal_success_base = 20.0
        self.terminal_efficiency_bonus = 15.0
        self.terminal_gentle_bonus = 10.0
        self.terminal_failure_penalty = 5.0
        self.terminal_timeout_penalty = 10.0
        self.terminal_osc_threshold = 15
        self.terminal_osc_penalty = 5.0

        # Core tracking
        self.prev_phi = None

        # Oscillation detection
        self.velocity_history = []
        self.direction_changes = 0
        self.hover_count = 0

        # Approach tracking
        self.consecutive_approach_steps = 0
        self.last_distance = None

        # NEW: Landing zone tracking
        self.landing_zone_steps = 0
        self.side_engine_overuse_count = 0
        self.non_side_engine_streak = 0
        # Track consecutive non-side actions for overuse reset
        self.non_side_engine_streak = 0

    def reset(self):
        """Reset all episode-specific tracking - CALL THIS AT START OF EACH EPISODE!"""
        self.prev_phi = None
        self.velocity_history = []
        self.direction_changes = 0
        self.hover_count = 0
        self.consecutive_approach_steps = 0
        self.last_distance = None
        self.landing_zone_steps = 0
        self.side_engine_overuse_count = 0

    def shape_reward(self, state, action, reward, done, step, terminated=None, truncated=None):
        """
        Enhanced reward shaping with landing zone control to fix side engine overuse.
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
        # NEW: LANDING ZONE DETECTION
        # =============================================================

        in_landing_zone = (altitude < self.landing_zone_altitude and
                           distance_to_pad < self.landing_zone_distance and
                           speed < self.landing_zone_speed)

        near_landing_zone = (altitude < self.landing_zone_altitude * 1.5 and
                             distance_to_pad < self.landing_zone_distance * 1.5 and
                             speed < self.landing_zone_speed * 1.2)

        if in_landing_zone:
            self.landing_zone_steps += 1
        else:
            self.landing_zone_steps = 0

        # =============================================================
        # 1. OSCILLATION PENALTY
        # =============================================================

        if self.enable_oscillation_penalty and len(self.velocity_history) >= 6:
            vx_sign_changes = 0
            for i in range(1, len(self.velocity_history)):
                if (np.sign(self.velocity_history[i]) != np.sign(self.velocity_history[i-1])
                        and abs(self.velocity_history[i]) > 0.05):
                    vx_sign_changes += 1
                    self.direction_changes += 1

            if vx_sign_changes >= 2:
                oscillation_penalty = -self.osc_penalty_coeff * vx_sign_changes
                # ENHANCED: Double penalty if oscillating in landing zone
                if in_landing_zone:
                    oscillation_penalty *= 2.0
                shaped_reward += oscillation_penalty

        # =============================================================
        # 2. APPROACH TRACKING
        # =============================================================

        if self.enable_approach_tracking:
            approach_bonus = 0.0

            if self.last_distance is not None:
                distance_change = self.last_distance - distance_to_pad

                if distance_change > 0.01:
                    self.consecutive_approach_steps += 1
                    approach_bonus += 0.5

                    if self.consecutive_approach_steps > 10:
                        approach_bonus += 1.0

                elif distance_change < -0.01:
                    self.consecutive_approach_steps = 0
                    if altitude < 1.0:
                        approach_bonus -= 1.0
                else:
                    self.consecutive_approach_steps = max(
                        0, self.consecutive_approach_steps - 1)

            self.last_distance = distance_to_pad
            shaped_reward += approach_bonus

        # =============================================================
        # 3. COMMITMENT BONUS (enhanced horizontal control)
        # =============================================================

        if self.enable_commitment_bonus:
            commitment_bonus = 0.0

            if distance_to_pad < 0.4 and altitude < 0.8:
                # Reward downward movement
                if vy < -0.1:
                    commitment_bonus += self.commitment_descent_bonus

                    if altitude < 0.4 and vy < -0.05:
                        commitment_bonus += self.commitment_descent_bonus * 0.75

                # Penalty for horizontal movement in commitment zone
                if abs(vx) > 0.2:
                    commitment_bonus -= self.commitment_vx_penalty
                    # ENHANCED: Extra penalty in landing zone
                    if in_landing_zone:
                        commitment_bonus -= self.commitment_vx_penalty * 0.5

                # Penalty for going up when should be landing
                if vy > 0.1:
                    commitment_bonus -= self.commitment_upward_penalty

            shaped_reward += commitment_bonus

        # =============================================================
        # 4. SPEED CONTROL
        # =============================================================

        if self.enable_speed_control:
            speed_control = 0.0

            if altitude < 0.3:
                if speed <= 0.3:
                    speed_control += self.speed_good_bonus
                    # ENHANCED: Extra bonus for being slow in landing zone
                    if in_landing_zone and speed <= 0.2:
                        speed_control += self.speed_good_bonus * 0.5
                elif speed > 0.6:
                    speed_control -= self.speed_bad_penalty
            elif altitude < 0.6:
                if speed <= 0.5:
                    speed_control += self.speed_good_bonus * 0.5
                elif speed > 0.8:
                    speed_control -= self.speed_bad_penalty * 0.67

            shaped_reward += speed_control

        # =============================================================
        # 5. ENHANCED ENGINE CORRECTION (fixes the main problem!)
        # =============================================================

        if self.enable_engine_correction:
            engine_bonus = 0.0

            # Penalize main engine when already slow and low
            if action == 2 and speed < 0.4 and altitude < 0.6:
                engine_bonus -= self.engine_main_penalty

            # ENHANCED: Context-aware side engine handling
            elif action == 1 or action == 3:  # Side engines
                self.non_side_engine_streak = 0  # Reset non-side streak

                if in_landing_zone:
                    # STRONG penalty for side engines when in perfect landing position
                    engine_bonus -= self.side_engine_landing_penalty
                    self.side_engine_overuse_count += 1

                    # Progressive penalty with cap to prevent excessive punishment
                    if self.side_engine_overuse_count > 5:
                        progressive_penalty = min(
                            self.progressive_penalty_cap,
                            (self.side_engine_overuse_count - 5) * 0.5
                        )
                        engine_bonus -= progressive_penalty

                elif near_landing_zone:
                    # Moderate penalty when close to landing zone
                    if self.progressive_side_penalty:
                        penalty_factor = 1.0 - \
                            (distance_to_pad / (self.landing_zone_distance * 1.5))
                        engine_bonus -= self.side_engine_landing_penalty * penalty_factor
                    else:
                        engine_bonus -= self.side_engine_landing_penalty * 0.5

                elif distance_to_pad > 0.2:
                    # Normal bonus for side engines when actually needed for positioning
                    engine_bonus += self.engine_side_bonus
                else:
                    # Small penalty for unnecessary side engine use
                    engine_bonus -= 0.5

            # NEW: SETTLING BONUS - reward doing nothing when well-positioned
            elif action == 0:  # Do nothing
                self.non_side_engine_streak += 1

                # Reset overuse counter after sustained non-side engine actions
                if self.non_side_engine_streak >= self.overuse_reset_threshold:
                    self.side_engine_overuse_count = max(
                        0, self.side_engine_overuse_count - 1)

                if in_landing_zone:
                    # BIG bonus for doing nothing when perfectly positioned
                    engine_bonus += self.settling_bonus

                elif distance_to_pad < 0.3 and speed < 0.4:
                    # Standard coasting bonus
                    engine_bonus += self.engine_coast_bonus

            # Main engine case
            else:  # action == 2
                self.non_side_engine_streak += 1

                # Reset overuse counter for main engine use too
                if self.non_side_engine_streak >= self.overuse_reset_threshold:
                    self.side_engine_overuse_count = max(
                        0, self.side_engine_overuse_count - 1)

            shaped_reward += engine_bonus

        # =============================================================
        # 6. ENHANCED POTENTIAL-BASED GUIDANCE
        # =============================================================

        if self.enable_potential_guidance:
            # Adjust guidance based on landing zone proximity
            if in_landing_zone:
                # Minimal potential guidance when in landing zone - let agent settle
                guidance_weight = self.horizontal_guidance_weight * 0.3
            elif near_landing_zone:
                # Reduced guidance when close
                guidance_weight = self.horizontal_guidance_weight * 0.6
            else:
                # Full guidance when far from landing
                guidance_weight = self.horizontal_guidance_weight

            phi = -(distance_to_pad * guidance_weight +
                    max(0, altitude - 0.1) * 1.0)

            if self.prev_phi is not None:
                potential_bonus = self.gamma * phi - self.prev_phi
                shaped_reward += potential_bonus
            self.prev_phi = phi

        # =============================================================
        # 7. HORIZONTAL PRECISION PENALTY (adjusted for landing zone)
        # =============================================================

        if self.enable_horizontal_precision:
            # Reduce horizontal penalty in landing zone to avoid conflicting signals
            if altitude < self.horizontal_penalty_altitude:
                penalty_coeff = self.horizontal_penalty_coeff
                if in_landing_zone:
                    penalty_coeff *= 0.5  # Reduce penalty when already well-positioned

                horizontal_penalty = -penalty_coeff * abs(x)
                shaped_reward += horizontal_penalty

        # =============================================================
        # 8. NEW: LANDING ZONE CONTROL
        # =============================================================

        if self.enable_landing_zone_control:
            landing_control_bonus = 0.0

            # Reward sustained presence in landing zone with minimal movement
            if in_landing_zone and self.landing_zone_steps > 5:
                # Big bonus for staying in landing zone calmly
                if abs(vx) < 0.1 and abs(vy) < 0.2:
                    landing_control_bonus += 2.0

                # Additional bonus for extended calm presence
                if self.landing_zone_steps > 10:
                    landing_control_bonus += 1.0

            # Penalty for leaving landing zone unnecessarily
            if self.landing_zone_steps > 0 and not in_landing_zone:
                if distance_to_pad > self.landing_zone_distance * 1.2:
                    landing_control_bonus -= 2.0  # Don't drift away from landing zone!

            shaped_reward += landing_control_bonus

        # =============================================================
        # 9. ANTI-HOVERING (enhanced for landing zone)
        # =============================================================

        if speed < 0.15 and altitude > 0.2:
            self.hover_count += 1
            if self.hover_count > self.hover_threshold_steps:
                hover_penalty = -min(self.hover_penalty_max,
                                     (self.hover_count - self.hover_threshold_steps) * 0.1)
                shaped_reward += hover_penalty
        else:
            self.hover_count = 0

        # =============================================================
        # 10. TERMINAL REWARDS (enhanced for landing completion)
        # =============================================================

        if done:
            terminal_bonus = 0.0

            if terminated:
                if reward > 0:
                    terminal_bonus = self.terminal_success_base

                    if step < 300:
                        terminal_bonus += self.terminal_efficiency_bonus

                    if speed < 0.3:
                        terminal_bonus += self.terminal_gentle_bonus

                    # NEW: Extra bonus for calm landing (low side engine overuse)
                    if self.side_engine_overuse_count < 10:
                        terminal_bonus += 10.0  # Bonus for not overusing side engines

                else:
                    if speed > 0.7:
                        terminal_bonus = -self.terminal_failure_penalty
                    else:
                        terminal_bonus = -self.terminal_failure_penalty * 0.4
            else:
                terminal_bonus = -self.terminal_timeout_penalty

                if self.direction_changes > self.terminal_osc_threshold:
                    terminal_bonus -= self.terminal_osc_penalty

                # NEW: Extra timeout penalty for side engine overuse
                if self.side_engine_overuse_count > 20:
                    terminal_bonus -= 5.0

            shaped_reward += terminal_bonus

        # =============================================================
        # 11. FUEL COMPENSATION
        # =============================================================

        if action != 0:
            fuel_compensation = 0.05
            # Reduce fuel compensation for side engines in landing zone
            if (action == 1 or action == 3) and in_landing_zone:
                fuel_compensation *= 0.5
            shaped_reward += fuel_compensation

        # Apply safety clamps
        max_total_shaping = 20.0
        if abs(shaped_reward - reward) > max_total_shaping:
            excess = shaped_reward - reward
            clamped_excess = np.sign(excess) * max_total_shaping
            shaped_reward = reward + clamped_excess

        return shaped_reward

    def get_debug_info(self, state=None):
        """Return current shaping state for debugging with enhanced landing zone analysis"""
        debug_info = {
            'direction_changes': self.direction_changes,
            'hover_count': self.hover_count,
            'consecutive_approach_steps': self.consecutive_approach_steps,
            'velocity_history_length': len(self.velocity_history),
            'last_distance': self.last_distance,
            'landing_zone_steps': self.landing_zone_steps,
            'side_engine_overuse_count': self.side_engine_overuse_count,
            'non_side_engine_streak': self.non_side_engine_streak,
            'horizontal_guidance_weight': self.horizontal_guidance_weight,
            'horizontal_penalty_coeff': self.horizontal_penalty_coeff
        }

        # Add current landing zone status if state provided
        if state is not None:
            x, y, vx, vy = state[0], state[1], state[2], state[3]
            speed = np.hypot(vx, vy)
            altitude = max(0, y)
            distance_to_pad = abs(x)

            in_landing_zone = (altitude < self.landing_zone_altitude and
                               distance_to_pad < self.landing_zone_distance and
                               speed < self.landing_zone_speed)

            near_landing_zone = (altitude < self.landing_zone_altitude * 1.5 and
                                 distance_to_pad < self.landing_zone_distance * 1.5 and
                                 speed < self.landing_zone_speed * 1.2)

            debug_info.update({
                'current_altitude': altitude,
                'current_distance_to_pad': distance_to_pad,
                'current_speed': speed,
                'in_landing_zone': in_landing_zone,
                'near_landing_zone': near_landing_zone,
                'zone_altitude_threshold': self.landing_zone_altitude,
                'zone_distance_threshold': self.landing_zone_distance,
                'zone_speed_threshold': self.landing_zone_speed,
                'settling_bonus_active': in_landing_zone,
                'side_engine_penalty_active': in_landing_zone or near_landing_zone
            })

        return debug_info

    def compute_landing_zone_status(self, state):
        """Compute landing zone status for a given state - useful for external analysis"""
        x, y, vx, vy = state[0], state[1], state[2], state[3]
        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        in_landing_zone = (altitude < self.landing_zone_altitude and
                           distance_to_pad < self.landing_zone_distance and
                           speed < self.landing_zone_speed)

        near_landing_zone = (altitude < self.landing_zone_altitude * 1.5 and
                             distance_to_pad < self.landing_zone_distance * 1.5 and
                             speed < self.landing_zone_speed * 1.2)

        return {
            'altitude': altitude,
            'distance_to_pad': distance_to_pad,
            'speed': speed,
            'in_landing_zone': in_landing_zone,
            'near_landing_zone': near_landing_zone
        }

    def update_parameters(self, **kwargs):
        """Update tunable parameters for easy hyperparameter sweeps"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"✅ Updated {key} = {value}")
            else:
                print(f"❌ Warning: Parameter '{key}' not found")

    def set_landing_strictness(self, mode="moderate"):
        """Preset configurations for different landing completion strictness"""
        modes = {
            "gentle": {
                'landing_zone_altitude': 0.25,
                'landing_zone_distance': 0.2,
                'landing_zone_speed': 0.5,
                'settling_bonus': 2.0,
                'side_engine_landing_penalty': 2.0
            },
            "moderate": {
                'landing_zone_altitude': 0.2,
                'landing_zone_distance': 0.15,
                'landing_zone_speed': 0.4,
                'settling_bonus': 3.0,
                'side_engine_landing_penalty': 4.0
            },
            "strict": {
                'landing_zone_altitude': 0.15,
                'landing_zone_distance': 0.1,
                'landing_zone_speed': 0.3,
                'settling_bonus': 4.0,
                'side_engine_landing_penalty': 6.0
            },
            "extreme": {
                'landing_zone_altitude': 0.1,
                'landing_zone_distance': 0.08,
                'landing_zone_speed': 0.25,
                'settling_bonus': 5.0,
                'side_engine_landing_penalty': 8.0
            }
        }

        if mode not in modes:
            print(f"❌ Unknown mode '{mode}'. Available: {list(modes.keys())}")
            return

        for param, value in modes[mode].items():
            setattr(self, param, value)

        print(f"✅ Set landing strictness to '{mode}' mode")
        print(f"   Landing zone altitude: {self.landing_zone_altitude}")
        print(f"   Landing zone distance: {self.landing_zone_distance}")
        print(f"   Settling bonus: {self.settling_bonus}")
        print(
            f"   Side engine penalty in landing zone: {self.side_engine_landing_penalty}")

    def set_horizontal_precision_mode(self, mode="moderate"):
        """Preset configurations for different precision levels"""
        modes = {
            "gentle": {
                'horizontal_guidance_weight': 8.0,
                'horizontal_penalty_coeff': 1.5,
                'commitment_vx_penalty': 1.5,
                'engine_side_bonus': 1.2
            },
            "moderate": {
                'horizontal_guidance_weight': 10.0,
                'horizontal_penalty_coeff': 2.5,
                'commitment_vx_penalty': 2.0,
                'engine_side_bonus': 1.5
            },
            "aggressive": {
                'horizontal_guidance_weight': 12.0,
                'horizontal_penalty_coeff': 3.0,
                'commitment_vx_penalty': 2.5,
                'engine_side_bonus': 2.0
            },
            "extreme": {
                'horizontal_guidance_weight': 15.0,
                'horizontal_penalty_coeff': 4.0,
                'commitment_vx_penalty': 3.0,
                'engine_side_bonus': 2.5
            }
        }

        if mode not in modes:
            print(f"❌ Unknown mode '{mode}'. Available: {list(modes.keys())}")
            return

        for param, value in modes[mode].items():
            setattr(self, param, value)

        print(f"✅ Set horizontal precision to '{mode}' mode")
        print(
            f"   Horizontal guidance weight: {self.horizontal_guidance_weight}")
        print(f"   Horizontal penalty coeff: {self.horizontal_penalty_coeff}")
        print(f"   Commitment vx penalty: {self.commitment_vx_penalty}")
        print(f"   Engine side bonus: {self.engine_side_bonus}")
