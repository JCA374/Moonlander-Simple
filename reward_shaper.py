"""
Reward shaping for Lunar Lander DQN training.

This module implements comprehensive reward shaping with 11 distinct mechanisms
to guide the agent toward successful landings:
1. Oscillation penalty
2. Approach tracking
3. Commitment bonus
4. Speed control
5. Engine correction
6. Potential-based guidance
7. Horizontal precision penalty
8. Landing zone control
9. Anti-hovering
10. Terminal rewards
11. Fuel compensation
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any
from numpy.typing import NDArray

from config import RewardShaperConfig
from constants import (
    Action,
    StateIndex,
    COMMITMENT_ZONE_DISTANCE,
    COMMITMENT_ZONE_ALTITUDE,
    VERY_LOW_ALTITUDE,
    LOW_ALTITUDE,
    MEDIUM_ALTITUDE,
    DESCENDING_VELOCITY,
    SLOW_DESCENT,
    ASCENDING_VELOCITY,
    HIGH_HORIZONTAL_VELOCITY,
    MINIMAL_HORIZONTAL_VELOCITY,
    CLOSE_TO_PAD,
    GENTLE_LANDING_SPEED,
    EFFICIENT_EPISODE_LENGTH,
    VERY_SLOW_SPEED,
    NEAR_LANDING_MULTIPLIER
)


class RewardShaper:
    """
    Enhanced reward shaper for Lunar Lander training.

    Implements multiple reward shaping mechanisms to guide the agent toward
    successful landings while preventing common failure modes like:
    - Excessive oscillation (side-to-side movement)
    - Side engine overuse during final landing
    - Hovering without progress
    - Poor speed control

    Args:
        config: Reward shaper configuration (optional, uses defaults if None)

    Example:
        >>> shaper = RewardShaper()
        >>> shaper.reset()  # Call at start of each episode
        >>> shaped_reward = shaper.shape_reward(state, action, reward, done, step)
    """

    def __init__(self, config: Optional[RewardShaperConfig] = None, **kwargs):
        """
        Initialize reward shaper.

        Args:
            config: Configuration object (optional)
            **kwargs: Individual config parameters (for backwards compatibility)
        """
        # Use provided config or create from kwargs or use defaults
        if config is None:
            # Support legacy initialization with individual parameters
            if kwargs:
                config = RewardShaperConfig(**kwargs)
            else:
                config = RewardShaperConfig()

        self.config = config

        # Copy config values to instance for easier access
        self._load_config(config)

        # Episode-specific tracking (reset each episode)
        self.prev_phi: Optional[float] = None
        self.velocity_history: list[float] = []
        self.direction_changes: int = 0
        self.hover_count: int = 0
        self.consecutive_approach_steps: int = 0
        self.last_distance: Optional[float] = None
        self.landing_zone_steps: int = 0
        self.side_engine_overuse_count: int = 0
        self.non_side_engine_streak: int = 0

    def _load_config(self, config: RewardShaperConfig) -> None:
        """Load configuration values into instance variables."""
        # Enable/disable flags
        self.enable_oscillation_penalty = config.enable_oscillation_penalty
        self.enable_commitment_bonus = config.enable_commitment_bonus
        self.enable_speed_control = config.enable_speed_control
        self.enable_engine_correction = config.enable_engine_correction
        self.enable_potential_guidance = config.enable_potential_guidance
        self.enable_approach_tracking = config.enable_approach_tracking
        self.enable_horizontal_precision = config.enable_horizontal_precision
        self.enable_landing_zone_control = config.enable_landing_zone_control

        # Core parameters
        self.gamma = config.gamma

        # Reward coefficients
        self.osc_penalty_coeff = config.osc_penalty_coeff
        self.commitment_descent_bonus = config.commitment_descent_bonus
        self.commitment_vx_penalty = config.commitment_vx_penalty
        self.commitment_upward_penalty = config.commitment_upward_penalty
        self.speed_good_bonus = config.speed_good_bonus
        self.speed_bad_penalty = config.speed_bad_penalty
        self.engine_main_penalty = config.engine_main_penalty
        self.engine_side_bonus = config.engine_side_bonus
        self.engine_coast_bonus = config.engine_coast_bonus
        self.hover_penalty_max = config.hover_penalty_max
        self.hover_threshold_steps = config.hover_threshold_steps
        self.horizontal_guidance_weight = config.horizontal_guidance_weight
        self.horizontal_penalty_coeff = config.horizontal_penalty_coeff
        self.horizontal_penalty_altitude = config.horizontal_penalty_altitude

        # Landing zone parameters
        self.landing_zone_altitude = config.landing_zone_altitude
        self.landing_zone_distance = config.landing_zone_distance
        self.landing_zone_speed = config.landing_zone_speed
        self.settling_bonus = config.settling_bonus
        self.side_engine_landing_penalty = config.side_engine_landing_penalty
        self.progressive_side_penalty = config.progressive_side_penalty
        self.progressive_penalty_cap = config.progressive_penalty_cap
        self.overuse_reset_threshold = config.overuse_reset_threshold

        # Terminal rewards
        self.terminal_success_base = config.terminal_success_base
        self.terminal_efficiency_bonus = config.terminal_efficiency_bonus
        self.terminal_gentle_bonus = config.terminal_gentle_bonus
        self.terminal_failure_penalty = config.terminal_failure_penalty
        self.terminal_timeout_penalty = config.terminal_timeout_penalty
        self.terminal_osc_threshold = config.terminal_osc_threshold
        self.terminal_osc_penalty = config.terminal_osc_penalty

        # Fuel compensation
        self.fuel_compensation_amount = config.fuel_compensation_amount

        # Safety limits
        self.max_total_shaping = config.max_total_shaping

    def reset(self) -> None:
        """
        Reset all episode-specific tracking.

        MUST be called at the start of each episode to ensure clean state.
        """
        self.prev_phi = None
        self.velocity_history = []
        self.direction_changes = 0
        self.hover_count = 0
        self.consecutive_approach_steps = 0
        self.last_distance = None
        self.landing_zone_steps = 0
        self.side_engine_overuse_count = 0
        self.non_side_engine_streak = 0

    def shape_reward(
        self,
        state: NDArray[np.float64],
        action: int,
        reward: float,
        done: bool,
        step: int,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None
    ) -> float:
        """
        Apply reward shaping to guide learning.

        Args:
            state: Current state vector (8 elements)
            action: Action taken (0-3)
            reward: Original reward from environment
            done: Whether episode is done
            step: Current step number
            terminated: Whether episode terminated naturally (optional)
            truncated: Whether episode was truncated by time limit (optional)

        Returns:
            Shaped reward value
        """
        shaped_reward = reward

        # Extract state components
        x = state[StateIndex.X_POSITION]
        y = state[StateIndex.Y_POSITION]
        vx = state[StateIndex.X_VELOCITY]
        vy = state[StateIndex.Y_VELOCITY]
        angle = state[StateIndex.ANGLE]

        # Derived quantities
        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        # Update velocity history for oscillation detection
        self.velocity_history.append(vx)
        if len(self.velocity_history) > 10:
            self.velocity_history.pop(0)

        # ================================================================
        # LANDING ZONE DETECTION
        # ================================================================
        in_landing_zone = (
            altitude < self.landing_zone_altitude and
            distance_to_pad < self.landing_zone_distance and
            speed < self.landing_zone_speed
        )

        near_landing_zone = (
            altitude < self.landing_zone_altitude * NEAR_LANDING_MULTIPLIER and
            distance_to_pad < self.landing_zone_distance * NEAR_LANDING_MULTIPLIER and
            speed < self.landing_zone_speed * 1.2
        )

        if in_landing_zone:
            self.landing_zone_steps += 1
        else:
            self.landing_zone_steps = 0

        # ================================================================
        # 1. OSCILLATION PENALTY
        # ================================================================
        if self.enable_oscillation_penalty:
            shaped_reward += self._compute_oscillation_penalty(in_landing_zone)

        # ================================================================
        # 2. APPROACH TRACKING
        # ================================================================
        if self.enable_approach_tracking:
            shaped_reward += self._compute_approach_bonus(distance_to_pad, altitude)

        # ================================================================
        # 3. COMMITMENT BONUS
        # ================================================================
        if self.enable_commitment_bonus:
            shaped_reward += self._compute_commitment_bonus(
                distance_to_pad, altitude, vx, vy, in_landing_zone
            )

        # ================================================================
        # 4. SPEED CONTROL
        # ================================================================
        if self.enable_speed_control:
            shaped_reward += self._compute_speed_control(altitude, speed, in_landing_zone)

        # ================================================================
        # 5. ENGINE CORRECTION
        # ================================================================
        if self.enable_engine_correction:
            shaped_reward += self._compute_engine_correction(
                action, speed, altitude, distance_to_pad, in_landing_zone, near_landing_zone
            )

        # ================================================================
        # 6. POTENTIAL-BASED GUIDANCE
        # ================================================================
        if self.enable_potential_guidance:
            shaped_reward += self._compute_potential_guidance(
                distance_to_pad, altitude, in_landing_zone, near_landing_zone
            )

        # ================================================================
        # 7. HORIZONTAL PRECISION PENALTY
        # ================================================================
        if self.enable_horizontal_precision:
            shaped_reward += self._compute_horizontal_precision(altitude, x, in_landing_zone)

        # ================================================================
        # 8. LANDING ZONE CONTROL
        # ================================================================
        if self.enable_landing_zone_control:
            shaped_reward += self._compute_landing_zone_control(
                in_landing_zone, distance_to_pad, vx, vy
            )

        # ================================================================
        # 9. ANTI-HOVERING
        # ================================================================
        shaped_reward += self._compute_hover_penalty(speed, altitude)

        # ================================================================
        # 10. TERMINAL REWARDS
        # ================================================================
        if done:
            shaped_reward += self._compute_terminal_reward(
                terminated, reward, step, speed
            )

        # ================================================================
        # 11. FUEL COMPENSATION
        # ================================================================
        shaped_reward += self._compute_fuel_compensation(action, in_landing_zone)

        # ================================================================
        # SAFETY CLAMP
        # ================================================================
        shaped_reward = self._clamp_shaped_reward(shaped_reward, reward)

        return shaped_reward

    def _compute_oscillation_penalty(self, in_landing_zone: bool) -> float:
        """Penalize rapid direction changes (oscillation)."""
        if len(self.velocity_history) < 6:
            return 0.0

        vx_sign_changes = 0
        for i in range(1, len(self.velocity_history)):
            if (np.sign(self.velocity_history[i]) != np.sign(self.velocity_history[i-1]) and
                    abs(self.velocity_history[i]) > MINIMAL_HORIZONTAL_VELOCITY):
                vx_sign_changes += 1
                self.direction_changes += 1

        if vx_sign_changes >= 2:
            oscillation_penalty = -self.osc_penalty_coeff * vx_sign_changes
            # Double penalty if oscillating in landing zone
            if in_landing_zone:
                oscillation_penalty *= 2.0
            return oscillation_penalty

        return 0.0

    def _compute_approach_bonus(self, distance_to_pad: float, altitude: float) -> float:
        """Reward sustained approach toward landing pad."""
        if self.last_distance is None:
            self.last_distance = distance_to_pad
            return 0.0

        distance_change = self.last_distance - distance_to_pad
        approach_bonus = 0.0

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
            self.consecutive_approach_steps = max(0, self.consecutive_approach_steps - 1)

        self.last_distance = distance_to_pad
        return approach_bonus

    def _compute_commitment_bonus(
        self,
        distance_to_pad: float,
        altitude: float,
        vx: float,
        vy: float,
        in_landing_zone: bool
    ) -> float:
        """Reward commitment to landing when close to pad."""
        if distance_to_pad >= COMMITMENT_ZONE_DISTANCE or altitude >= COMMITMENT_ZONE_ALTITUDE:
            return 0.0

        commitment_bonus = 0.0

        # Reward downward movement
        if vy < DESCENDING_VELOCITY:
            commitment_bonus += self.commitment_descent_bonus

            if altitude < COMMITMENT_ZONE_ALTITUDE / 2 and vy < SLOW_DESCENT:
                commitment_bonus += self.commitment_descent_bonus * 0.75

        # Penalty for horizontal movement in commitment zone
        if abs(vx) > HIGH_HORIZONTAL_VELOCITY:
            commitment_bonus -= self.commitment_vx_penalty
            # Extra penalty in landing zone
            if in_landing_zone:
                commitment_bonus -= self.commitment_vx_penalty * 0.5

        # Penalty for going up when should be landing
        if vy > ASCENDING_VELOCITY:
            commitment_bonus -= self.commitment_upward_penalty

        return commitment_bonus

    def _compute_speed_control(
        self,
        altitude: float,
        speed: float,
        in_landing_zone: bool
    ) -> float:
        """Reward appropriate speeds based on altitude."""
        speed_control = 0.0

        if altitude < LOW_ALTITUDE:
            if speed <= LOW_ALTITUDE:
                speed_control += self.speed_good_bonus
                # Extra bonus for being slow in landing zone
                if in_landing_zone and speed <= VERY_LOW_ALTITUDE:
                    speed_control += self.speed_good_bonus * 0.5
            elif speed > 0.6:
                speed_control -= self.speed_bad_penalty

        elif altitude < MEDIUM_ALTITUDE:
            if speed <= 0.5:
                speed_control += self.speed_good_bonus * 0.5
            elif speed > 0.8:
                speed_control -= self.speed_bad_penalty * 0.67

        return speed_control

    def _compute_engine_correction(
        self,
        action: int,
        speed: float,
        altitude: float,
        distance_to_pad: float,
        in_landing_zone: bool,
        near_landing_zone: bool
    ) -> float:
        """Context-aware engine usage rewards/penalties."""
        engine_bonus = 0.0

        # Penalize main engine when already slow and low
        if action == Action.FIRE_MAIN_ENGINE and speed < 0.4 and altitude < MEDIUM_ALTITUDE:
            engine_bonus -= self.engine_main_penalty

        # Side engines - context-aware
        elif action in (Action.FIRE_LEFT_ENGINE, Action.FIRE_RIGHT_ENGINE):
            self.non_side_engine_streak = 0  # Reset non-side streak

            if in_landing_zone:
                # STRONG penalty for side engines when in perfect landing position
                engine_bonus -= self.side_engine_landing_penalty
                self.side_engine_overuse_count += 1

                # Progressive penalty with cap
                if self.side_engine_overuse_count > 5:
                    progressive_penalty = min(
                        self.progressive_penalty_cap,
                        (self.side_engine_overuse_count - 5) * 0.5
                    )
                    engine_bonus -= progressive_penalty

            elif near_landing_zone:
                # Moderate penalty when close to landing zone
                if self.progressive_side_penalty:
                    penalty_factor = 1.0 - (distance_to_pad / (self.landing_zone_distance * NEAR_LANDING_MULTIPLIER))
                    engine_bonus -= self.side_engine_landing_penalty * penalty_factor
                else:
                    engine_bonus -= self.side_engine_landing_penalty * 0.5

            elif distance_to_pad > CLOSE_TO_PAD:
                # Normal bonus for side engines when actually needed for positioning
                engine_bonus += self.engine_side_bonus
            else:
                # Small penalty for unnecessary side engine use
                engine_bonus -= 0.5

        # Do nothing
        elif action == Action.DO_NOTHING:
            self.non_side_engine_streak += 1

            # Reset overuse counter after sustained non-side engine actions
            if self.non_side_engine_streak >= self.overuse_reset_threshold:
                self.side_engine_overuse_count = max(0, self.side_engine_overuse_count - 1)

            if in_landing_zone:
                # BIG bonus for doing nothing when perfectly positioned
                engine_bonus += self.settling_bonus
            elif distance_to_pad < LOW_ALTITUDE and speed < 0.4:
                # Standard coasting bonus
                engine_bonus += self.engine_coast_bonus

        # Main engine case (not covered above)
        else:  # action == Action.FIRE_MAIN_ENGINE
            self.non_side_engine_streak += 1

            # Reset overuse counter for main engine use too
            if self.non_side_engine_streak >= self.overuse_reset_threshold:
                self.side_engine_overuse_count = max(0, self.side_engine_overuse_count - 1)

        return engine_bonus

    def _compute_potential_guidance(
        self,
        distance_to_pad: float,
        altitude: float,
        in_landing_zone: bool,
        near_landing_zone: bool
    ) -> float:
        """Potential-based shaping for gradual guidance."""
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

        phi = -(distance_to_pad * guidance_weight + max(0, altitude - 0.1) * 1.0)

        if self.prev_phi is not None:
            potential_bonus = self.gamma * phi - self.prev_phi
            self.prev_phi = phi
            return potential_bonus

        self.prev_phi = phi
        return 0.0

    def _compute_horizontal_precision(
        self,
        altitude: float,
        x: float,
        in_landing_zone: bool
    ) -> float:
        """Penalize horizontal deviation when low."""
        if altitude >= self.horizontal_penalty_altitude:
            return 0.0

        penalty_coeff = self.horizontal_penalty_coeff
        # Reduce penalty in landing zone to avoid conflicting signals
        if in_landing_zone:
            penalty_coeff *= 0.5

        return -penalty_coeff * abs(x)

    def _compute_landing_zone_control(
        self,
        in_landing_zone: bool,
        distance_to_pad: float,
        vx: float,
        vy: float
    ) -> float:
        """Reward sustained calm presence in landing zone."""
        landing_control_bonus = 0.0

        # Reward sustained presence in landing zone with minimal movement
        if in_landing_zone and self.landing_zone_steps > 5:
            # Big bonus for staying in landing zone calmly
            if abs(vx) < MINIMAL_HORIZONTAL_VELOCITY * 2 and abs(vy) < VERY_LOW_ALTITUDE:
                landing_control_bonus += 2.0

            # Additional bonus for extended calm presence
            if self.landing_zone_steps > 10:
                landing_control_bonus += 1.0

        # Penalty for leaving landing zone unnecessarily
        if self.landing_zone_steps > 0 and not in_landing_zone:
            if distance_to_pad > self.landing_zone_distance * 1.2:
                landing_control_bonus -= 2.0  # Don't drift away!

        return landing_control_bonus

    def _compute_hover_penalty(self, speed: float, altitude: float) -> float:
        """Penalize stationary behavior at altitude."""
        if speed < VERY_SLOW_SPEED and altitude > VERY_LOW_ALTITUDE:
            self.hover_count += 1
            if self.hover_count > self.hover_threshold_steps:
                hover_penalty = -min(
                    self.hover_penalty_max,
                    (self.hover_count - self.hover_threshold_steps) * 0.1
                )
                return hover_penalty
        else:
            self.hover_count = 0

        return 0.0

    def _compute_terminal_reward(
        self,
        terminated: Optional[bool],
        reward: float,
        step: int,
        speed: float
    ) -> float:
        """Compute rewards for episode termination."""
        terminal_bonus = 0.0

        if terminated:
            if reward > 0:
                # Successful landing
                terminal_bonus = self.terminal_success_base

                if step < EFFICIENT_EPISODE_LENGTH:
                    terminal_bonus += self.terminal_efficiency_bonus

                if speed < GENTLE_LANDING_SPEED:
                    terminal_bonus += self.terminal_gentle_bonus

                # Extra bonus for calm landing (low side engine overuse)
                if self.side_engine_overuse_count < 10:
                    terminal_bonus += 10.0

            else:
                # Crash
                if speed > 0.7:
                    terminal_bonus = -self.terminal_failure_penalty
                else:
                    terminal_bonus = -self.terminal_failure_penalty * 0.4
        else:
            # Timeout
            terminal_bonus = -self.terminal_timeout_penalty

            if self.direction_changes > self.terminal_osc_threshold:
                terminal_bonus -= self.terminal_osc_penalty

            # Extra timeout penalty for side engine overuse
            if self.side_engine_overuse_count > 20:
                terminal_bonus -= 5.0

        return terminal_bonus

    def _compute_fuel_compensation(self, action: int, in_landing_zone: bool) -> float:
        """Compensate for environment's built-in fuel penalty."""
        if action == Action.DO_NOTHING:
            return 0.0

        fuel_compensation = self.fuel_compensation_amount

        # Reduce fuel compensation for side engines in landing zone
        if action in (Action.FIRE_LEFT_ENGINE, Action.FIRE_RIGHT_ENGINE) and in_landing_zone:
            fuel_compensation *= 0.5

        return fuel_compensation

    def _clamp_shaped_reward(self, shaped_reward: float, original_reward: float) -> float:
        """Apply safety clamps to prevent excessive shaping."""
        excess = shaped_reward - original_reward

        if abs(excess) > self.max_total_shaping:
            clamped_excess = np.sign(excess) * self.max_total_shaping
            return original_reward + clamped_excess

        return shaped_reward

    def get_debug_info(self, state: Optional[NDArray[np.float64]] = None) -> Dict[str, Any]:
        """
        Get current shaping state for debugging.

        Args:
            state: Optional state vector to compute current landing zone status

        Returns:
            Dictionary with debug information
        """
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
            x, y = state[StateIndex.X_POSITION], state[StateIndex.Y_POSITION]
            vx, vy = state[StateIndex.X_VELOCITY], state[StateIndex.Y_VELOCITY]
            speed = np.hypot(vx, vy)
            altitude = max(0, y)
            distance_to_pad = abs(x)

            in_landing_zone = (
                altitude < self.landing_zone_altitude and
                distance_to_pad < self.landing_zone_distance and
                speed < self.landing_zone_speed
            )

            near_landing_zone = (
                altitude < self.landing_zone_altitude * NEAR_LANDING_MULTIPLIER and
                distance_to_pad < self.landing_zone_distance * NEAR_LANDING_MULTIPLIER and
                speed < self.landing_zone_speed * 1.2
            )

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

    def compute_landing_zone_status(self, state: NDArray[np.float64]) -> Dict[str, Any]:
        """
        Compute landing zone status for a given state.

        Useful for external analysis and visualization.

        Args:
            state: State vector

        Returns:
            Dictionary with landing zone information
        """
        x, y = state[StateIndex.X_POSITION], state[StateIndex.Y_POSITION]
        vx, vy = state[StateIndex.X_VELOCITY], state[StateIndex.Y_VELOCITY]
        speed = np.hypot(vx, vy)
        altitude = max(0, y)
        distance_to_pad = abs(x)

        in_landing_zone = (
            altitude < self.landing_zone_altitude and
            distance_to_pad < self.landing_zone_distance and
            speed < self.landing_zone_speed
        )

        near_landing_zone = (
            altitude < self.landing_zone_altitude * NEAR_LANDING_MULTIPLIER and
            distance_to_pad < self.landing_zone_distance * NEAR_LANDING_MULTIPLIER and
            speed < self.landing_zone_speed * 1.2
        )

        return {
            'altitude': altitude,
            'distance_to_pad': distance_to_pad,
            'speed': speed,
            'in_landing_zone': in_landing_zone,
            'near_landing_zone': near_landing_zone
        }

    def update_parameters(self, **kwargs) -> None:
        """
        Update tunable parameters for easy hyperparameter sweeps.

        Args:
            **kwargs: Parameter name-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                print(f"✅ Updated {key} = {value}")
            elif hasattr(self.config, key):
                setattr(self.config, key, value)
                setattr(self, key, value)
                print(f"✅ Updated {key} = {value}")
            else:
                print(f"❌ Warning: Parameter '{key}' not found")

    def set_landing_strictness(self, mode: str) -> None:
        """
        Set landing zone strictness preset mode.

        Args:
            mode: One of "gentle", "moderate", "strict", "extreme"

        Raises:
            ValueError: If mode is invalid
        """
        self.config.set_landing_strictness(mode)
        self._load_config(self.config)  # Reload parameters

        print(f"✅ Set landing strictness to '{mode}' mode")
        print(f"   Landing zone altitude: {self.landing_zone_altitude}")
        print(f"   Landing zone distance: {self.landing_zone_distance}")
        print(f"   Settling bonus: {self.settling_bonus}")
        print(f"   Side engine penalty: {self.side_engine_landing_penalty}")

    def set_horizontal_precision_mode(self, mode: str) -> None:
        """
        Set horizontal precision preset mode.

        Args:
            mode: One of "gentle", "moderate", "aggressive", "extreme"

        Raises:
            ValueError: If mode is invalid
        """
        self.config.set_horizontal_precision_mode(mode)
        self._load_config(self.config)  # Reload parameters

        print(f"✅ Set horizontal precision to '{mode}' mode")
        print(f"   Horizontal guidance weight: {self.horizontal_guidance_weight}")
        print(f"   Horizontal penalty coeff: {self.horizontal_penalty_coeff}")
        print(f"   Commitment vx penalty: {self.commitment_vx_penalty}")
        print(f"   Engine side bonus: {self.engine_side_bonus}")
