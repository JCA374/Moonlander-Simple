"""
Configuration management for Moonlander DQN training.

This module provides dataclasses for managing hyperparameters and configuration
settings across the DQN agent, reward shaper, and training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import json
import yaml
from pathlib import Path


@dataclass
class DQNConfig:
    """Configuration for DQN Agent hyperparameters."""

    # Network architecture
    state_size: int = 8
    action_size: int = 4
    hidden_size: int = 128

    # Learning parameters
    learning_rate: float = 0.0003
    gamma: float = 0.99  # Discount factor
    tau: float = 0.001  # Soft update parameter for target network

    # Exploration parameters
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995
    epsilon_resume: float = 0.6  # Epsilon when resuming training

    # Memory and batch parameters
    replay_buffer_size: int = 50000
    batch_size: int = 64
    min_replay_size: int = 64  # Minimum samples before training starts

    # Training frequency
    target_update_frequency: int = 100  # Episodes between target network updates

    def __post_init__(self):
        """Validate configuration parameters."""
        assert 0 < self.learning_rate < 1, "Learning rate must be between 0 and 1"
        assert 0 < self.gamma <= 1, "Gamma must be between 0 and 1"
        assert 0 < self.tau <= 1, "Tau must be between 0 and 1"
        assert 0 <= self.epsilon_min <= self.epsilon_start <= 1, "Invalid epsilon values"
        assert 0 < self.epsilon_decay <= 1, "Epsilon decay must be between 0 and 1"
        assert self.batch_size <= self.replay_buffer_size, "Batch size cannot exceed buffer size"


@dataclass
class RewardShaperConfig:
    """Configuration for reward shaping parameters."""

    # Enable/disable mechanisms
    enable_oscillation_penalty: bool = True
    enable_commitment_bonus: bool = True
    enable_speed_control: bool = True
    enable_engine_correction: bool = True
    enable_potential_guidance: bool = True
    enable_approach_tracking: bool = False
    enable_horizontal_precision: bool = True
    enable_landing_zone_control: bool = True

    # Core parameters
    gamma: float = 0.99

    # Oscillation control
    osc_penalty_coeff: float = 1.5

    # Commitment zone parameters
    commitment_descent_bonus: float = 2.0
    commitment_vx_penalty: float = 2.5
    commitment_upward_penalty: float = 2.0

    # Speed control
    speed_good_bonus: float = 2.0
    speed_bad_penalty: float = 3.0

    # Engine control
    engine_main_penalty: float = 2.0
    engine_side_bonus: float = 2.0
    engine_coast_bonus: float = 1.0

    # Hover prevention
    hover_penalty_max: float = 3.0
    hover_threshold_steps: int = 20

    # Horizontal guidance
    horizontal_guidance_weight: float = 12.0
    horizontal_penalty_coeff: float = 3.0
    horizontal_penalty_altitude: float = 0.4

    # Landing zone control
    landing_zone_altitude: float = 0.18
    landing_zone_distance: float = 0.12
    landing_zone_speed: float = 0.35
    settling_bonus: float = 3.0
    side_engine_landing_penalty: float = 4.0
    progressive_side_penalty: bool = True
    progressive_penalty_cap: float = 3.0
    overuse_reset_threshold: int = 3

    # Terminal rewards
    terminal_success_base: float = 20.0
    terminal_efficiency_bonus: float = 15.0
    terminal_gentle_bonus: float = 10.0
    terminal_failure_penalty: float = 5.0
    terminal_timeout_penalty: float = 10.0
    terminal_osc_threshold: int = 15
    terminal_osc_penalty: float = 5.0

    # Fuel compensation
    fuel_compensation_amount: float = 0.05

    # Safety limits
    max_total_shaping: float = 20.0

    def set_horizontal_precision_mode(self, mode: str):
        """Set horizontal precision preset mode."""
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
            raise ValueError(f"Unknown mode '{mode}'. Available: {list(modes.keys())}")

        for param, value in modes[mode].items():
            setattr(self, param, value)

    def set_landing_strictness(self, mode: str):
        """Set landing zone strictness preset mode."""
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
            raise ValueError(f"Unknown mode '{mode}'. Available: {list(modes.keys())}")

        for param, value in modes[mode].items():
            setattr(self, param, value)


@dataclass
class TrainingConfig:
    """Configuration for training loop parameters."""

    # Episode settings
    episodes_new_training: int = 25000
    episodes_resume_training: int = 200000
    max_steps_per_episode: int = 1000

    # Model evaluation and saving
    evaluation_interval: int = 200  # Episodes between evaluations
    checkpoint_count: int = 10  # Number of checkpoints to save
    eval_episodes: int = 5  # Episodes for quick evaluation
    initial_eval_episodes: int = 10  # Episodes for initial baseline
    baseline_multiplier: float = 0.8  # Baseline = initial_performance * this

    # Model saving criteria
    score_improvement_threshold: float = 20.0  # Min score improvement to save
    plateau_warning_episodes: int = 3000  # Warn if no improvement for this many
    milestone_save_interval: int = 2000  # Save milestone if no improvement
    milestone_min_landing_rate: float = 0.8  # Min landing rate for milestone
    milestone_min_score: float = 500.0  # Min score for milestone

    # Learning rate scheduling
    lr_scheduler_step_size: int = 5000
    lr_scheduler_gamma: float = 0.5

    # PyTorch settings
    torch_num_threads: int = 4

    # Directory settings
    models_dir: str = "models"
    logs_dir: str = "logs"
    videos_dir: str = "videos"

    # Progress reporting
    progress_report_interval: int = 50  # Episodes between progress reports

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.episodes_new_training > 0, "Episodes must be positive"
        assert self.max_steps_per_episode > 0, "Max steps must be positive"
        assert self.evaluation_interval > 0, "Evaluation interval must be positive"
        assert 0 < self.baseline_multiplier <= 1, "Baseline multiplier must be between 0 and 1"


@dataclass
class MoonlanderConfig:
    """Master configuration combining all sub-configurations."""

    dqn: DQNConfig = field(default_factory=DQNConfig)
    reward_shaper: RewardShaperConfig = field(default_factory=RewardShaperConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @classmethod
    def from_dict(cls, config_dict: dict) -> MoonlanderConfig:
        """Create configuration from dictionary."""
        dqn_config = DQNConfig(**config_dict.get('dqn', {}))
        reward_config = RewardShaperConfig(**config_dict.get('reward_shaper', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))

        return cls(dqn=dqn_config, reward_shaper=reward_config, training=training_config)

    @classmethod
    def from_json(cls, path: str | Path) -> MoonlanderConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MoonlanderConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'dqn': self.dqn.__dict__,
            'reward_shaper': self.reward_shaper.__dict__,
            'training': self.training.__dict__
        }

    def save_json(self, path: str | Path):
        """Save configuration to JSON file."""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def save_yaml(self, path: str | Path):
        """Save configuration to YAML file."""
        path = Path(path)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = MoonlanderConfig()
