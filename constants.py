"""
Constants and enums for Moonlander DQN project.

This module centralizes all magic numbers, action definitions, and thresholds
used throughout the codebase.
"""

from enum import IntEnum


# ============================================================================
# Action Space
# ============================================================================

class LanderAction(IntEnum):
    """Actions available to the Lunar Lander agent."""
    DO_NOTHING = 0
    FIRE_LEFT_ENGINE = 1
    FIRE_MAIN_ENGINE = 2
    FIRE_RIGHT_ENGINE = 3


# Alias for convenience
Action = LanderAction

# Action name mapping for display
ACTION_NAMES = {
    Action.DO_NOTHING: "Do Nothing",
    Action.FIRE_LEFT_ENGINE: "Fire Left",
    Action.FIRE_MAIN_ENGINE: "Fire Main",
    Action.FIRE_RIGHT_ENGINE: "Fire Right"
}


# ============================================================================
# State Space Indices
# ============================================================================

class StateIndex(IntEnum):
    """Indices for state vector components."""
    X_POSITION = 0
    Y_POSITION = 1
    X_VELOCITY = 2
    Y_VELOCITY = 3
    ANGLE = 4
    ANGULAR_VELOCITY = 5
    LEFT_LEG_CONTACT = 6
    RIGHT_LEG_CONTACT = 7


# ============================================================================
# Landing Criteria
# ============================================================================

# Position thresholds
LANDING_PAD_X_MIN = -0.5
LANDING_PAD_X_MAX = 0.5
LANDING_PAD_CENTER = 0.0

# Speed thresholds
MAX_SAFE_LANDING_SPEED = 0.5
GENTLE_LANDING_SPEED = 0.3
VERY_SLOW_SPEED = 0.15

# Angle thresholds
MAX_SAFE_LANDING_ANGLE = 0.2

# Altitude levels
VERY_LOW_ALTITUDE = 0.2
LOW_ALTITUDE = 0.3
MEDIUM_ALTITUDE = 0.6
HIGH_ALTITUDE = 0.8


# ============================================================================
# Reward Shaping Zones
# ============================================================================

# Commitment zone (where agent should commit to landing)
COMMITMENT_ZONE_DISTANCE = 0.4
COMMITMENT_ZONE_ALTITUDE = 0.8

# Near landing zone (approach phase)
NEAR_LANDING_MULTIPLIER = 1.5

# Distance thresholds
CLOSE_TO_PAD = 0.2
VERY_CLOSE_TO_PAD = 0.1


# ============================================================================
# Velocity Thresholds
# ============================================================================

# Horizontal velocity thresholds
HIGH_HORIZONTAL_VELOCITY = 0.2
MINIMAL_HORIZONTAL_VELOCITY = 0.05

# Vertical velocity thresholds
DESCENDING_VELOCITY = -0.1
SLOW_DESCENT = -0.05
ASCENDING_VELOCITY = 0.1


# ============================================================================
# Training Metrics
# ============================================================================

# Score thresholds for success classification
PERFECT_LANDING_SCORE = 200
DECENT_LANDING_SCORE = 100

# Episode length thresholds
EFFICIENT_EPISODE_LENGTH = 300
MAX_EPISODE_STEPS = 1000

# Streak tracking
MIN_REPORTABLE_STREAK = 3


# ============================================================================
# Model Evaluation
# ============================================================================

# Landing rate thresholds
PERFECT_LANDING_RATE = 1.0
HIGH_LANDING_RATE = 0.8
MODERATE_LANDING_RATE = 0.5

# Performance classification
VERY_HIGH_VARIANCE = 10000
HIGH_ACTION_BIAS_THRESHOLD = 0.7
LOW_ACTION_USAGE_THRESHOLD = 0.05

# Early training period (more lenient evaluation)
EARLY_TRAINING_EPISODES = 1000


# ============================================================================
# Failure Classification
# ============================================================================

class FailureReason:
    """String constants for failure reason classification."""
    SUCCESS = "success"
    NO_GROUND_CONTACT = "no_ground_contact"
    ONE_LEG_ONLY = "one_leg_only"
    OUTSIDE_PAD = "outside_pad"
    TOO_FAST = "too_fast"
    OTHER_CRASH = "other_crash"
    TIMEOUT = "timeout"


# ============================================================================
# File Paths
# ============================================================================

MODEL_BEST_FILENAME = "moonlander_best.pth"
MODEL_FINAL_FILENAME = "moonlander_final.pth"
MODEL_CHECKPOINT_PREFIX = "moonlander_checkpoint_"
MODEL_IMPROVING_PREFIX = "moonlander_improving_"
MODEL_BACKUP_PREFIX = "moonlander_backup_"

LOG_PREFIX = "training_"
CSV_PREFIX = "scores_"
DEBUG_PREFIX = "debug_"


# ============================================================================
# Environment Settings
# ============================================================================

ENVIRONMENT_NAME = "LunarLander-v2"


# ============================================================================
# Display Settings
# ============================================================================

# Console output
SEPARATOR_LINE = "=" * 60
SUBSEPARATOR_LINE = "-" * 60

# Emojis for console output
EMOJI_ROCKET = "🚀"
EMOJI_TARGET = "🎯"
EMOJI_TROPHY = "🏆"
EMOJI_FIRE = "🔥"
EMOJI_WARNING = "⚠️"
EMOJI_SUCCESS = "✅"
EMOJI_FAIL = "❌"
EMOJI_CLOCK = "⏰"
EMOJI_CHART = "📊"
EMOJI_SAVE = "💾"
EMOJI_SEARCH = "🔍"


# ============================================================================
# Logging
# ============================================================================

# Window sizes for averaging
SHORT_WINDOW = 10
MEDIUM_WINDOW = 50
LONG_WINDOW = 100

# Reporting intervals
REPORT_EVERY_N_EPISODES = 50
SAVE_CHECKPOINT_EVERY_N = 200
