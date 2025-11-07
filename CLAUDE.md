# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) reinforcement learning project that trains an agent to land the Lunar Lander in OpenAI Gymnasium's LunarLander-v2 environment. The project features sophisticated reward shaping to guide learning and extensive logging/evaluation capabilities.

**Recent Major Refactoring (2024):**
- Added comprehensive configuration system with dataclasses
- Implemented type hints throughout codebase
- Extracted model management into separate module
- Replaced magic numbers with named constants and enums
- Modularized training loop into focused functions

## Common Commands

### Training
```bash
# Start fresh training
python train.py

# Training will prompt for episode count if resuming from existing model
# Default: 25,000 episodes for new training, 200,000 for resumed training
```

### Evaluation & Visualization
```bash
# Play the best model visually
python play_best.py

# Play specific model with custom episodes
python play_best.py models/moonlander_final.pth 10

# Detailed evaluation with metrics
python evaluator.py

# Compare multiple models
python evaluator.py compare models/moonlander_best.pth models/moonlander_final.pth

# Analyze agent behavior problems
python landing_analysis.py
```

### Running Tests
```bash
# Speed diagnostics
python tests/speed_diagnostics.py

# Debug analyzer
python tests/debug_analyzer.py
```

## Architecture Overview

### Configuration System (`config.py`) - **NEW**
**Centralized Configuration Management**
- `DQNConfig`: Agent hyperparameters (learning rate, epsilon, tau, batch size, memory size, etc.)
- `RewardShaperConfig`: All 50+ reward shaping parameters with preset modes
- `TrainingConfig`: Training loop parameters (episodes, evaluation intervals, checkpoints, etc.)
- `MoonlanderConfig`: Master config combining all sub-configurations
- Support for loading/saving configs from JSON/YAML files
- Built-in validation for parameter ranges (assertions in `__post_init__`)
- Example usage:
```python
from config import MoonlanderConfig

# Use defaults
config = MoonlanderConfig()

# Load from file
config = MoonlanderConfig.from_yaml('my_config.yaml')

# Modify and save
config.dqn.learning_rate = 0.0001
config.save_yaml('modified_config.yaml')
```

### Constants (`constants.py`) - **NEW**
**Named Constants and Enums**
- `LanderAction`: IntEnum for actions (DO_NOTHING, FIRE_LEFT_ENGINE, FIRE_MAIN_ENGINE, FIRE_RIGHT_ENGINE)
- `StateIndex`: IntEnum for state vector indices (X_POSITION, Y_POSITION, X_VELOCITY, etc.)
- `FailureReason`: String constants for failure classification
- Centralized thresholds, speeds, and zone definitions
- Eliminates magic numbers throughout codebase (0.5 → LANDING_PAD_X_MAX, etc.)
- Use `Action.FIRE_MAIN_ENGINE` instead of `2` for clarity

### Model Management (`model_manager.py`) - **NEW**
**Extracted Model Saving Logic**
- `ModelManager`: Handles all model evaluation, saving, and loading
- `ModelEvaluationResult`: Dataclass for clean evaluation results with fields:
  - `should_save`: bool
  - `save_type`: "best", "improving", or "checkpoint"
  - `improvement_reason`: str (human-readable explanation)
  - `new_best_score`: float
  - `new_best_rate`: float
  - `reset_counter`: bool
- Conservative saving criteria to prevent regressions
- Automatic backup creation for best models (timestamped)
- Debug utilities (`debug_model_saving()`) for understanding save decisions
- Separates concerns: model management vs training loop

### Core Components

**DQN Agent** (`dqn_agent.py`)
- Dueling DQN architecture with value and advantage streams
- Double DQN for action selection (online network selects, target network evaluates)
- Soft target network updates with tau parameter (default 0.001)
- Layer normalization for training stability
- Replay buffer with 50,000 transitions
- Epsilon-greedy exploration with adaptive decay
- **NEW: Config-based initialization** with `DQNConfig` dataclass
- **NEW: Comprehensive type hints** using `from __future__ import annotations`
- **NEW: Saves/loads full state** including optimizer, target network, and config
- **NEW: Better error handling** with specific exceptions and helpful messages
- **NEW: `get_stats()` method** for agent inspection

**Reward Shaper** (`reward_shaper.py`) - **Most Complex Component**
- Extensive reward engineering to guide the agent toward successful landings
- 11 distinct reward shaping mechanisms that can be toggled via config flags:
  1. Oscillation penalty (discourages side-to-side movement)
  2. Approach tracking (rewards getting closer to landing pad)
  3. Commitment bonus (rewards descent when close to pad)
  4. Speed control (rewards/penalizes based on altitude-appropriate speeds)
  5. Engine correction (context-aware penalties/bonuses for engine use)
  6. Potential-based guidance (distance-to-pad and altitude shaping)
  7. Horizontal precision penalty (penalizes deviation from center)
  8. Landing zone control (manages behavior in final landing phase)
  9. Anti-hovering (penalizes stationary behavior at altitude)
  10. Terminal rewards (bonuses for success, penalties for failure)
  11. Fuel compensation (offsets built-in fuel penalty)
- **NEW: Refactored into composable helper methods** (`_compute_oscillation_penalty()`, etc.)
- **NEW: Config integration** while maintaining backwards compatibility
- **NEW: Uses constants** for thresholds (StateIndex, GENTLE_LANDING_SPEED, etc.)
- **NEW: Type hints** with numpy typing (NDArray)

**Landing Zone Control System** (in `reward_shaper.py`)
- Detects when agent is in "landing zone" (low altitude, close to pad, slow speed)
- Strongly penalizes side engine use during landing to prevent "teetering"
- Rewards "settling" behavior (doing nothing when well-positioned)
- Progressive penalties for repeated side engine overuse
- Configurable strictness modes: gentle, moderate, strict, extreme

**Training Loop** (`train.py`) - **Heavily Refactored**
- **NEW: Modular function-based design** with clear separation of concerns:
  - `setup_environment()`: Creates Gymnasium environment
  - `initialize_training()`: Sets up all components (agent, shaper, logger, etc.)
  - `load_existing_model()`: Handles model loading and episode count
  - `establish_baseline()`: Runs initial evaluation
  - `classify_failure()`: Categorizes failure reasons
  - `run_episode()`: Executes single training episode
  - `print_progress()`: Displays training progress
  - `evaluate_and_save()`: Evaluates and potentially saves model
  - `plot_training_results()`: Generates training plots
- Uses `ModelManager` for all model operations
- Sophisticated model evaluation logic that tracks:
  - True landing success rate (game's positive terminal reward)
  - Shaped vs original scores
  - Episodes since last improvement
- Conservative model saving prevents "regressive" saves:
  - Primary criterion: landing rate improvement
  - Secondary: score improvement with same landing rate
  - Special cases: early training flexibility, milestone saves, perfect performance
- Learning rate scheduling (StepLR with gamma=0.5 every 5000 steps)
- Comprehensive logging of episode data, failures, and fuel usage
- Dynamic checkpoint intervals (10 checkpoints per training run)

### Model Evaluation Logic

The training system uses `ModelManager.evaluate_model()` in model_manager.py which implements a hierarchical evaluation:
1. **Best models**: Landing rate improvements or score improvements with same landing rate
2. **Improving models**: Early training progress (first 1000 episodes) or milestone checkpoints (every 2000 episodes with decent performance)
3. Tracks `episodes_since_best` to detect training plateaus
4. Provides detailed debugging output via `debug_model_saving()`
5. Returns `ModelEvaluationResult` dataclass for clean interfaces

### Key Configuration Parameters

**DQN Agent Defaults (in `DQNConfig`):**
- Learning rate: 0.0003 (Adam optimizer)
- Tau (soft update): 0.001
- Epsilon decay: 0.995 (0.6 when resuming training)
- Epsilon min: 0.02
- Batch size: 64
- Gamma (discount): 0.99
- Replay buffer size: 50,000
- Hidden layer size: 128

**Reward Shaper Modes (in `RewardShaperConfig`):**
```python
# Horizontal precision (affects guidance strength)
reward_shaper.set_horizontal_precision_mode("aggressive")  # gentle, moderate, aggressive, extreme

# Landing strictness (affects landing zone thresholds)
reward_shaper.set_landing_strictness("moderate")  # gentle, moderate, strict, extreme
```

**Training Config Defaults:**
- New training: 25,000 episodes
- Resume training: 200,000 episodes
- Evaluation interval: 200 episodes
- Checkpoint count: 10
- Progress report interval: 50 episodes

### Logging & Evaluation

**TrainingLogger** (`logger.py`)
- Comprehensive session logging to JSON, CSV, and debug text files
- Tracks learning diagnostics: plateaus, oscillations, action distribution
- Landing streak tracking
- Loss and Q-value variance monitoring
- Automatic problem detection (high variance, action bias, exploration issues)

**Evaluator** (`evaluator.py`)
- `quick_evaluate()`: Fast 5-episode evaluation for training loop decisions
- `detailed_evaluate()`: Comprehensive analysis with outcome breakdown
- `compare_models()`: Side-by-side model comparison
- Tracks multiple metrics: game success, both-legs landing, final speeds, behavioral analysis

**Landing Analyzer** (`landing_analysis.py`)
- Deep behavioral analysis with trajectory plotting
- Identifies specific problems: action bias, poor control, crashes
- Provides actionable recommendations for fixing training issues

### Video Generation

- `individual_video_generator.py`: Creates videos from individual episodes
- `merge_movies.py`: Concatenates multiple videos with comments
- Videos saved to `videos/` or `model_to_video/` directories

## Directory Structure

- `models/` - Saved model checkpoints
  - `moonlander_best.pth` - Best performing model
  - `moonlander_final.pth` - Final model from training
  - `moonlander_improving_*.pth` - Intermediate improving models
  - `moonlander_checkpoint_*.pth` - Regular checkpoints
  - `moonlander_backup_*.pth` - Backups of best models
- `logs/` - Training session logs (JSON, CSV, debug text)
- `videos/` - Generated episode videos
- `old/` - Previous reward shaper implementations
- `tests/` - Diagnostic and debugging scripts
- `config.py` - **NEW:** Configuration management
- `constants.py` - **NEW:** Named constants and enums
- `model_manager.py` - **NEW:** Model evaluation and saving logic

## Important Implementation Details

### Configuration-Based Initialization

All major components now support config-based initialization:
```python
from config import MoonlanderConfig

config = MoonlanderConfig()
config.dqn.learning_rate = 0.0001
config.reward_shaper.landing_zone_altitude = 0.15

agent = DQNAgent(state_size, action_size, config=config.dqn)
reward_shaper = RewardShaper(config=config.reward_shaper)
```

### Type Hints

The codebase now uses comprehensive type hints:
- `from __future__ import annotations` for forward references
- `Optional`, `Tuple`, `Dict` from `typing`
- `NDArray` from `numpy.typing` for array types
- `str | Path` union syntax for modern Python

### Fuel Compensation Logic

Added in `reward_shaper.py:628`, the fuel compensation provides a small reward offset (0.05) when engines are used to counteract the environment's built-in fuel penalty. This prevents the agent from being overly penalized for necessary engine usage. The compensation is reduced by 50% for side engines in the landing zone.

### Landing Success Detection

True landing success is determined by `terminated and reward > 0`. The code also tracks auxiliary metrics like both-legs-touching and one-leg-touching for debugging, but model evaluation uses the game's own success criterion.

### Failure Analysis

The training loop uses `classify_failure()` in train.py:181 to categorize failures:
- `FailureReason.SUCCESS` - Successful landing
- `FailureReason.NO_GROUND_CONTACT` - Didn't touch ground
- `FailureReason.ONE_LEG_ONLY` - Only one leg touched
- `FailureReason.OUTSIDE_PAD` - Landed outside the pad (|x| > 0.5)
- `FailureReason.TOO_FAST` - Speed > 1.0 at landing
- `FailureReason.OTHER_CRASH` - Other crash reasons
- `FailureReason.TIMEOUT` - Episode didn't terminate within 1000 steps

### Model Loading Behavior

When resuming training with an existing `models/moonlander_best.pth`:
- Loads the model weights via `ModelManager.load_best_model()`
- Resets epsilon to 0.6 (not the saved value) for continued exploration
- Creates a timestamped backup automatically
- Establishes fresh baseline via initial evaluation (80% of initial performance)
- Prompts for number of training episodes

## Debugging & Troubleshooting

If the agent isn't learning well:
1. Check action distribution in training output - should be relatively balanced
2. Run `landing_analysis.py` to identify specific behavioral problems
3. Check logs for plateau warnings (no improvement for 3000+ episodes)
4. Verify landing zone behavior - agent should settle calmly, not oscillate
5. Compare shaped vs original rewards - large divergence may indicate overfitting to shaped rewards
6. Use `ModelManager.debug_model_saving()` to understand why models aren't being saved

If side engines are overused during landing:
- Landing zone control system should prevent this
- Check `side_engine_overuse_count` in reward shaper debug info
- Consider stricter landing mode: `reward_shaper.set_landing_strictness("strict")`
- Verify landing zone thresholds in config are appropriate

## Code Quality Improvements

Recent improvements follow reinforcement learning and Python best practices:
- **Configuration Management**: Centralized, validated configs with file I/O
- **Type Safety**: Comprehensive type hints for better IDE support and error detection
- **Modularity**: Separated concerns (model management, constants, training logic)
- **Error Handling**: Specific exceptions with helpful error messages
- **Documentation**: Detailed docstrings with Args, Returns, and Examples
- **Constants**: Named constants instead of magic numbers
- **Clean Code**: Extracted functions, clear naming, single responsibility principle
