# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Deep Q-Network (DQN) reinforcement learning project that trains an agent to land the Lunar Lander in OpenAI Gymnasium's LunarLander-v2 environment. The project features sophisticated reward shaping to guide learning and extensive logging/evaluation capabilities.

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

### Core Components

**DQN Agent** (`dqn_agent.py`)
- Dueling DQN architecture with value and advantage streams
- Double DQN for action selection (online network selects, target network evaluates)
- Soft target network updates with tau parameter (default 0.001)
- Layer normalization for training stability
- Replay buffer with 50,000 transitions
- Epsilon-greedy exploration with adaptive decay

**Reward Shaper** (`reward_shaper.py`) - **Most Complex Component**
- Extensive reward engineering to guide the agent toward successful landings
- 11 distinct reward shaping mechanisms that can be toggled via flags:
  1. Oscillation penalty (discourages side-to-side movement)
  2. Approach tracking (rewards getting closer to landing pad)
  3. Commitment bonus (rewards descent when close to pad)
  4. Speed control (rewards/penalizes based on altitude-appropriate speeds)
  5. Engine correction (context-aware penalties/bonuses for engine use)
  6. Potential-based guidance (distance-to-pad and altitude shaping)
  7. Horizontal precision penalty (penalizes deviation from center)
  8. Landing zone control (NEW: manages behavior in final landing phase)
  9. Anti-hovering (penalizes stationary behavior at altitude)
  10. Terminal rewards (bonuses for success, penalties for failure)
  11. Fuel compensation (offsets built-in fuel penalty)

**Landing Zone Control System** (in `reward_shaper.py`)
- Detects when agent is in "landing zone" (low altitude, close to pad, slow speed)
- Strongly penalizes side engine use during landing to prevent "teetering"
- Rewards "settling" behavior (doing nothing when well-positioned)
- Progressive penalties for repeated side engine overuse
- Configurable strictness modes: gentle, moderate, strict, extreme

**Training Loop** (`train.py`)
- Sophisticated model evaluation logic that tracks:
  - True landing success rate (game's positive terminal reward)
  - Shaped vs original scores
  - Episodes since last improvement
- Conservative model saving that prevents "regressive" saves:
  - Primary criterion: landing rate improvement
  - Secondary: score improvement with same landing rate
  - Special cases: early training flexibility, milestone saves, perfect performance
- Learning rate scheduling (StepLR with gamma=0.5 every 5000 steps)
- Comprehensive logging of episode data, failures, and fuel usage
- Dynamic checkpoint intervals (10 checkpoints per training run)

### Model Evaluation Logic

The training system uses `improved_model_evaluation_logic()` in train.py:712 which implements a hierarchical evaluation:
1. **Best models**: Landing rate improvements or score improvements with same landing rate
2. **Improving models**: Early training progress or milestone checkpoints
3. Tracks `episodes_since_best` to detect training plateaus
4. Provides detailed debugging output via `debug_model_saving()`

### Key Configuration Parameters

**DQN Agent Defaults:**
- Learning rate: 0.0003 (Adam optimizer)
- Tau (soft update): 0.001
- Epsilon decay: 0.995 (0.6 when resuming training)
- Epsilon min: 0.02
- Batch size: 64
- Gamma (discount): 0.99

**Reward Shaper Modes:**
```python
# Horizontal precision (affects guidance strength)
reward_shaper.set_horizontal_precision_mode("aggressive")  # gentle, moderate, aggressive, extreme

# Landing strictness (affects landing zone thresholds)
reward_shaper.set_landing_strictness("moderate")  # gentle, moderate, strict, extreme
```

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
- `logs/` - Training session logs (JSON, CSV, debug text)
- `videos/` - Generated episode videos
- `old/` - Previous reward shaper implementations
- `tests/` - Diagnostic and debugging scripts

## Important Implementation Details

### Fuel Compensation Logic

Added in `reward_shaper.py:427`, the fuel compensation provides a small reward offset (0.05) when engines are used to counteract the environment's built-in fuel penalty. This prevents the agent from being overly penalized for necessary engine usage. The compensation is reduced by 50% for side engines in the landing zone.

### Landing Success Detection

True landing success is determined by `terminated and reward > 0` (see train.py:242). The code also tracks auxiliary metrics like both-legs-touching and one-leg-touching for debugging, but model evaluation uses the game's own success criterion.

### Failure Analysis

The training loop categorizes failures (train.py:246-263):
- `no_ground_contact` - Didn't touch ground
- `one_leg_only` - Only one leg touched
- `outside_pad` - Landed outside the pad (|x| > 0.5)
- `too_fast` - Speed > 1.0 at landing
- `other_crash` - Other crash reasons
- `timeout` - Episode didn't terminate within 1000 steps

### Model Loading Behavior

When resuming training with an existing `models/moonlander_best.pth`:
- Loads the model weights
- Resets epsilon to 0.6 (not the saved value) for continued exploration
- Creates a timestamped backup
- Establishes fresh baseline via initial evaluation (80% of initial performance)
- Prompts for number of training episodes

## Debugging & Troubleshooting

If the agent isn't learning well:
1. Check action distribution in training output - should be relatively balanced
2. Run `landing_analysis.py` to identify specific behavioral problems
3. Check logs for plateau warnings (no improvement for 3000+ episodes)
4. Verify landing zone behavior - agent should settle calmly, not oscillate
5. Compare shaped vs original rewards - large divergence may indicate overfitting to shaped rewards

If side engines are overused during landing:
- Landing zone control system should prevent this
- Check `side_engine_overuse_count` in reward shaper debug info
- Consider stricter landing mode: `reward_shaper.set_landing_strictness("strict")`
