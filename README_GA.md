# Genetic Algorithm for Lunar Lander

This directory contains a genetic algorithm implementation for training the Lunar Lander agent using evolutionary strategies instead of deep reinforcement learning.

## Overview

The genetic algorithm approach evolves a **simple linear controller** (36 parameters) using **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy) to solve the Lunar Lander task. This provides an interesting alternative to the DQN approach used in the main training script.

## Key Differences from DQN

| Aspect | Genetic Algorithm | DQN |
|--------|------------------|-----|
| **Controller** | Linear (36 params) | Neural Network (~17K params) |
| **Learning** | Population-based evolution | Gradient-based (backprop) |
| **Interpretability** | High (inspect weights) | Low (black box) |
| **Sample Efficiency** | Lower (needs many episodes) | Higher (experience replay) |
| **Parallelization** | Easy (evaluate independently) | Hard (GPU-bound) |
| **Implementation** | Simple (~300 lines) | Complex (~1000+ lines) |

## Files

- `linear_controller.py` - Simple linear policy controller (8 states → 4 actions)
- `ga_evaluator.py` - Fitness evaluation for controllers
- `cma_es_optimizer.py` - CMA-ES optimizer with simple GA fallback
- `train_ga.py` - Main training script
- `play_ga.py` - Visualize trained controllers
- `plot_ga_training.py` - Plot training history
- `compare_ga_dqn.py` - Compare GA vs DQN performance
- `README_GA.md` - This file

## Installation

### Required Packages

```bash
pip install numpy gymnasium matplotlib
```

### Optional (for optimal CMA-ES):

```bash
pip install cma
```

If `cma` is not installed, a simple genetic algorithm fallback will be used automatically.

## Quick Start

### 1. Train a Controller

Basic training (200 generations, ~20-30 minutes):

```bash
python train_ga.py
```

Fast training (50 generations, ~5-10 minutes):

```bash
python train_ga.py --generations 50 --population 30 --episodes 5
```

Thorough training (500 generations, ~1-2 hours):

```bash
python train_ga.py --generations 500 --population 100 --episodes 20
```

With parallel evaluation (faster on multi-core CPUs):

```bash
python train_ga.py --parallel 4
```

### 2. Visualize the Trained Controller

Play 5 episodes with rendering:

```bash
python play_ga.py --model ga_models/ga_best.npy --episodes 5
```

Evaluate over 100 episodes (no rendering, faster):

```bash
python play_ga.py --model ga_models/ga_best.npy --evaluate 100 --no-render
```

Show what the controller learned:

```bash
python play_ga.py --model ga_models/ga_best.npy --describe
```

### 3. Plot Training Progress

```bash
python plot_ga_training.py --history ga_models/training_history.json
```

Save plot instead of displaying:

```bash
python plot_ga_training.py --save
```

### 4. Compare with DQN

```bash
python compare_ga_dqn.py --ga-model ga_models/ga_best.npy --dqn-model moonlander_best.pth --episodes 100
```

## How It Works

### Linear Controller

The controller is a simple linear mapping:

```
action_scores = state @ weights + bias
action = argmax(action_scores)
```

Where:
- `state` is 8-dimensional (position, velocity, angle, leg contacts)
- `weights` is an 8×4 matrix
- `bias` is a 4-dimensional vector
- Total parameters: 8×4 + 4 = 36

### CMA-ES Optimization

CMA-ES is a state-of-the-art evolutionary algorithm that:

1. **Generates** a population of candidate solutions (parameter vectors)
2. **Evaluates** each by running episodes in the environment
3. **Updates** the search distribution based on fitness
4. **Adapts** the covariance matrix to learn problem structure
5. **Repeats** until convergence or max generations

### Fitness Function

Fitness combines reward and landing success:

```python
fitness = avg_reward + (landing_rate × 100)
```

This encourages both high episode rewards and successful landings.

## Training Parameters

Key hyperparameters in `train_ga.py`:

- `--generations`: Number of evolution cycles (default: 200)
- `--population`: Population size (default: 50)
- `--episodes`: Episodes per fitness evaluation (default: 10)
- `--sigma`: Initial step size (default: 0.5)
- `--parallel`: Number of parallel workers (default: 1)

### Recommended Settings

**Quick Test:**
```bash
python train_ga.py --generations 50 --population 30 --episodes 5
```

**Standard Training:**
```bash
python train_ga.py --generations 200 --population 50 --episodes 10
```

**High Performance:**
```bash
python train_ga.py --generations 500 --population 100 --episodes 20 --parallel 4
```

## Expected Results

With standard settings (200 generations, population 50), you should see:

- **Generation 0:** ~-200 fitness, 0-10% landing rate (random)
- **Generation 50:** ~50-100 fitness, 20-40% landing rate
- **Generation 100:** ~150-200 fitness, 50-70% landing rate
- **Generation 200:** ~200-250 fitness, 70-90% landing rate

The best controllers can achieve:
- **Fitness:** 250-300
- **Landing Rate:** 80-95%
- **Average Reward:** 180-220

## Interpreting the Controller

Use the `--describe` flag to see what the controller learned:

```bash
python play_ga.py --model ga_models/ga_best.npy --describe
```

This shows:
- Bias values (baseline action preferences)
- Weight matrix (how each state influences each action)
- Most influential weights

Example insights:
- Large negative weight from "vertical velocity" to "do nothing" → fires engine when falling fast
- Large positive weight from "altitude" to "main engine" → fires more at high altitude
- Strong coupling between "angle" and side engines → corrects orientation

## Computational Cost

### Time per Generation

Depends on:
- Population size (N individuals)
- Episodes per evaluation (E episodes)
- Episode length (~200-1000 steps)

**Rough estimate:**
- Population 50, 10 episodes: ~30-60 seconds per generation
- Total for 200 generations: ~20-40 minutes

### Parallelization

Use `--parallel N` to evaluate N individuals simultaneously:

```bash
python train_ga.py --parallel 4  # 4x speedup on 4+ core CPU
```

Benefits:
- Near-linear speedup with CPU cores
- No GPU required
- Each worker is independent

## Advantages of GA Approach

1. **Simplicity:** Much simpler than DQN (no replay buffer, target networks, etc.)
2. **Interpretability:** Can inspect and understand learned weights
3. **Robustness:** Less sensitive to hyperparameters than DQN
4. **Parallelization:** Easy to distribute across CPU cores
5. **No gradients:** Works with any controller structure

## Disadvantages of GA Approach

1. **Sample inefficiency:** Needs many full episode evaluations
2. **Scalability:** Harder to scale to complex problems (but fine for LunarLander)
3. **Expressiveness:** Linear controller less expressive than neural network
4. **Exploration:** May get stuck in local optima

## Advanced Usage

### Custom Fitness Function

Edit `ga_evaluator.py` to customize fitness:

```python
# Current (balanced)
fitness = avg_reward + (landing_rate * 100)

# Prioritize landing
fitness = landing_rate * 200 + avg_reward * 0.5

# Reward only
fitness = avg_reward

# Multiplicative
fitness = avg_reward * (1 + landing_rate)
```

### Evolving Neural Networks

To evolve a small neural network instead of linear controller:

1. Create `neural_controller.py` with similar interface
2. Replace `LinearController` in `train_ga.py`
3. Adjust population size and sigma for larger parameter space

**Note:** This will be much slower due to more parameters!

### Hybrid Approach

Combine GA and DQN:

1. Use GA to find good initialization
2. Fine-tune with DQN from that starting point
3. Potentially faster convergence than DQN from scratch

## Troubleshooting

### "No improvement after many generations"

- Try increasing sigma: `--sigma 1.0`
- Increase population: `--population 100`
- More episodes per eval: `--episodes 20`

### "Training too slow"

- Reduce episodes: `--episodes 5`
- Smaller population: `--population 30`
- Use parallelization: `--parallel 4`

### "Landing rate stuck at 0%"

- Try longer training: `--generations 500`
- Increase sigma for more exploration: `--sigma 1.0`
- Check that gymnasium is working: `python -c "import gymnasium; env = gymnasium.make('LunarLander-v2')"`

### "Import error: cma module not found"

This is fine! The code will automatically fall back to a simple GA. For better performance:

```bash
pip install cma
```

## Research Questions

This implementation enables exploring:

1. How does linear controller compare to neural network?
2. Is CMA-ES competitive with DQN on this task?
3. What is the minimum parameter count needed to solve LunarLander?
4. Can we interpret what the algorithm learned?
5. How does parallelization scaling compare between GA and DQN?

## Citation

If you use this code for research, please cite:

- CMA-ES: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial"
- Lunar Lander: Brockman et al. (2016). "OpenAI Gym"

## Contributing

Suggestions for improvements:

- [ ] Add more controller types (neural network, fuzzy logic)
- [ ] Implement other evolutionary algorithms (NEAT, Genetic Programming)
- [ ] Add curriculum learning (start easy, increase difficulty)
- [ ] Multi-objective optimization (reward + fuel efficiency)
- [ ] Transfer learning to other environments

## License

Same as the main project.
