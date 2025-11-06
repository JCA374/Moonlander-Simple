# File Organization

This document describes the purpose of each file in the repository.

## Core Training Files (DQN)

- **train.py** - Main DQN training script with reward shaping
- **dqn_agent.py** - Double DQN agent with dueling architecture
- **reward_shaper.py** - Sophisticated reward shaping system (10+ components)
- **evaluator.py** - Evaluation functions for DQN training
- **logger.py** - Training logging and diagnostics
- **play_best.py** - Visualize trained DQN models

## Genetic Algorithm Files (NEW)

- **train_ga.py** - Main GA training script using CMA-ES
- **linear_controller.py** - Simple linear policy controller (36 params)
- **ga_evaluator.py** - Fitness evaluation for GA controllers
- **cma_es_optimizer.py** - CMA-ES optimizer with simple GA fallback
- **play_ga.py** - Visualize trained GA controllers
- **plot_ga_training.py** - Plot GA training progress
- **compare_ga_dqn.py** - Compare GA vs DQN performance
- **README_GA.md** - Complete GA documentation

## Utility Files

- **landing_analysis.py** - Analyze landing behavior and patterns
- **individual_video_generator.py** - Generate videos of episodes
- **merge_movies.py** - Merge multiple videos together

## Test Files

- **test_ga_components.py** - Unit tests for GA components (works without Box2D)
- **tests/debug_analyzer.py** - Debug and diagnostic tools
- **tests/speed_diagnostics.py** - Speed and performance diagnostics

## Configuration Files

- **requirements.txt** - Python package dependencies
- **.gitignore** - Git ignore patterns
- **CLAUDE.local.md** - Local project notes (not in git)

## Directories

- **old/** - Archived old versions (in .gitignore)
- **videos/** - Generated videos (in .gitignore)
- **publish/** - Published visualizations (dqn-visualization.html)
- **ga_models/** - Saved GA models (in .gitignore)
- **__pycache__/** - Python cache (in .gitignore)

## Models and Outputs (Not in Git)

- **moonlander_best.pth** - Best DQN model
- **moonlander_improving_*.pth** - Improving DQN models
- **moonlander_checkpoint_*.pth** - DQN checkpoints
- **ga_models/ga_best.npy** - Best GA model
- **ga_models/ga_checkpoint_*.npy** - GA checkpoints
- **ga_models/training_history.json** - GA training history
- **logs/** - Training logs

## Quick Reference

### To train with DQN:
```bash
python train.py
```

### To train with GA:
```bash
python train_ga.py --generations 200
```

### To visualize trained models:
```bash
python play_best.py                  # DQN model
python play_ga.py                    # GA model
```

### To compare approaches:
```bash
python compare_ga_dqn.py --episodes 100
```

### To run tests:
```bash
python test_ga_components.py         # GA component tests
```
