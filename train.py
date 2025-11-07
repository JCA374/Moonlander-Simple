"""
Training script for Lunar Lander DQN agent.

This script orchestrates the training loop, including:
- Environment setup
- Agent initialization
- Reward shaping
- Model evaluation and saving
- Comprehensive logging
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import StepLR
from gymnasium.wrappers import TimeLimit
from typing import Tuple, Optional

from dqn_agent import DQNAgent
from reward_shaper import RewardShaper
from logger import TrainingLogger
from evaluator import quick_evaluate
from model_manager import ModelManager
from config import MoonlanderConfig, DQNConfig, RewardShaperConfig, TrainingConfig
from constants import (
    ENVIRONMENT_NAME,
    FailureReason,
    MAX_EPISODE_STEPS,
    StateIndex,
    LANDING_PAD_X_MIN,
    LANDING_PAD_X_MAX,
    MAX_SAFE_LANDING_SPEED
)


def setup_environment(max_steps: int = MAX_EPISODE_STEPS) -> gym.Env:
    """
    Create and configure the training environment.

    Args:
        max_steps: Maximum steps per episode

    Returns:
        Configured Gymnasium environment
    """
    base_env = gym.make(ENVIRONMENT_NAME)
    env = TimeLimit(base_env, max_episode_steps=max_steps)
    return env


def initialize_training(
    config: Optional[MoonlanderConfig] = None
) -> Tuple[gym.Env, DQNAgent, RewardShaper, TrainingLogger, ModelManager, TrainingConfig]:
    """
    Initialize all training components.

    Args:
        config: Master configuration (uses defaults if None)

    Returns:
        Tuple of (env, agent, reward_shaper, logger, model_manager, training_config)
    """
    if config is None:
        config = MoonlanderConfig()

    # Set PyTorch threads
    torch.set_num_threads(config.training.torch_num_threads)

    # Create environment
    env = setup_environment(config.training.max_steps_per_episode)
    state_size = env.observation_space.shape[0]
    action_size = env.observation_space.n

    # Create agent with config
    agent = DQNAgent(state_size, action_size, config=config.dqn)

    # Create reward shaper with config
    reward_shaper = RewardShaper(config=config.reward_shaper)
    reward_shaper.set_horizontal_precision_mode("aggressive")

    # Create logger
    logger = TrainingLogger(log_dir=config.training.logs_dir)

    # Create model manager
    model_manager = ModelManager(models_dir=config.training.models_dir)

    # Log training configuration
    logger.log_config({
        "episodes": "TBD",  # Will be set later
        "state_size": state_size,
        "action_size": action_size,
        "dqn_config": config.dqn.__dict__,
        "reward_shaper_config": {
            "horizontal_precision_mode": "aggressive",
            "landing_strictness": "moderate"
        },
        "training_config": config.training.__dict__
    })

    return env, agent, reward_shaper, logger, model_manager, config.training


def load_existing_model(
    agent: DQNAgent,
    model_manager: ModelManager,
    training_config: TrainingConfig
) -> Tuple[bool, int]:
    """
    Load existing model if available.

    Args:
        agent: DQN agent to load into
        model_manager: Model manager for loading
        training_config: Training configuration

    Returns:
        Tuple of (model_loaded, num_episodes)
    """
    model_loaded, backup_path = model_manager.load_best_model(agent)

    if model_loaded:
        episodes_input = input(
            f"How many episodes to train? (default {training_config.episodes_resume_training}): "
        ).strip()
        episodes = int(episodes_input) if episodes_input else training_config.episodes_resume_training

        # Start fresh - don't use old model's peak performance as baseline
        model_manager.best_eval_score = float('-inf')
        model_manager.best_landing_rate = 0.0
        print("✨ Starting fresh evaluation criteria to allow for adaptation")

        return True, episodes

    # No existing model
    print("🆕 No previous model found, starting from scratch")
    return False, training_config.episodes_new_training


def establish_baseline(
    agent: DQNAgent,
    reward_shaper: RewardShaper,
    model_manager: ModelManager,
    training_config: TrainingConfig
) -> None:
    """
    Establish baseline performance via initial evaluation.

    Args:
        agent: DQN agent
        reward_shaper: Reward shaper
        model_manager: Model manager
        training_config: Training configuration
    """
    print("🎯 Running initial evaluation to establish baseline...")
    current_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation

    initial_score, initial_rate = quick_evaluate(
        agent,
        episodes=training_config.initial_eval_episodes,
        reward_shaper=reward_shaper
    )

    agent.epsilon = current_epsilon  # Restore epsilon

    print(f"📊 Initial performance: Score={initial_score:.2f}, "
          f"Landing rate={initial_rate*100:.1f}%")

    # Set initial baseline
    if initial_rate > 0:
        model_manager.best_landing_rate = initial_rate * training_config.baseline_multiplier
        model_manager.best_eval_score = initial_score * training_config.baseline_multiplier
    else:
        model_manager.best_landing_rate = 0.0
        model_manager.best_eval_score = initial_score


def classify_failure(
    terminated: bool,
    reward: float,
    next_state: np.ndarray
) -> str:
    """
    Classify the reason for episode failure.

    Args:
        terminated: Whether episode terminated naturally
        reward: Final reward
        next_state: Final state

    Returns:
        Failure reason string
    """
    if not terminated:
        return FailureReason.TIMEOUT

    if reward > 0:
        return FailureReason.SUCCESS

    # Extract state components
    x = next_state[StateIndex.X_POSITION]
    speed = np.sqrt(next_state[StateIndex.X_VELOCITY]**2 +
                    next_state[StateIndex.Y_VELOCITY]**2)
    leg1 = next_state[StateIndex.LEFT_LEG_CONTACT]
    leg2 = next_state[StateIndex.RIGHT_LEG_CONTACT]

    # Classify failure
    if not (leg1 or leg2):
        return FailureReason.NO_GROUND_CONTACT
    elif not (leg1 and leg2):
        return FailureReason.ONE_LEG_ONLY
    elif abs(x) > LANDING_PAD_X_MAX:
        return FailureReason.OUTSIDE_PAD
    elif speed > MAX_SAFE_LANDING_SPEED:
        return FailureReason.TOO_FAST
    else:
        return FailureReason.OTHER_CRASH


def run_episode(
    env: gym.Env,
    agent: DQNAgent,
    reward_shaper: RewardShaper,
    scheduler: StepLR,
    logger: TrainingLogger,
    episode: int
) -> dict:
    """
    Run a single training episode.

    Args:
        env: Environment
        agent: DQN agent
        reward_shaper: Reward shaper
        scheduler: Learning rate scheduler
        logger: Training logger
        episode: Episode number

    Returns:
        Dictionary with episode data
    """
    state, _ = env.reset()
    reward_shaper.reset()

    total_reward = 0.0
    original_reward = 0.0
    fuel_used = 0
    actions_taken = []

    max_steps = env.spec.max_episode_steps
    for step in range(max_steps):
        action = agent.act(state)
        actions_taken.append(action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Track original reward and fuel usage
        original_reward += reward
        if action != 0:
            fuel_used += 1

        # Apply reward shaping
        shaped_reward = reward_shaper.shape_reward(
            state, action, reward, done, step, terminated, truncated
        )

        agent.remember(state, action, shaped_reward, next_state, done)
        state = next_state
        total_reward += shaped_reward

        if done:
            break

    # Training step with logging
    loss, q_variance = agent.replay()
    if loss is not None and episode % 10 == 0:
        logger.log_training_step(loss, q_variance)

    scheduler.step()

    # Update target network periodically
    if episode % 100 == 0:
        agent.update_target_network()

    # Classify landing success and failure
    true_success = terminated and reward > 0
    both_legs_touching = terminated and next_state[StateIndex.LEFT_LEG_CONTACT] and next_state[StateIndex.RIGHT_LEG_CONTACT]
    one_leg_touching = terminated and (next_state[StateIndex.LEFT_LEG_CONTACT] or next_state[StateIndex.RIGHT_LEG_CONTACT])

    failure_reason = classify_failure(terminated, reward, next_state)

    # Prepare episode data
    episode_data = {
        "original_reward": original_reward,
        "landing_success": true_success,
        "both_legs_touching": both_legs_touching,
        "one_leg_touching": one_leg_touching,
        "failure_reason": failure_reason,
        "final_x_position": next_state[StateIndex.X_POSITION],
        "final_y_position": next_state[StateIndex.Y_POSITION],
        "final_speed": np.sqrt(next_state[StateIndex.X_VELOCITY]**2 +
                              next_state[StateIndex.Y_VELOCITY]**2),
        "final_angle": next_state[StateIndex.ANGLE],
        "fuel_used": fuel_used,
        "loss": loss if loss is not None else 0,
        "q_variance": q_variance if q_variance is not None else 0
    }

    # Log episode
    logger.log_episode(episode, total_reward, agent.epsilon, step + 1, episode_data)

    return episode_data


def print_progress(
    episode: int,
    logger: TrainingLogger,
    scores_window: list
) -> None:
    """
    Print training progress.

    Args:
        episode: Current episode
        logger: Training logger
        scores_window: Recent scores for averaging
    """
    avg_score = np.mean(scores_window)

    recent_episodes = logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):]

    avg_original = np.mean([ep["original_reward"] for ep in recent_episodes])
    true_landings = len([ep for ep in recent_episodes if ep.get("landing_success", False)])
    both_legs_landings = len([ep for ep in recent_episodes if ep.get("both_legs_touching", False)])
    outside_pad_failures = len([ep for ep in recent_episodes if ep.get("failure_reason") == FailureReason.OUTSIDE_PAD])

    epsilon = logger.log_data["episodes"][-1]["epsilon"] if logger.log_data["episodes"] else 0

    print(f"Episode {episode}: Score {avg_score:.2f} | Original {avg_original:.2f} | "
          f"TRUE {true_landings}% | Legs {both_legs_landings}% | "
          f"OutPad {outside_pad_failures} | ε {epsilon:.3f}")


def evaluate_and_save(
    episode: int,
    agent: DQNAgent,
    reward_shaper: RewardShaper,
    model_manager: ModelManager,
    logger: TrainingLogger,
    training_config: TrainingConfig
) -> None:
    """
    Evaluate agent and potentially save model.

    Args:
        episode: Current episode
        agent: DQN agent
        reward_shaper: Reward shaper
        model_manager: Model manager
        logger: Training logger
        training_config: Training configuration
    """
    current_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation

    eval_score, true_landing_rate = quick_evaluate(
        agent,
        episodes=training_config.eval_episodes,
        reward_shaper=reward_shaper
    )

    agent.epsilon = current_epsilon  # Restore epsilon

    # Debug the decision process
    model_manager.debug_model_saving(episode, eval_score, true_landing_rate)

    # Evaluate and potentially save
    result = model_manager.evaluate_model(episode, eval_score, true_landing_rate)

    if result.should_save:
        model_manager.save_model(agent, result, episode, logger)
    else:
        model_manager.update_episodes_since_best(training_config.evaluation_interval)

    print(f"📊 Eval {episode}: Score {eval_score:.2f} | "
          f"TRUE {true_landing_rate*100:.1f}% | "
          f"Best {model_manager.best_landing_rate*100:.1f}%")

    # Warning if no improvement for too long
    if model_manager.episodes_since_best > training_config.plateau_warning_episodes:
        print(f"⚠️  No improvement for {model_manager.episodes_since_best} episodes - "
              f"consider adjusting hyperparameters")

        # Suggest actions
        if true_landing_rate < 0.5:
            print("💡 Low landing rate - consider strengthening reward shaping")
        elif true_landing_rate > 0.9 and eval_score < model_manager.best_eval_score:
            print("💡 Good landing rate but low score - may be overfit to shaped rewards")


def train_moonlander(config: Optional[MoonlanderConfig] = None) -> DQNAgent:
    """
    Main training function for Lunar Lander DQN agent.

    Args:
        config: Master configuration (uses defaults if None)

    Returns:
        Trained DQN agent
    """
    # Initialize all components
    env, agent, reward_shaper, logger, model_manager, training_config = initialize_training(config)

    # Load existing model or start fresh
    model_loaded, episodes = load_existing_model(agent, model_manager, training_config)

    # Update logger with episode count
    logger.log_data["config"]["episodes"] = episodes
    logger.log_data["config"]["resumed_training"] = model_loaded

    # Add learning rate scheduler
    scheduler = StepLR(
        agent.optimizer,
        step_size=training_config.lr_scheduler_step_size,
        gamma=training_config.lr_scheduler_gamma
    )

    # Establish baseline performance
    establish_baseline(agent, reward_shaper, model_manager, training_config)

    # Calculate checkpoint interval
    checkpoint_interval = max(1, episodes // training_config.checkpoint_count)
    print(f"💾 Will save checkpoints every {checkpoint_interval} episodes")

    # Training loop
    scores_window = []

    for episode in range(episodes):
        # Run episode
        episode_data = run_episode(env, agent, reward_shaper, scheduler, logger, episode)

        scores_window.append(logger.log_data["episodes"][-1]["score"])
        if len(scores_window) > 100:
            scores_window.pop(0)

        # Progress reporting
        if episode % training_config.progress_report_interval == 0:
            print_progress(episode, logger, scores_window)

        # Evaluation and model saving
        if episode > 0 and episode % training_config.evaluation_interval == 0:
            evaluate_and_save(episode, agent, reward_shaper, model_manager, logger, training_config)

        # Save checkpoint
        if episode > 0 and episode % checkpoint_interval == 0:
            checkpoint_path = model_manager.save_checkpoint(agent, episode)
            progress = (episode / episodes) * 100
            print(f"💾 Checkpoint {episode} ({progress:.1f}%)")

        # Save at episode 0 to have initial state
        if episode == 0:
            model_manager.save_checkpoint(agent, episode)
            print(f"💾 Initial checkpoint saved")

    # Save final model
    model_manager.save_final_model(agent, logger)

    # Ensure we have a best model saved
    model_manager.ensure_best_model_exists(agent)

    env.close()

    # Save logs and print summary
    logger.save_log()
    logger.print_summary()

    # Plot results
    plot_training_results(logger)

    return agent


def plot_training_results(logger: TrainingLogger) -> None:
    """
    Plot training results.

    Args:
        logger: Training logger with episode data
    """
    episodes_data = logger.log_data["episodes"]
    scores = [ep["score"] for ep in episodes_data]

    plt.figure(figsize=(12, 8))

    # Shaped reward progress
    plt.subplot(2, 2, 1)
    plt.plot(scores)
    plt.title('Shaped Reward Progress')
    plt.xlabel('Episode')
    plt.ylabel('Shaped Score')

    # Original reward progress
    plt.subplot(2, 2, 2)
    original_scores = [ep["original_reward"] for ep in episodes_data]
    plt.plot(original_scores)
    plt.title('Original Reward Progress')
    plt.xlabel('Episode')
    plt.ylabel('Original Score')

    # Landing success rate
    plt.subplot(2, 2, 3)
    landing_rate = []
    window_size = 100
    for i in range(len(episodes_data)):
        start_idx = max(0, i - window_size + 1)
        window_episodes = episodes_data[start_idx:i+1]
        landings = sum(1 for ep in window_episodes if ep.get("landing_success", False))
        landing_rate.append(landings / len(window_episodes) * 100)
    plt.plot(landing_rate)
    plt.title('True Landing Success Rate (%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    # Fuel usage
    plt.subplot(2, 2, 4)
    fuel_usage = [ep["fuel_used"] for ep in episodes_data]
    plt.plot(fuel_usage)
    plt.title('Fuel Usage per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Fuel Used')

    plt.tight_layout()
    plt.show()


def cleanup_improving_models(models_dir: str = 'models', keep_every_n: int = 5) -> None:
    """
    Clean up excessive 'improving' models, keeping only every Nth one.

    Args:
        models_dir: Directory containing models
        keep_every_n: Keep every Nth model (default 5)
    """
    import glob
    import re
    from pathlib import Path

    models_path = Path(models_dir)
    improving_files = list(models_path.glob('moonlander_improving_*.pth'))

    if not improving_files:
        print("No improving models found to clean up")
        return

    # Extract episode numbers and sort
    model_info = []
    for file_path in improving_files:
        filename = file_path.name
        match = re.search(r'improving_(\d+)', filename)
        if match:
            episode = int(match.group(1))
            model_info.append((episode, file_path))

    model_info.sort()  # Sort by episode number

    print(f"Found {len(model_info)} improving models")

    # Keep every Nth model
    to_keep = []
    to_delete = []

    for i, (episode, file_path) in enumerate(model_info):
        if i % keep_every_n == 0:
            to_keep.append((episode, file_path))
        else:
            to_delete.append((episode, file_path))

    print(f"Keeping {len(to_keep)} models, deleting {len(to_delete)} models")

    for episode, file_path in to_delete:
        try:
            file_path.unlink()
            print(f"Deleted: moonlander_improving_{episode}.pth")
        except Exception as e:
            print(f"Could not delete: {file_path} - {e}")

    print(f"Cleanup complete! Kept models: {[ep for ep, _ in to_keep]}")


if __name__ == "__main__":
    # Optional: Clean up old improving models first
    cleanup_choice = input(
        "Clean up old 'improving' models first? (y/n): "
    ).strip().lower()

    if cleanup_choice == 'y':
        cleanup_improving_models(keep_every_n=5)

    # Train the agent
    agent = train_moonlander()
