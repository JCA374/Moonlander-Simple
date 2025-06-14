import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import shutil
from dqn_agent import DQNAgent
from logger import TrainingLogger
from evaluator import quick_evaluate
from reward_shaper import RewardShaper
import torch

torch.set_num_threads(4)


def improved_model_evaluation_logic(episode, eval_score, true_landing_rate,
                                    best_eval_score, best_landing_rate, episodes_since_best):
    """
    Improved model saving logic that's more conservative and meaningful.
    Returns: (should_save, save_type, improvement_reason, new_best_score, new_best_rate, reset_counter)
    """

    is_better_model = False
    improvement_reason = ""
    save_type = "best"
    new_best_score = best_eval_score
    new_best_rate = best_landing_rate
    reset_counter = False

    # === PRIMARY CRITERIA: TRUE IMPROVEMENTS ===

    # 1. Landing rate improvement (most important)
    if true_landing_rate > best_landing_rate:
        is_better_model = True
        save_type = "best"
        improvement_reason = f"Landing rate: {best_landing_rate*100:.1f}% -> {true_landing_rate*100:.1f}%"
        new_best_rate = true_landing_rate
        new_best_score = eval_score
        reset_counter = True

    # 2. Score improvement with same landing rate (secondary)
    elif true_landing_rate == best_landing_rate and eval_score > best_eval_score + 20:
        is_better_model = True
        save_type = "best"
        improvement_reason = f"Score: {best_eval_score:.1f} -> {eval_score:.1f} (same landing rate)"
        new_best_score = eval_score
        reset_counter = True

    # === SPECIAL CASES ===

    # 3. Early training flexibility (only first 1000 episodes)
    elif episode < 1000 and true_landing_rate >= best_landing_rate * 0.9 and eval_score > best_eval_score:
        is_better_model = True
        save_type = "improving"
        improvement_reason = f"Early improvement: Score {best_eval_score:.1f} -> {eval_score:.1f}"
        # Don't update best metrics for "improving" saves

    # 4. Milestone saves (much more conservative)
    elif episodes_since_best > 2000 and true_landing_rate >= 0.8 and eval_score > 500:
        # Only save if it's been a REALLY long time and performance is genuinely good
        is_better_model = True
        save_type = "improving"
        improvement_reason = f"Milestone save after {episodes_since_best} episodes: {true_landing_rate*100:.1f}% success, score {eval_score:.1f}"
        # Don't update best metrics for "improving" saves

    # 5. Perfect performance checkpoint (even if score lower)
    elif true_landing_rate == 1.0 and eval_score > 0:
        is_better_model = True
        save_type = "improving" if true_landing_rate <= best_landing_rate else "best"
        improvement_reason = f"Perfect landing rate achieved: 100% success, score {eval_score:.1f}"
        # Update best metrics if it's also a score improvement
        if eval_score > best_eval_score or true_landing_rate > best_landing_rate:
            save_type = "best"
            new_best_score = max(eval_score, best_eval_score)
            new_best_rate = true_landing_rate
            reset_counter = True

    return is_better_model, save_type, improvement_reason, new_best_score, new_best_rate, reset_counter


def debug_model_saving(episode, eval_score, true_landing_rate, best_eval_score, best_landing_rate, episodes_since_best):
    """Debug why models are or aren't being saved - MUST MATCH actual logic"""
    print(f"🔍 Model Saving Debug (Episode {episode}):")
    print(
        f"   Current: Score={eval_score:.2f}, Landing Rate={true_landing_rate*100:.1f}%")
    print(
        f"   Best:    Score={best_eval_score:.2f}, Landing Rate={best_landing_rate*100:.1f}%")
    print(f"   Episodes since best: {episodes_since_best}")

    # Use the SAME logic as the actual saving function
    should_save, save_type, improvement_reason, _, _, _ = improved_model_evaluation_logic(
        episode, eval_score, true_landing_rate, best_eval_score, best_landing_rate, episodes_since_best
    )

    if should_save:
        print(f"   ✅ WILL SAVE ({save_type.upper()}): {improvement_reason}")
    else:
        print(f"   ❌ NO SAVE: No significant improvement")

    return should_save, save_type, improvement_reason


def train_moonlander():
    from gymnasium.wrappers import TimeLimit
    base_env = gym.make('LunarLander-v2')
    env = TimeLimit(base_env, max_episode_steps=1000)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # pass in soft update tau parameter
    agent = DQNAgent(state_size, action_size, tau=0.001)

    # Create models folder if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)

    # Load the best model if it exists
    best_model_path = os.path.join(models_dir, 'moonlander_best.pth')
    if os.path.exists(best_model_path):
        print(f"🔄 Loading previous model from {best_model_path}")
        agent.load(best_model_path)
        agent.epsilon = 0.6
        agent.epsilon_decay = 0.9995

        # Create backup of loaded model
        timestamp = int(time.time())
        backup_path = os.path.join(
            models_dir, f'moonlander_backup_{timestamp}.pth')
        shutil.copy(best_model_path, backup_path)
        print(f"📁 Backup saved to {backup_path}")

        episodes_input = input(
            "How many episodes to train? (default 200000): ").strip()
        episodes = int(episodes_input) if episodes_input else 200000

        # START FRESH - Don't use old model's peak performance as baseline
        best_eval_score = float('-inf')
        best_landing_rate = 0.0
        print("✨ Starting fresh evaluation criteria to allow for adaptation")

    else:
        print("🆕 No previous model found, starting from scratch")
        episodes = 25000
        best_eval_score = float('-inf')
        best_landing_rate = 0.0

    # Add step learning-rate scheduler
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(agent.optimizer, step_size=5000, gamma=0.5)

    # ENHANCED: Set up reward shaper with better horizontal precision
    reward_shaper = RewardShaper(enable_approach_tracking=True)
    reward_shaper.set_horizontal_precision_mode(
        "aggressive")  # Better horizontal control

    logger = TrainingLogger()

    # Log training configuration
    logger.log_config({
        "episodes": episodes,
        "state_size": state_size,
        "action_size": action_size,
        "learning_rate": agent.learning_rate,
        "epsilon_start": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "reward_shaping": True,
        "horizontal_precision_mode": "aggressive",
        "resumed_training": os.path.exists(best_model_path)
    })

    scores = []
    scores_window = []

    # FIXED: Proper tracking of episodes since best model
    episodes_since_best = 0

    # Add early evaluation to establish realistic baseline
    print("🎯 Running initial evaluation to establish baseline...")
    current_epsilon = agent.epsilon
    agent.epsilon = 0
    initial_score, initial_rate = quick_evaluate(
        agent, episodes=10, reward_shaper=reward_shaper)
    agent.epsilon = current_epsilon

    print(
        f"📊 Initial performance: Score={initial_score:.2f}, Landing rate={initial_rate*100:.1f}%")

    # Set initial baseline
    if initial_rate > 0:
        best_landing_rate = initial_rate * 0.8
        best_eval_score = initial_score * 0.8
    else:
        best_landing_rate = 0.0
        best_eval_score = initial_score

    # Calculate checkpoint interval
    checkpoint_interval = max(1, episodes // 10)
    print(f"💾 Will save checkpoints every {checkpoint_interval} episodes")

    for episode in range(episodes):
        state, _ = env.reset()
        reward_shaper.reset()
        total_reward = 0
        original_reward = 0
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
                state, action, reward, done, step, terminated, truncated)

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

        if episode % 100 == 0:
            agent.update_target_network()

        # CORRECTED LANDING SUCCESS DETECTION
        true_success = terminated and reward > 0
        both_legs_touching = terminated and next_state[6] and next_state[7]
        one_leg_touching = terminated and (next_state[6] or next_state[7])

        # Failure analysis
        failure_reason = "success"
        if terminated and reward <= 0:
            x_pos = next_state[0]
            speed = np.sqrt(next_state[2]**2 + next_state[3]**2)

            if not (next_state[6] or next_state[7]):
                failure_reason = "no_ground_contact"
            elif not both_legs_touching:
                failure_reason = "one_leg_only"
            elif abs(x_pos) > 0.5:
                failure_reason = "outside_pad"
            elif speed > 1.0:
                failure_reason = "too_fast"
            else:
                failure_reason = "other_crash"
        elif not terminated:
            failure_reason = "timeout"

        # Log comprehensive data
        logger.log_episode(episode, total_reward, agent.epsilon, step + 1, {
            "original_reward": original_reward,
            "landing_success": true_success,
            "both_legs_touching": both_legs_touching,
            "one_leg_touching": one_leg_touching,
            "failure_reason": failure_reason,
            "final_x_position": next_state[0],
            "final_y_position": next_state[1],
            "final_speed": np.sqrt(next_state[2]**2 + next_state[3]**2),
            "final_angle": next_state[4],
            "fuel_used": fuel_used,
            "loss": loss if loss is not None else 0,
            "q_variance": q_variance if q_variance is not None else 0
        })

        scores.append(total_reward)
        scores_window.append(total_reward)

        if len(scores_window) > 100:
            scores_window.pop(0)

        # Progress reporting
        if episode % 50 == 0:
            avg_score = np.mean(scores_window)
            avg_original = np.mean(
                [ep["original_reward"] for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):]])

            true_landings = len([ep for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):]
                                 if ep.get("landing_success", False)])

            both_legs_landings = len([ep for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):]
                                      if ep.get("both_legs_touching", False)])

            recent_episodes = logger.log_data["episodes"][-min(
                100, len(logger.log_data["episodes"])):]
            outside_pad_failures = len(
                [ep for ep in recent_episodes if ep.get("failure_reason") == "outside_pad"])

            print(f"Episode {episode}: Score {avg_score:.2f} | Original {avg_original:.2f} | TRUE {true_landings}% | Legs {both_legs_landings}% | OutPad {outside_pad_failures} | ε {agent.epsilon:.3f}")

        # FIXED EVALUATION: Use improved logic with proper tracking
        if episode > 0 and episode % 200 == 0:
            current_epsilon = agent.epsilon
            agent.epsilon = 0

            eval_score, true_landing_rate = quick_evaluate(
                agent, reward_shaper=reward_shaper)
            agent.epsilon = current_epsilon

            # Debug the decision process (now matches actual logic)
            should_save, save_type, improvement_reason = debug_model_saving(
                episode, eval_score, true_landing_rate, best_eval_score, best_landing_rate, episodes_since_best
            )

            # Use improved model evaluation logic
            is_better_model, save_type, improvement_reason, new_best_score, new_best_rate, reset_counter = \
                improved_model_evaluation_logic(
                    episode, eval_score, true_landing_rate,
                    best_eval_score, best_landing_rate, episodes_since_best
                )

            if is_better_model:
                # Choose filename based on save type
                if save_type == "best":
                    save_path = os.path.join(models_dir, 'moonlander_best.pth')
                    # Backup existing best model
                    if os.path.exists(save_path):
                        backup_path = os.path.join(
                            models_dir, f'moonlander_best_backup_{episode}.pth')
                        shutil.copy(save_path, backup_path)

                    # Update best metrics only for "best" saves
                    best_landing_rate = new_best_rate
                    best_eval_score = new_best_score
                    if reset_counter:
                        episodes_since_best = 0

                else:  # save_type == "improving"
                    save_path = os.path.join(
                        models_dir, f'moonlander_improving_{episode}.pth')
                    # Don't update best metrics for "improving" saves

                agent.save(save_path)
                logger.log_milestone(
                    episode, f"NEW {save_type.upper()} MODEL! {improvement_reason}")
                print(
                    f"🎯 [Episode {episode}] NEW {save_type.upper()} MODEL! {improvement_reason}")
                print(f"💾 Saved to: {save_path}")

            else:
                # FIXED: Properly increment episodes_since_best
                episodes_since_best += 200

            print(
                f"📊 Eval {episode}: Score {eval_score:.2f} | TRUE {true_landing_rate*100:.1f}% | Best {best_landing_rate*100:.1f}%")

            # Warning if no improvement for too long
            if episodes_since_best > 3000:
                print(
                    f"⚠️  No improvement for {episodes_since_best} episodes - consider adjusting hyperparameters")

                # Suggest actions
                if true_landing_rate < 0.5:
                    print("💡 Low landing rate - consider strengthening reward shaping")
                elif true_landing_rate > 0.9 and eval_score < best_eval_score:
                    print(
                        "💡 Good landing rate but low score - may be overfit to shaped rewards")

        # Save checkpoint based on dynamic interval
        if episode > 0 and episode % checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                models_dir, f'moonlander_checkpoint_{episode}.pth')
            agent.save(checkpoint_path)
            progress = (episode / episodes) * 100
            logger.log_milestone(
                episode, f"Checkpoint saved at episode {episode} ({progress:.1f}% complete)")
            print(f"💾 Checkpoint {episode} ({progress:.1f}%)")

        # Save at episode 0 to have initial state
        if episode == 0:
            checkpoint_path = os.path.join(
                models_dir, f'moonlander_checkpoint_{episode}.pth')
            agent.save(checkpoint_path)
            print(f"💾 Initial checkpoint saved")

    # Save the final model
    logger.log_milestone(
        episodes-1, "Training completed. Saving final model...")
    agent.save(os.path.join(models_dir, 'moonlander_final.pth'))

    # Ensure we have a best model saved
    if best_eval_score == float('-inf'):
        agent.save(os.path.join(models_dir, 'moonlander_best.pth'))
        logger.log_milestone(
            episodes-1, "No evaluation performed, saving current model as best")

    env.close()

    # Save logs and print summary
    logger.save_log()
    logger.print_summary()

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(scores)
    plt.title('Shaped Reward Progress')
    plt.xlabel('Episode')
    plt.ylabel('Shaped Score')

    plt.subplot(2, 2, 2)
    original_scores = [ep["original_reward"]
                       for ep in logger.log_data["episodes"]]
    plt.plot(original_scores)
    plt.title('Original Reward Progress')
    plt.xlabel('Episode')
    plt.ylabel('Original Score')

    plt.subplot(2, 2, 3)
    landing_rate = []
    window_size = 100
    for i in range(len(logger.log_data["episodes"])):
        start_idx = max(0, i - window_size + 1)
        window_episodes = logger.log_data["episodes"][start_idx:i+1]
        landings = sum(1 for ep in window_episodes if ep.get(
            "landing_success", False))
        landing_rate.append(landings / len(window_episodes) * 100)
    plt.plot(landing_rate)
    plt.title('True Landing Success Rate (%)')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.subplot(2, 2, 4)
    fuel_usage = [ep["fuel_used"] for ep in logger.log_data["episodes"]]
    plt.plot(fuel_usage)
    plt.title('Fuel Usage per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Fuel Used')

    plt.tight_layout()
    plt.show()

    return agent


# Helper function to clean up old improving models
def cleanup_improving_models(models_dir='models', keep_every_n=5):
    """
    Clean up excessive 'improving' models, keeping only every Nth one
    """
    import glob
    import re

    improving_files = glob.glob(os.path.join(
        models_dir, 'moonlander_improving_*.pth'))

    if not improving_files:
        print("No improving models found to clean up")
        return

    # Extract episode numbers and sort
    model_info = []
    for file_path in improving_files:
        filename = os.path.basename(file_path)
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
            os.remove(file_path)
            print(f"Deleted: moonlander_improving_{episode}.pth")
        except:
            print(f"Could not delete: {file_path}")

    print(f"Cleanup complete! Kept models: {[ep for ep, _ in to_keep]}")


if __name__ == "__main__":
    # Optional: Clean up old improving models first
    cleanup_choice = input(
        "Clean up old 'improving' models first? (y/n): ").strip().lower()
    if cleanup_choice == 'y':
        cleanup_improving_models(keep_every_n=5)

    agent = train_moonlander()
