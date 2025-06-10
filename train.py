import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from dqn_agent import DQNAgent
from precision_landing_shaper import create_precision_shaper
from logger import TrainingLogger
from evaluator import quick_evaluate

import torch
torch.set_num_threads(4)   # or torch.get_num_threads() // 2


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
    best_model_path = 'moonlander_best.pth'
    if os.path.exists(best_model_path):
        print(f"Loading previous best model from {best_model_path}")
        agent.load(best_model_path)
        # Force exploration of new strategies
        agent.epsilon = 0.6
        agent.epsilon_decay = 0.9995
        episodes_input = input("How many episodes to train? (default 25000): ").strip()
        episodes = int(episodes_input) if episodes_input else 25000
        
        # Get baseline evaluation of loaded model to prevent immediate overwrite
        print("Evaluating loaded model to establish baseline...")
        current_epsilon = agent.epsilon
        agent.epsilon = 0  # No exploration during evaluation
        baseline_score, baseline_rate = quick_evaluate(agent)
        agent.epsilon = current_epsilon  # Restore epsilon
        print(f"Loaded model baseline: Score {baseline_score:.2f}, Landing rate: {baseline_rate*100:.1f}%")
    else:
        print("No previous best model found, starting from scratch")
        episodes = 25000
        baseline_score = float('-inf')  # No baseline for new training
        baseline_rate = 0.0  # No baseline landing rate for new training
    
    # add step learning-rate scheduler - halve LR every 5000 episodes  
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(agent.optimizer, step_size=5000, gamma=0.5)
    
    reward_shaper = create_precision_shaper("precision")
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
        "reward_shaping": True
    })
    scores = []
    scores_window = []
    
    # Best model tracking
    best_eval_score = baseline_score  # Use baseline from loaded model or -inf for new training
    best_landing_rate = baseline_rate  # Track true landing rate
    episodes_since_best = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        reward_shaper.reset()
        total_reward = 0
        original_reward = 0
        fuel_used = 0
        hover_penalty = 0
        actions_taken = []
        
        # let the wrapper (or default) dictate max steps
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
            shaped_reward = reward_shaper.shape_reward(state, action, reward, done, step, terminated, truncated)
            
            # No hover penalty tracking for simple shaper
            
            agent.remember(state, action, shaped_reward, next_state, done)
            state = next_state
            total_reward += shaped_reward
            
            if done:
                break
                
        # Training step with logging
        loss, q_variance = agent.replay()
        # Log training stats only every 10 episodes
        if loss is not None and episode % 10 == 0:
            logger.log_training_step(loss, q_variance)
        # step the LR scheduler once per episode
        scheduler.step()

        
        if episode % 100 == 0:  # Less frequent target updates for stability
            agent.update_target_network()
            
        # CORRECTED LANDING SUCCESS DETECTION
        
        # 1. TRUE SUCCESS: What the environment actually rewards (use this for metrics)
        true_success = terminated and reward > 0
        
        # 2. DEBUGGING METRICS: Useful for analysis but not training decisions
        both_legs_touching = terminated and next_state[6] and next_state[7]
        one_leg_touching = terminated and (next_state[6] or next_state[7])
        
        # 3. FAILURE ANALYSIS: Understand why landings fail
        failure_reason = "success"
        if terminated and reward <= 0:
            x_pos = next_state[0]
            speed = np.sqrt(next_state[2]**2 + next_state[3]**2)
            
            if not (next_state[6] or next_state[7]):
                failure_reason = "no_ground_contact"
            elif not both_legs_touching:
                failure_reason = "one_leg_only"
            elif abs(x_pos) > 0.5:  # Outside landing pad
                failure_reason = "outside_pad"
            elif speed > 1.0:
                failure_reason = "too_fast"
            else:
                failure_reason = "other_crash"
        elif not terminated:
            failure_reason = "timeout"
            
        # 4. LOG COMPREHENSIVE DATA
        logger.log_episode(episode, total_reward, agent.epsilon, step + 1, {
            "original_reward": original_reward,
            "true_success": true_success,              # ✅ REAL success metric
            "both_legs_touching": both_legs_touching,  # For debugging
            "one_leg_touching": one_leg_touching,      # For debugging
            "failure_reason": failure_reason,          # Why it failed
            "final_x_position": next_state[0],         # Where it landed
            "final_y_position": next_state[1],         # Height at landing
            "final_speed": np.sqrt(next_state[2]**2 + next_state[3]**2),
            "final_angle": next_state[4],              # Landing angle
            "fuel_used": fuel_used,
            "loss": loss if loss is not None else 0,
            "q_variance": q_variance if q_variance is not None else 0
        })
            
        scores.append(total_reward)
        scores_window.append(total_reward)
        
        if len(scores_window) > 100:
            scores_window.pop(0)
            
        # 5. CORRECT SUCCESS TRACKING: Use true_success for all metrics
        if episode % 50 == 0:
            avg_score = np.mean(scores_window)
            avg_original = np.mean([ep["original_reward"] for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):]])
            
            # ✅ CORRECTED: Use true_success instead of landing_success
            true_landings = len([ep for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):] 
                               if ep.get("true_success", False)])
            
            # Also track the debugging metrics to see the gap
            both_legs_landings = len([ep for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):] 
                                    if ep.get("both_legs_touching", False)])
            
            # Failure analysis
            recent_episodes = logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):] 
            outside_pad_failures = len([ep for ep in recent_episodes if ep.get("failure_reason") == "outside_pad"])
            
            print(f"Episode {episode}, Shaped Score: {avg_score:.2f}, Original: {avg_original:.2f}")
            print(f"  TRUE Landings: {true_landings}/100 ({true_landings}%)")  # ✅ Real success rate
            print(f"  Both legs touching: {both_legs_landings}/100 ({both_legs_landings}%)")  # Debugging
            print(f"  Outside pad failures: {outside_pad_failures}/100")  # Key insight
            print(f"  Epsilon: {agent.epsilon:.3f}")
            
        # 6. EVALUATION: Use true_success for model saving decisions
        if episode % 200 == 0:  # Evaluate every 200 episodes including episode 0
            print(f"Running evaluation at episode {episode}...")
            current_epsilon = agent.epsilon
            agent.epsilon = 0
            
            eval_score, true_landing_rate = quick_evaluate(agent)  # Make sure quick_evaluate uses true_success too
            agent.epsilon = current_epsilon
            
            # Save best model based on TRUE landing rate, not both_legs_touching
            if true_landing_rate > best_landing_rate:  # Track best TRUE landing rate
                print(f"NEW BEST MODEL! Previous best: {best_landing_rate*100:.1f}%, Current: {true_landing_rate*100:.1f}%")
                best_landing_rate = true_landing_rate
                episodes_since_best = 0
                agent.save('moonlander_best.pth')
                logger.log_milestone(episode, f"NEW BEST MODEL! True landing rate: {true_landing_rate*100:.1f}%")
            else:
                episodes_since_best += 200
                
            print(f"[Eval] Episode {episode}: Score {eval_score:.2f}, TRUE Landings: {true_landing_rate*100:.1f}%, Best Rate: {best_landing_rate*100:.1f}%")
        
        # Save checkpoint periodically
        if episode > 0 and episode % 2000 == 0:
            checkpoint_path = os.path.join(models_dir, f'moonlander_checkpoint_{episode}.pth')
            agent.save(checkpoint_path)
            logger.log_milestone(episode, f"Checkpoint saved at episode {episode}")
            
            
    # Save the final model
    if episode == episodes - 1:
        logger.log_milestone(episode, "Training completed. Saving final model...")
        agent.save('moonlander_final.pth')
        
    # Ensure we have a best model saved
    if best_eval_score == float('-inf'):
        agent.save('moonlander_best.pth')
        logger.log_milestone(episode, "No evaluation performed, saving current model as best")
            
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
    original_scores = [ep["original_reward"] for ep in logger.log_data["episodes"]]
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
        landings = sum(1 for ep in window_episodes if ep.get("landing_success", False))
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

if __name__ == "__main__":
    agent = train_moonlander()