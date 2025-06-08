import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os
from dqn_agent import DQNAgent
from improved_reward_shaper import ImprovedRewardShaper
from logger import TrainingLogger
from evaluator import quick_evaluate

def train_moonlander():
    env = gym.make('LunarLander-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    
    # Load the best model if it exists
    best_model_path = 'moonlander_best.pth'
    if os.path.exists(best_model_path):
        print(f"Loading previous best model from {best_model_path}")
        agent.load(best_model_path)
    else:
        print("No previous best model found, starting from scratch")
    
    reward_shaper = ImprovedRewardShaper()
    logger = TrainingLogger()
    
    # Log training configuration
    logger.log_config({
        "episodes": 5000,
        "state_size": state_size,
        "action_size": action_size,
        "learning_rate": agent.learning_rate,
        "epsilon_start": agent.epsilon,
        "epsilon_min": agent.epsilon_min,
        "epsilon_decay": agent.epsilon_decay,
        "reward_shaping": True
    })
    
    episodes = 25000  # Longer training for better convergence
    scores = []
    scores_window = []
    
    # Best model tracking
    best_eval_score = float('-inf')
    best_original_score = float('-inf')
    episodes_since_best = 0
    
    for episode in range(episodes):
        state, _ = env.reset()
        reward_shaper.reset()
        total_reward = 0
        original_reward = 0
        fuel_used = 0
        hover_penalty = 0
        actions_taken = []
        
        for step in range(500):
            action = agent.act(state)
            actions_taken.append(action)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Track original reward and fuel usage
            original_reward += reward
            if action != 0:
                fuel_used += 1
                
            # Apply reward shaping
            shaped_reward = reward_shaper.shape_reward(state, action, reward, done, step)
            
            # No hover penalty tracking for simple shaper
            
            agent.remember(state, action, shaped_reward, next_state, done)
            state = next_state
            total_reward += shaped_reward
            
            if done:
                break
                
        # Training step with logging
        loss, q_variance = agent.replay()
        if loss is not None:
            logger.log_training_step(loss, q_variance)
        
        if episode % 100 == 0:  # Less frequent target updates for stability
            agent.update_target_network()
            
        # Check if landing was successful
        landing_success = terminated and next_state[6] and next_state[7]  # Both legs touching
        
        # Log episode data
        logger.log_episode(episode, total_reward, agent.epsilon, step + 1, {
            "original_reward": original_reward,
            "landing_success": landing_success,
            "hover_penalty": 0,  # Not tracked in simple shaper
            "fuel_used": fuel_used,
            "actions_taken": actions_taken,
            "loss": loss if loss is not None else 0,
            "q_variance": q_variance if q_variance is not None else 0
        })
            
        scores.append(total_reward)
        scores_window.append(total_reward)
        
        if len(scores_window) > 100:
            scores_window.pop(0)
            
        if episode % 50 == 0:
            avg_score = np.mean(scores_window)
            avg_original = np.mean([ep["original_reward"] for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):]])
            successful_landings = len([ep for ep in logger.log_data["episodes"][-min(100, len(logger.log_data["episodes"])):] if ep.get("landing_success", False)])
            
            print(f"Episode {episode}, Shaped Score: {avg_score:.2f}, Original: {avg_original:.2f}, Landings: {successful_landings}/100, Epsilon: {agent.epsilon:.3f}")
            
        # Evaluate and save best model periodically
        if episode > 0 and episode % 200 == 0:
            # Quick evaluation
            current_epsilon = agent.epsilon
            agent.epsilon = 0  # No exploration during evaluation
            
            eval_score, landing_rate = quick_evaluate(agent)
            agent.epsilon = current_epsilon  # Restore epsilon
            
            # Check if this is the best model so far
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                episodes_since_best = 0
                agent.save('moonlander_best.pth')
                logger.log_milestone(episode, f"NEW BEST MODEL! Eval score: {eval_score:.2f}, Landing rate: {landing_rate*100:.1f}%")
            else:
                episodes_since_best += 200
                
            print(f"[Eval] Episode {episode}: Score {eval_score:.2f}, Landings {landing_rate*100:.1f}%, Best: {best_eval_score:.2f}")
        
        # Save checkpoint periodically
        if episode > 0 and episode % 2000 == 0:
            agent.save(f'moonlander_checkpoint_{episode}.pth')
            logger.log_milestone(episode, f"Checkpoint saved at episode {episode}")
            
        if np.mean(scores_window) >= 200:
            logger.log_milestone(episode, f"Environment solved! Average score: {np.mean(scores_window):.2f}")
            agent.save('moonlander_dqn.pth')
            break
            
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
    plt.title('Landing Success Rate (%)')
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