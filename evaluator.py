import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from improved_reward_shaper import ImprovedRewardShaper

def quick_evaluate(agent, episodes=5):
    """
    Quick evaluation of agent performance
    Returns average shaped reward over episodes
    """
    eval_env = gym.make('LunarLander-v2')
    reward_shaper = ImprovedRewardShaper()
    scores = []
    landings = 0
    
    for _ in range(episodes):
        state, _ = eval_env.reset()
        reward_shaper.reset()
        total_reward = 0
        
        max_steps = eval_env.spec.max_episode_steps
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            
            # Apply same reward shaping as training
            shaped_reward = reward_shaper.shape_reward(state, action, reward, done, step, terminated, truncated)
            
            state = next_state
            total_reward += shaped_reward
            
            if done:
                if terminated and next_state[6] and next_state[7]:  # Successful landing
                    landings += 1
                break
                
        scores.append(total_reward)
        
    eval_env.close()
    
    avg_score = np.mean(scores)
    landing_rate = landings / episodes
    
    return avg_score, landing_rate