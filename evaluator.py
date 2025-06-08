import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent

def quick_evaluate(agent, episodes=5):
    """
    Quick evaluation of agent performance
    Returns average original reward over episodes
    """
    eval_env = gym.make('LunarLander-v2')
    scores = []
    landings = 0
    
    for _ in range(episodes):
        state, _ = eval_env.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            
            state = next_state
            total_reward += reward
            
            if done:
                if terminated and next_state[6] and next_state[7]:  # Successful landing
                    landings += 1
                break
                
        scores.append(total_reward)
        
    eval_env.close()
    
    avg_score = np.mean(scores)
    landing_rate = landings / episodes
    
    return avg_score, landing_rate