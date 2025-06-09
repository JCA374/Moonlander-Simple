import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from improved_reward_shaper import ImprovedRewardShaper

def evaluate_agent(model_path='moonlander_dqn.pth', episodes=10, render=True):
    env = gym.make('LunarLander-v2', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    agent.epsilon = 0
    
    reward_shaper = ImprovedRewardShaper()
    scores = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        reward_shaper.reset()
        total_reward = 0
        
        for step in range(500):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Apply same reward shaping as training
            shaped_reward = reward_shaper.shape_reward(state, action, reward, done, step, terminated, truncated)
            
            state = next_state
            total_reward += shaped_reward
            
            if done:
                break
                
        scores.append(total_reward)
        print(f"Episode {episode + 1}: Score = {total_reward:.2f}")
        
    env.close()
    
    print(f"\nAverage Score over {episodes} episodes: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    
    return scores

if __name__ == "__main__":
    try:
        evaluate_agent()
    except FileNotFoundError:
        print("Model file not found. Please train the agent first by running 'python train.py'")