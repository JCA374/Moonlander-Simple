import gymnasium as gym
import numpy as np
import time
from dqn_agent import DQNAgent

def play_best_model(model_path='moonlander_best.pth', episodes=5, delay=0.02):
    """
    Play the best trained model with visual rendering
    
    Args:
        model_path: Path to the saved model
        episodes: Number of episodes to play
        delay: Delay between frames (seconds) for better visualization
    """
    try:
        env = gym.make('LunarLander-v2', render_mode='human')
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        
        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
        agent.epsilon = 0  # No exploration, only exploitation
        
        print(f"Playing {episodes} episodes with the best model...")
        print("Actions: 0=Do nothing, 1=Fire left, 2=Fire main, 3=Fire right")
        print("Goal: Land between the flags with minimal fuel usage\n")
        
        total_scores = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            step_count = 0
            
            print(f"Episode {episode + 1}/{episodes}")
            
            while True:
                action = agent.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                state = next_state
                total_reward += reward
                step_count += 1
                
                # Add small delay for better visualization
                time.sleep(delay)
                
                if done:
                    break
                    
                # Safety limit
                if step_count > 1000:
                    break
                    
            total_scores.append(total_reward)
            
            if terminated:
                if total_reward >= 200:
                    print(f"✅ Successful landing! Score: {total_reward:.2f}")
                elif total_reward >= 100:
                    print(f"🟡 Decent landing. Score: {total_reward:.2f}")
                else:
                    print(f"❌ Crashed or poor landing. Score: {total_reward:.2f}")
            else:
                print(f"⏰ Episode timed out. Score: {total_reward:.2f}")
                
            print(f"Steps taken: {step_count}\n")
            
        env.close()
        
        print("=== Performance Summary ===")
        print(f"Average Score: {np.mean(total_scores):.2f}")
        print(f"Best Score: {max(total_scores):.2f}")
        print(f"Worst Score: {min(total_scores):.2f}")
        print(f"Success Rate: {len([s for s in total_scores if s >= 200])/len(total_scores)*100:.1f}%")
        
    except FileNotFoundError:
        print("❌ Model file not found!")
        print("Please train the model first by running: python train.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you have installed all dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    play_best_model()