import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent
from improved_reward_shaper import ImprovedRewardShaper


def quick_evaluate(agent, episodes=5):
    """
    Fixed evaluation that properly detects successful landings
    """
    eval_env = gym.make('LunarLander-v2')
    reward_shaper = ImprovedRewardShaper()

    scores = []
    original_scores = []
    true_landings = 0  # When game gives positive terminal reward
    both_legs_landings = 0  # When both legs touch

    for _ in range(episodes):
        state, _ = eval_env.reset()
        reward_shaper.reset()
        total_shaped_reward = 0
        total_original_reward = 0

        max_steps = eval_env.spec.max_episode_steps
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(
                action)
            done = terminated or truncated

            # Track original reward
            total_original_reward += reward

            # Apply reward shaping
            shaped_reward = reward_shaper.shape_reward(
                state, action, reward, done, step, terminated, truncated)
            total_shaped_reward += shaped_reward

            state = next_state

            if done:
                # Check for successful landing by game's rules
                if terminated and reward > 0:
                    true_landings += 1

                # Check if both legs touched (your current metric)
                if terminated and next_state[6] and next_state[7]:
                    both_legs_landings += 1
                break

        scores.append(total_shaped_reward)
        original_scores.append(total_original_reward)

    eval_env.close()

    avg_shaped_score = np.mean(scores)
    avg_original_score = np.mean(original_scores)
    true_landing_rate = true_landings / episodes
    both_legs_rate = both_legs_landings / episodes

    # Return shaped score and true landing rate for compatibility
    # But also print more info
    print(
        f"  Original score: {avg_original_score:.2f}, True landings: {true_landing_rate*100:.1f}%, Both legs: {both_legs_rate*100:.1f}%")

    return avg_shaped_score, true_landing_rate
