import gymnasium as gym
import numpy as np
from dqn_agent import DQNAgent


def quick_evaluate(agent, episodes=5, reward_shaper=None):
    """
    Fast evaluation for training loop - returns metrics for model saving decisions
    """
    eval_env = gym.make('LunarLander-v2')

    scores = []
    original_scores = []
    true_landings = 0  # When game gives positive terminal reward
    both_legs_landings = 0  # When both legs touch (debugging metric)

    for _ in range(episodes):
        state, _ = eval_env.reset()
        if reward_shaper:
            reward_shaper.reset()

        total_shaped_reward = 0
        total_original_reward = 0

        max_steps = eval_env.spec.max_episode_steps or 1000
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = eval_env.step(
                action)
            done = terminated or truncated

            # Track original reward
            total_original_reward += reward

            # Apply reward shaping if provided
            if reward_shaper:
                shaped_reward = reward_shaper.shape_reward(
                    state, action, reward, done, step, terminated, truncated)
                total_shaped_reward += shaped_reward
            else:
                total_shaped_reward += reward

            state = next_state

            if done:
                # Check for successful landing by game's rules
                if terminated and reward > 0:
                    true_landings += 1

                # Check if both legs touched (debugging metric)
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

    return avg_shaped_score, true_landing_rate


def detailed_evaluate(model_path='moonlander_best.pth', episodes=10, render=True, reward_shaper=None):
    """
    Comprehensive evaluation with detailed analysis and visualization
    """
    try:
        env = gym.make('LunarLander-v2',
                       render_mode='human' if render else None)
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        agent = DQNAgent(state_size, action_size)
        agent.load(model_path)
        agent.epsilon = 0  # No exploration during evaluation

        print(f"🔍 Detailed Evaluation of {model_path}")
        print(f"Episodes: {episodes}, Render: {render}")
        print("="*60)

        scores = []
        original_scores = []
        episode_data = []

        for episode in range(episodes):
            state, _ = env.reset()
            if reward_shaper:
                reward_shaper.reset()

            total_shaped_reward = 0
            total_original_reward = 0
            step_count = 0
            actions_taken = []

            max_steps = env.spec.max_episode_steps or 1000
            for step in range(max_steps):
                action = agent.act(state)
                actions_taken.append(action)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                # Track rewards
                total_original_reward += reward

                if reward_shaper:
                    shaped_reward = reward_shaper.shape_reward(
                        state, action, reward, done, step, terminated, truncated)
                    total_shaped_reward += shaped_reward
                else:
                    total_shaped_reward += reward

                state = next_state
                step_count += 1

                if done:
                    break

            # Analyze episode outcome
            final_state = next_state
            x, y = final_state[0], final_state[1]
            vx, vy = final_state[2], final_state[3]
            angle = final_state[4]
            leg1, leg2 = final_state[6], final_state[7]

            final_speed = np.sqrt(vx**2 + vy**2)

            # Determine success type
            game_success = terminated and reward > 0
            both_legs_success = terminated and leg1 and leg2

            # Determine failure reason
            if not terminated:
                outcome = "TIMEOUT"
                icon = "⏰"
            elif game_success:
                if final_speed < 0.3:
                    outcome = "PERFECT LANDING"
                    icon = "🎯"
                else:
                    outcome = "ROUGH LANDING"
                    icon = "✅"
            else:
                if abs(x) > 0.5:
                    outcome = "MISSED PAD"
                    icon = "📍"
                elif final_speed > 0.8:
                    outcome = "TOO FAST"
                    icon = "💥"
                elif not (leg1 or leg2):
                    outcome = "NO CONTACT"
                    icon = "🚁"
                else:
                    outcome = "OTHER CRASH"
                    icon = "❌"

            # Action analysis
            action_counts = [actions_taken.count(i) for i in range(4)]
            main_engine_pct = action_counts[2] / len(actions_taken) * 100

            episode_info = {
                'episode': episode + 1,
                'outcome': outcome,
                'game_success': game_success,
                'both_legs_success': both_legs_success,
                'shaped_score': total_shaped_reward,
                'original_score': total_original_reward,
                'final_speed': final_speed,
                'final_position': (x, y),
                'final_angle': angle,
                'steps': step_count,
                'main_engine_pct': main_engine_pct,
                'actions': action_counts
            }
            episode_data.append(episode_info)

            scores.append(total_shaped_reward)
            original_scores.append(total_original_reward)

            # Print episode summary
            print(f"{icon} Episode {episode + 1}: {outcome}")
            print(
                f"   Scores: Shaped={total_shaped_reward:.1f}, Original={total_original_reward:.1f}")
            print(
                f"   Final: Pos=({x:.2f}, {y:.2f}), Speed={final_speed:.3f}, Angle={angle:.2f}")
            print(
                f"   Steps: {step_count}, Main Engine: {main_engine_pct:.1f}%")

        env.close()

        # Overall analysis
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)

        game_successes = sum(1 for ep in episode_data if ep['game_success'])
        both_legs_successes = sum(
            1 for ep in episode_data if ep['both_legs_success'])

        print(f"🎯 Success Metrics:")
        print(
            f"   Game Success Rate: {game_successes}/{episodes} ({game_successes/episodes*100:.1f}%)")
        print(
            f"   Both Legs Rate: {both_legs_successes}/{episodes} ({both_legs_successes/episodes*100:.1f}%)")

        print(f"\n📈 Score Analysis:")
        print(
            f"   Average Shaped Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
        print(
            f"   Average Original Score: {np.mean(original_scores):.2f} ± {np.std(original_scores):.2f}")
        print(f"   Best Score: {max(original_scores):.2f}")
        print(f"   Worst Score: {min(original_scores):.2f}")

        # Speed analysis
        final_speeds = [ep['final_speed'] for ep in episode_data]
        print(f"\n🚀 Speed Analysis:")
        print(f"   Average Final Speed: {np.mean(final_speeds):.3f}")
        print(
            f"   Speed Range: {min(final_speeds):.3f} - {max(final_speeds):.3f}")
        print(
            f"   Gentle Landings (speed < 0.3): {sum(1 for s in final_speeds if s < 0.3)}/{episodes}")

        # Behavioral analysis
        avg_main_engine = np.mean([ep['main_engine_pct']
                                  for ep in episode_data])
        print(f"\n🎮 Behavioral Analysis:")
        print(f"   Average Main Engine Usage: {avg_main_engine:.1f}%")
        print(
            f"   Average Episode Length: {np.mean([ep['steps'] for ep in episode_data]):.1f} steps")

        # Outcome breakdown
        print(f"\n🔍 Outcome Breakdown:")
        outcomes = {}
        for ep in episode_data:
            outcome = ep['outcome']
            outcomes[outcome] = outcomes.get(outcome, 0) + 1

        for outcome, count in outcomes.items():
            print(f"   {outcome}: {count}/{episodes} ({count/episodes*100:.1f}%)")

        print("="*60)

        return {
            'average_shaped_score': np.mean(scores),
            'average_original_score': np.mean(original_scores),
            'game_success_rate': game_successes / episodes,
            'both_legs_success_rate': both_legs_successes / episodes,
            'average_final_speed': np.mean(final_speeds),
            'episode_data': episode_data
        }

    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the agent first by running 'python train.py'")
        return None
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        return None


def compare_models(model_paths, episodes=10):
    """
    Compare multiple models side by side
    """
    print("🔄 COMPARING MODELS")
    print("="*60)

    results = {}

    for model_path in model_paths:
        print(f"\n📊 Evaluating {model_path}...")
        result = detailed_evaluate(model_path, episodes=episodes, render=False)
        if result:
            results[model_path] = result

    # Comparison table
    if len(results) > 1:
        print("\n📋 COMPARISON SUMMARY")
        print("="*80)
        print(
            f"{'Model':<25} {'Success%':<10} {'Avg Score':<12} {'Avg Speed':<12} {'Main Eng%':<10}")
        print("-"*80)

        for model_path, result in results.items():
            model_name = model_path.split('/')[-1][:24]  # Shorten name
            success_rate = result['game_success_rate'] * 100
            avg_score = result['average_original_score']
            avg_speed = result['average_final_speed']

            # Calculate average main engine usage from episode data
            avg_main_eng = np.mean([ep['main_engine_pct']
                                   for ep in result['episode_data']])

            print(
                f"{model_name:<25} {success_rate:<10.1f} {avg_score:<12.1f} {avg_speed:<12.3f} {avg_main_eng:<10.1f}")

    return results


def evaluate_agent(model_path='moonlander_best.pth', episodes=10, render=True, reward_shaper=None):
    """
    Main evaluation function - wrapper for detailed_evaluate for backward compatibility
    """
    return detailed_evaluate(model_path, episodes, render, reward_shaper)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "compare":
            # Compare multiple models
            models = sys.argv[2:] if len(sys.argv) > 2 else [
                'moonlander_best.pth',
                'moonlander_final.pth'
            ]
            compare_models(models)
        else:
            # Evaluate specific model
            model_path = sys.argv[1]
            detailed_evaluate(model_path)
    else:
        # Default evaluation
        detailed_evaluate()
