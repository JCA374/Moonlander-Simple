import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from improved_reward_shaper import ImprovedRewardShaper
import pandas as pd


class LandingDebugger:
    """Comprehensive debugging tool for the moonlander agent"""

    def __init__(self, model_path='moonlander_best.pth'):
        self.env = gym.make('LunarLander-v2')
        self.agent = DQNAgent(8, 4)
        try:
            self.agent.load(model_path)
            self.agent.epsilon = 0  # No exploration for debugging
            print(f"✅ Loaded model: {model_path}")
        except FileNotFoundError:
            print(f"❌ Model not found: {model_path}")
            return

        self.reward_shaper = ImprovedRewardShaper()

    def analyze_episode(self, render=False, verbose=True):
        """Run detailed analysis of a single episode"""
        if render:
            env = gym.make('LunarLander-v2', render_mode='human')
        else:
            env = self.env

        state, _ = env.reset()
        self.reward_shaper.reset()

        # Data collection
        states = [state.copy()]
        actions = []
        rewards = []
        shaped_rewards = []
        step_data = []

        total_original = 0
        total_shaped = 0

        for step in range(1000):
            action = self.agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Apply reward shaping
            shaped_reward = self.reward_shaper.shape_reward(
                state, action, reward, done, step, terminated, truncated
            )

            # Collect data
            actions.append(action)
            rewards.append(reward)
            shaped_rewards.append(shaped_reward)
            states.append(next_state.copy())

            # Detailed step analysis
            x, y, vx, vy, angle, ang_vel, leg1, leg2 = next_state
            step_info = {
                'step': step,
                'action': action,
                'x': x, 'y': y, 'vx': vx, 'vy': vy,
                'angle': angle, 'ang_vel': ang_vel,
                'leg1': leg1, 'leg2': leg2,
                'reward': reward,
                'shaped_reward': shaped_reward,
                'speed': np.sqrt(vx**2 + vy**2),
                'distance_to_pad': np.sqrt(x**2 + y**2),
                'terminated': terminated,
                'truncated': truncated
            }
            step_data.append(step_info)

            total_original += reward
            total_shaped += shaped_reward
            state = next_state

            if done:
                break

        if render:
            env.close()

        # Analysis
        final_state = states[-1]
        landing_analysis = self._analyze_landing(step_data[-1], total_original)

        if verbose:
            self._print_episode_summary(
                step_data, landing_analysis, total_original, total_shaped)

        return {
            'step_data': step_data,
            'landing_analysis': landing_analysis,
            'total_original': total_original,
            'total_shaped': total_shaped,
            'episode_length': len(step_data)
        }

    def _analyze_landing(self, final_step, total_reward):
        """Analyze why landing succeeded or failed"""
        x, y = final_step['x'], final_step['y']
        vx, vy = final_step['vx'], final_step['vy']
        angle = final_step['angle']
        leg1, leg2 = final_step['leg1'], final_step['leg2']
        terminated = final_step['terminated']

        analysis = {
            'game_success': terminated and final_step['reward'] > 0,
            'both_legs_touching': leg1 and leg2,
            'position_ok': abs(x) < 0.5,  # Within landing pad
            'altitude_ok': y >= 0,  # Above ground
            'speed_ok': np.sqrt(vx**2 + vy**2) < 0.5,  # Slow enough
            'angle_ok': abs(angle) < 0.2,  # Upright enough
            'final_reward': final_step['reward'],
            'total_reward': total_reward
        }

        # Determine failure reason
        if not analysis['game_success']:
            if not terminated:
                analysis['failure_reason'] = "Timeout (too slow/hovering)"
            elif not analysis['position_ok']:
                analysis['failure_reason'] = f"Landed outside pad (x={x:.3f})"
            elif not analysis['speed_ok']:
                analysis['failure_reason'] = f"Too fast (speed={np.sqrt(vx**2 + vy**2):.3f})"
            elif not analysis['angle_ok']:
                analysis['failure_reason'] = f"Wrong angle ({angle:.3f} rad)"
            else:
                analysis['failure_reason'] = f"Unknown (final_reward={final_step['reward']:.2f})"
        else:
            analysis['failure_reason'] = "Success!"

        return analysis

    def _print_episode_summary(self, step_data, landing_analysis, total_original, total_shaped):
        """Print detailed episode summary"""
        print("\n" + "="*60)
        print("🔍 EPISODE DEBUG ANALYSIS")
        print("="*60)

        # Episode basics
        print(f"📊 Episode Length: {len(step_data)} steps")
        print(f"💰 Original Reward: {total_original:.2f}")
        print(f"🎨 Shaped Reward: {total_shaped:.2f}")
        print(
            f"📈 Reward Ratio: {total_shaped/total_original:.1f}x" if total_original != 0 else "∞x")

        # Landing analysis
        print(f"\n🛬 Landing Analysis:")
        for key, value in landing_analysis.items():
            if key == 'failure_reason':
                print(f"   ❌ Reason: {value}")
            elif isinstance(value, bool):
                icon = "✅" if value else "❌"
                print(f"   {icon} {key.replace('_', ' ').title()}: {value}")
            else:
                print(f"   📊 {key.replace('_', ' ').title()}: {value:.3f}")

        # Action distribution
        actions = [step['action'] for step in step_data]
        action_counts = {i: actions.count(i) for i in range(4)}
        action_names = {0: "Do Nothing", 1: "Left Engine",
                        2: "Main Engine", 3: "Right Engine"}

        print(f"\n🎮 Action Distribution:")
        for action, count in action_counts.items():
            pct = count / len(actions) * 100
            print(f"   {action_names[action]}: {count} ({pct:.1f}%)")

        # Critical moments
        print(f"\n🎯 Critical Moments:")

        # When did it get close to the pad?
        close_steps = [s for s in step_data if s['distance_to_pad'] < 0.5]
        if close_steps:
            first_close = close_steps[0]['step']
            print(f"   First close to pad: Step {first_close}")

        # Speed analysis near ground
        low_steps = [s for s in step_data if s['y'] < 0.3]
        if low_steps:
            avg_speed_low = np.mean([s['speed'] for s in low_steps])
            print(f"   Average speed when low: {avg_speed_low:.3f}")

        # Reward breakdown in final moments
        final_10_steps = step_data[-10:]
        final_original = sum(s['reward'] for s in final_10_steps)
        final_shaped = sum(s['shaped_reward'] for s in final_10_steps)
        print(
            f"   Final 10 steps - Original: {final_original:.2f}, Shaped: {final_shaped:.2f}")

    def compare_reward_functions(self, episodes=5):
        """Compare original vs shaped rewards to identify problems"""
        print("\n🔬 REWARD FUNCTION COMPARISON")
        print("="*50)

        results = []
        for ep in range(episodes):
            result = self.analyze_episode(verbose=False)
            results.append(result)

            landing = result['landing_analysis']
            print(f"Episode {ep+1}: Original={result['total_original']:.1f}, "
                  f"Shaped={result['total_shaped']:.1f}, Success={landing['game_success']}")

        # Statistics
        original_scores = [r['total_original'] for r in results]
        shaped_scores = [r['total_shaped'] for r in results]
        success_rate = sum(r['landing_analysis']['game_success']
                           for r in results) / episodes

        print(f"\n📈 Summary:")
        print(
            f"   Original: {np.mean(original_scores):.1f} ± {np.std(original_scores):.1f}")
        print(
            f"   Shaped: {np.mean(shaped_scores):.1f} ± {np.std(shaped_scores):.1f}")
        print(f"   Success Rate: {success_rate*100:.1f}%")
        print(
            f"   Reward Amplification: {np.mean(shaped_scores)/np.mean(original_scores):.1f}x")

        return results

    def test_specific_scenarios(self):
        """Test edge cases and specific scenarios"""
        print("\n🧪 SCENARIO TESTING")
        print("="*30)

        # Test what happens with different final states
        scenarios = [
            ("Perfect Landing", [0, 0.1, -0.05, -0.1, 0, 0, 1, 1]),
            ("Fast Landing", [0, 0.1, -0.2, -0.8, 0, 0, 1, 1]),
            ("Off-Center", [0.6, 0.1, -0.05, -0.1, 0, 0, 1, 1]),
            ("Tilted", [0, 0.1, -0.05, -0.1, 0.3, 0, 1, 1]),
        ]

        for name, final_state in scenarios:
            # Simulate final step reward
            shaped_reward = self.reward_shaper.shape_reward(
                final_state, 0, 100, True, 200, True, False  # Assume success
            )
            print(f"   {name}: Original=+100, Shaped={shaped_reward:.1f}")


def main():
    debugger = LandingDebugger()

    print("🚀 MoonLander Debugging Suite")
    print("1. Single episode analysis (with visualization)")
    print("2. Reward function comparison")
    print("3. Scenario testing")

    choice = input("\nChoose analysis (1-3, or 'all'): ").strip().lower()

    if choice in ['1', 'all']:
        debugger.analyze_episode(render=True)

    if choice in ['2', 'all']:
        debugger.compare_reward_functions()

    if choice in ['3', 'all']:
        debugger.test_specific_scenarios()


if __name__ == "__main__":
    main()
