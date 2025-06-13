import gymnasium as gym
import numpy as np
import time
import os
import glob
import re
from datetime import datetime
from dqn_agent import DQNAgent
from reward_shaper import RewardShaper


class ModelSelector:
    """Enhanced model selection and playing interface"""

    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = self.discover_models()

    def discover_models(self):
        """Find all available models and extract metadata"""
        if not os.path.exists(self.models_dir):
            print(f"❌ Models directory '{self.models_dir}' not found!")
            return []

        model_files = glob.glob(os.path.join(self.models_dir, '*.pth'))
        models = []

        for file_path in model_files:
            filename = os.path.basename(file_path)
            model_info = self.parse_model_filename(filename, file_path)
            models.append(model_info)

        # Sort models by type and episode number
        models.sort(key=lambda x: (x['type_priority'], x['episode']))
        return models

    def parse_model_filename(self, filename, file_path):
        """Extract information from model filename"""
        base_name = filename.replace('.pth', '')

        # Get file modification time
        mod_time = os.path.getmtime(file_path)
        mod_date = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M')

        # Get file size
        file_size = os.path.getsize(file_path) / 1024  # KB

        model_info = {
            'filename': filename,
            'path': file_path,
            'modified': mod_date,
            'size_kb': file_size,
            'episode': 0,
            'type': 'unknown',
            'type_priority': 99
        }

        # Parse different model types
        if filename == 'moonlander_best.pth':
            model_info.update({
                'type': 'best',
                'type_priority': 0,
                'description': 'Best performing model overall'
            })
        elif filename == 'moonlander_final.pth':
            model_info.update({
                'type': 'final',
                'type_priority': 1,
                'description': 'Final model from training session'
            })
        elif 'improving' in filename:
            # Extract episode number from improving models
            match = re.search(r'improving_(\d+)', filename)
            episode = int(match.group(1)) if match else 0
            model_info.update({
                'type': 'improving',
                'type_priority': 2,
                'episode': episode,
                'description': f'Good performance model from episode {episode}'
            })
        elif 'checkpoint' in filename:
            # Extract episode number from checkpoints
            match = re.search(r'checkpoint_(\d+)', filename)
            episode = int(match.group(1)) if match else 0
            model_info.update({
                'type': 'checkpoint',
                'type_priority': 3,
                'episode': episode,
                'description': f'Training checkpoint from episode {episode}'
            })
        elif 'backup' in filename:
            # Extract episode number from backups
            match = re.search(r'backup_(\d+)', filename)
            episode = int(match.group(1)) if match else 0
            model_info.update({
                'type': 'backup',
                'type_priority': 4,
                'episode': episode,
                'description': f'Backup model from episode {episode}'
            })
        else:
            model_info['description'] = 'Unknown model type'

        return model_info

    def display_models(self):
        """Display all available models in a nice format"""
        if not self.models:
            print("❌ No models found!")
            return False

        print("\n" + "="*80)
        print("🚀 AVAILABLE MOONLANDER MODELS")
        print("="*80)

        current_type = None
        for i, model in enumerate(self.models):
            # Add section headers for different model types
            if model['type'] != current_type:
                current_type = model['type']
                type_icons = {
                    'best': '🏆',
                    'final': '🏁',
                    'improving': '📈',
                    'checkpoint': '💾',
                    'backup': '🔄',
                    'unknown': '❓'
                }
                type_names = {
                    'best': 'BEST MODELS',
                    'final': 'FINAL MODELS',
                    'improving': 'IMPROVING MODELS',
                    'checkpoint': 'CHECKPOINT MODELS',
                    'backup': 'BACKUP MODELS',
                    'unknown': 'OTHER MODELS'
                }
                icon = type_icons.get(model['type'], '❓')
                name = type_names.get(model['type'], 'UNKNOWN')
                print(f"\n{icon} {name}:")
                print("-" * 40)

            # Display model info
            print(f"{i+1:2d}. {model['filename']}")
            print(f"    📝 {model['description']}")
            print(f"    📅 Modified: {model['modified']}")
            print(f"    📊 Size: {model['size_kb']:.1f} KB")

        print("\n" + "="*80)
        return True

    def select_model(self):
        """Interactive model selection"""
        if not self.display_models():
            return None

        while True:
            try:
                print("\n🎮 Model Selection Options:")
                print("• Enter model number (1-{})".format(len(self.models)))
                print("• Type 'best' for best model")
                print("• Type 'latest' for most recently modified")
                print("• Type 'quit' to exit")

                choice = input("\nSelect model: ").strip().lower()

                if choice == 'quit':
                    return None
                elif choice == 'best':
                    # Find the best model
                    best_models = [
                        m for m in self.models if m['type'] == 'best']
                    if best_models:
                        return best_models[0]
                    else:
                        print("❌ No 'best' model found!")
                        continue
                elif choice == 'latest':
                    # Find most recently modified
                    latest_model = max(
                        self.models, key=lambda x: os.path.getmtime(x['path']))
                    return latest_model
                else:
                    # Try to parse as number
                    model_num = int(choice)
                    if 1 <= model_num <= len(self.models):
                        return self.models[model_num - 1]
                    else:
                        print(
                            f"❌ Invalid number! Please enter 1-{len(self.models)}")

            except ValueError:
                print(
                    "❌ Invalid input! Please enter a number, 'best', 'latest', or 'quit'")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                return None


class EnhancedModelPlayer:
    """Enhanced model player with detailed analysis"""

    def __init__(self):
        self.selector = ModelSelector()

    def analyze_episode(self, episode_data):
        """Analyze a single episode and return detailed stats"""
        states = episode_data['states']
        actions = episode_data['actions']
        final_state = states[-1]

        # Extract trajectory data
        positions = [(s[0], s[1]) for s in states]
        speeds = [np.hypot(s[2], s[3]) for s in states]

        # Final landing analysis
        x, y, vx, vy, angle = final_state[:5]
        leg1, leg2 = final_state[6], final_state[7]
        final_speed = np.hypot(vx, vy)

        # Determine landing quality
        if episode_data['terminated'] and episode_data['reward'] > 0:
            if final_speed < 0.25:
                landing_quality = "Perfect"
                quality_icon = "🎯"
            elif final_speed < 0.4:
                landing_quality = "Excellent"
                quality_icon = "✨"
            elif final_speed < 0.6:
                landing_quality = "Good"
                quality_icon = "✅"
            else:
                landing_quality = "Rough"
                quality_icon = "⚠️"
        elif episode_data['terminated']:
            landing_quality = "Crashed"
            quality_icon = "💥"
        else:
            landing_quality = "Timeout"
            quality_icon = "⏰"

        # Action analysis
        action_counts = [actions.count(i) for i in range(4)]
        action_names = ['Do Nothing', 'Left Engine',
                        'Main Engine', 'Right Engine']
        fuel_efficiency = (action_counts[0] / len(actions)) * 100

        # Flight path analysis
        max_altitude = max(s[1] for s in states)
        min_distance = min(abs(s[0]) for s in states)

        return {
            'landing_quality': landing_quality,
            'quality_icon': quality_icon,
            'final_speed': final_speed,
            'final_position': (x, y),
            'final_angle': angle,
            'both_legs': leg1 and leg2,
            'fuel_efficiency': fuel_efficiency,
            'max_altitude': max_altitude,
            'min_distance': min_distance,
            'episode_length': len(actions),
            'action_distribution': dict(zip(action_names, action_counts))
        }

    def play_model(self, model_info, episodes=5, delay=0.02, detailed_analysis=True):
        """Play selected model with enhanced analysis"""
        print(f"\n🚀 Loading model: {model_info['filename']}")
        print(f"📝 {model_info['description']}")

        try:
            # Setup environment and agent
            env = gym.make('LunarLander-v2', render_mode='human')
            state_size = env.observation_space.shape[0]
            action_size = env.action_space.n

            agent = DQNAgent(state_size, action_size)
            agent.load(model_info['path'])
            agent.epsilon = 0  # No exploration

            print(f"✅ Model loaded successfully!")
            print(f"🎮 Playing {episodes} episodes...\n")

            episode_results = []

            for episode in range(episodes):
                print(f"{'='*60}")
                print(f"🎯 Episode {episode + 1}/{episodes}")
                print(f"{'='*60}")

                # Run episode
                state, _ = env.reset()
                states = [state.copy()]
                actions = []
                total_reward = 0

                for step in range(1000):  # Max steps
                    action = agent.act(state)
                    actions.append(action)

                    next_state, reward, terminated, truncated, _ = env.step(
                        action)
                    done = terminated or truncated

                    states.append(next_state.copy())
                    total_reward += reward
                    state = next_state

                    time.sleep(delay)

                    if done:
                        break

                # Store episode data
                episode_data = {
                    'episode': episode + 1,
                    'states': states,
                    'actions': actions,
                    'total_reward': total_reward,
                    'terminated': terminated,
                    'reward': reward,  # Final step reward
                    'steps': len(actions)
                }

                # Analyze episode
                if detailed_analysis:
                    analysis = self.analyze_episode(episode_data)
                    episode_data['analysis'] = analysis

                    # Print episode summary
                    print(
                        f"\n{analysis['quality_icon']} Landing Quality: {analysis['landing_quality']}")
                    print(f"💰 Total Reward: {total_reward:.1f}")
                    print(f"🚀 Final Speed: {analysis['final_speed']:.3f}")
                    print(
                        f"📍 Final Position: ({analysis['final_position'][0]:.2f}, {analysis['final_position'][1]:.2f})")
                    print(f"📐 Final Angle: {analysis['final_angle']:.2f} rad")
                    print(
                        f"🦵 Both Legs: {'Yes' if analysis['both_legs'] else 'No'}")
                    print(
                        f"⚡ Fuel Efficiency: {analysis['fuel_efficiency']:.1f}% coasting")
                    print(f"📏 Steps Taken: {analysis['episode_length']}")
                else:
                    # Simple summary
                    if terminated and reward > 0:
                        print(
                            f"✅ Successful landing! Reward: {total_reward:.1f}")
                    elif terminated:
                        print(f"❌ Crashed. Reward: {total_reward:.1f}")
                    else:
                        print(f"⏰ Timeout. Reward: {total_reward:.1f}")

                episode_results.append(episode_data)

                if episode < episodes - 1:
                    input("\nPress Enter for next episode...")

            env.close()

            # Overall summary
            self.print_session_summary(model_info, episode_results)

        except FileNotFoundError:
            print(f"❌ Model file not found: {model_info['path']}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")

    def print_session_summary(self, model_info, episode_results):
        """Print comprehensive session summary"""
        print("\n" + "="*80)
        print("📊 SESSION SUMMARY")
        print("="*80)

        print(f"🤖 Model: {model_info['filename']}")
        print(f"📝 Description: {model_info['description']}")
        print(f"📅 Modified: {model_info['modified']}")

        # Calculate statistics
        total_episodes = len(episode_results)
        rewards = [ep['total_reward'] for ep in episode_results]

        successful_episodes = sum(1 for ep in episode_results
                                  if ep['terminated'] and ep['reward'] > 0)
        crashed_episodes = sum(1 for ep in episode_results
                               if ep['terminated'] and ep['reward'] <= 0)
        timeout_episodes = sum(
            1 for ep in episode_results if not ep['terminated'])

        print(f"\n🎯 Performance Metrics:")
        print(
            f"   Success Rate: {successful_episodes}/{total_episodes} ({successful_episodes/total_episodes*100:.1f}%)")
        print(
            f"   Crash Rate: {crashed_episodes}/{total_episodes} ({crashed_episodes/total_episodes*100:.1f}%)")
        print(
            f"   Timeout Rate: {timeout_episodes}/{total_episodes} ({timeout_episodes/total_episodes*100:.1f}%)")

        print(f"\n💰 Reward Statistics:")
        print(f"   Average Reward: {np.mean(rewards):.2f}")
        print(f"   Best Reward: {max(rewards):.2f}")
        print(f"   Worst Reward: {min(rewards):.2f}")
        print(f"   Std Deviation: {np.std(rewards):.2f}")

        # Detailed analysis if available
        if episode_results and 'analysis' in episode_results[0]:
            analyses = [ep['analysis'] for ep in episode_results]

            # Landing quality distribution
            quality_counts = {}
            for analysis in analyses:
                quality = analysis['landing_quality']
                quality_counts[quality] = quality_counts.get(quality, 0) + 1

            print(f"\n🎯 Landing Quality Distribution:")
            for quality, count in quality_counts.items():
                print(
                    f"   {quality}: {count}/{total_episodes} ({count/total_episodes*100:.1f}%)")

            # Speed analysis
            final_speeds = [a['final_speed'] for a in analyses]
            fuel_efficiencies = [a['fuel_efficiency'] for a in analyses]

            print(f"\n🚀 Flight Analysis:")
            print(f"   Average Final Speed: {np.mean(final_speeds):.3f}")
            print(
                f"   Average Fuel Efficiency: {np.mean(fuel_efficiencies):.1f}%")
            print(
                f"   Average Episode Length: {np.mean([a['episode_length'] for a in analyses]):.1f} steps")

        print("="*80)

    def run(self):
        """Main interface loop"""
        print("🌙 MOONLANDER MODEL PLAYER")
        print("Play and analyze your trained models!")

        while True:
            try:
                model_info = self.selector.select_model()
                if model_info is None:
                    break

                print(f"\n🎮 Playing Options:")
                print("1. Quick play (5 episodes, basic analysis)")
                print("2. Detailed analysis (3 episodes, full stats)")
                print("3. Extended session (10 episodes)")
                print("4. Custom settings")
                print("5. Back to model selection")

                option = input("\nSelect option (1-5): ").strip()

                if option == '1':
                    self.play_model(model_info, episodes=5,
                                    detailed_analysis=False)
                elif option == '2':
                    self.play_model(model_info, episodes=3,
                                    detailed_analysis=True)
                elif option == '3':
                    self.play_model(model_info, episodes=10,
                                    detailed_analysis=True)
                elif option == '4':
                    episodes = int(input("Number of episodes (1-20): "))
                    delay = float(input("Delay between frames (0.01-0.1): "))
                    detailed = input(
                        "Detailed analysis? (y/n): ").lower() == 'y'
                    self.play_model(model_info, episodes=episodes,
                                    delay=delay, detailed_analysis=detailed)
                elif option == '5':
                    continue
                else:
                    print("❌ Invalid option!")

            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


if __name__ == "__main__":
    player = EnhancedModelPlayer()
    player.run()

