import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent
from precision_landing_shaper import GentleLandingShaper


def analyze_agent_behavior():
    """Deep dive into what the agent is actually doing"""

    env = gym.make('LunarLander-v2')
    agent = DQNAgent(8, 4)
    agent.load('models/moonlander_best.pth')
    agent.epsilon = 0

    episodes_data = []

    for episode in range(10):
        state, _ = env.reset()
        episode_data = {
            'states': [state.copy()],
            'actions': [],
            'rewards': [],
            'episode': episode
        }

        total_reward = 0
        for step in range(1000):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_data['actions'].append(action)
            episode_data['rewards'].append(reward)
            episode_data['states'].append(next_state.copy())

            total_reward += reward
            state = next_state

            if done:
                break

        episode_data['total_reward'] = total_reward
        episode_data['final_state'] = next_state
        episode_data['terminated'] = terminated
        episode_data['success'] = terminated and reward > 0
        episodes_data.append(episode_data)

        # Quick summary
        x, y, vx, vy, angle = next_state[:5]
        print(f"Ep {episode+1}: Reward={total_reward:.1f}, "
              f"Final pos=({x:.2f},{y:.2f}), "
              f"Speed={np.sqrt(vx**2+vy**2):.2f}, "
              f"Success={terminated and reward > 0}")

    env.close()
    return episodes_data


def plot_agent_behavior(episodes_data):
    """Visualize what the agent is doing wrong"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Agent Behavior Analysis - Why No Landings?', fontsize=16)

    # 1. Trajectory plot
    ax = axes[0, 0]
    for ep_data in episodes_data[:5]:  # Show first 5 episodes
        states = np.array(ep_data['states'])
        x, y = states[:, 0], states[:, 1]
        color = 'green' if ep_data['success'] else 'red'
        ax.plot(x, y, color=color, alpha=0.7, linewidth=2)

    # Landing pad
    ax.axhspan(-0.1, 0.1, xmin=0.4, xmax=0.6, alpha=0.3,
               color='yellow', label='Landing Pad')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 2)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Flight Trajectories')
    ax.grid(True)
    ax.legend()

    # 2. Final positions
    ax = axes[0, 1]
    final_x = [ep['final_state'][0] for ep in episodes_data]
    final_y = [ep['final_state'][1] for ep in episodes_data]
    colors = ['green' if ep['success'] else 'red' for ep in episodes_data]

    ax.scatter(final_x, final_y, c=colors, s=100, alpha=0.7)
    ax.axhspan(-0.1, 0.1, xmin=0.4, xmax=0.6, alpha=0.3, color='yellow')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.5, 1)
    ax.set_xlabel('Final X Position')
    ax.set_ylabel('Final Y Position')
    ax.set_title('Where Agent Ends Up')
    ax.grid(True)

    # 3. Action distribution
    ax = axes[0, 2]
    all_actions = []
    for ep_data in episodes_data:
        all_actions.extend(ep_data['actions'])

    action_counts = [all_actions.count(i) for i in range(4)]
    action_names = ['Do Nothing', 'Left Engine', 'Main Engine', 'Right Engine']
    bars = ax.bar(action_names, action_counts)
    ax.set_title('Action Usage Distribution')
    ax.set_ylabel('Count')

    # Color bars based on usage
    for i, bar in enumerate(bars):
        if action_counts[i] > len(all_actions) * 0.5:
            bar.set_color('red')  # Overused
        elif action_counts[i] < len(all_actions) * 0.1:
            bar.set_color('orange')  # Underused
        else:
            bar.set_color('blue')  # Balanced

    # 4. Speed at landing
    ax = axes[1, 0]
    final_speeds = []
    for ep_data in episodes_data:
        final_state = ep_data['final_state']
        vx, vy = final_state[2], final_state[3]
        speed = np.sqrt(vx**2 + vy**2)
        final_speeds.append(speed)

    ax.hist(final_speeds, bins=8, alpha=0.7,
            color='skyblue', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Max Safe Speed')
    ax.set_xlabel('Final Speed')
    ax.set_ylabel('Frequency')
    ax.set_title('Landing Speed Distribution')
    ax.legend()

    # 5. Angle at landing
    ax = axes[1, 1]
    final_angles = [ep['final_state'][4] for ep in episodes_data]
    ax.hist(final_angles, bins=8, alpha=0.7,
            color='lightcoral', edgecolor='black')
    ax.axvline(x=-0.2, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=0.2, color='red', linestyle='--',
               alpha=0.7, label='Safe Angle Range')
    ax.set_xlabel('Final Angle (radians)')
    ax.set_ylabel('Frequency')
    ax.set_title('Landing Angle Distribution')
    ax.legend()

    # 6. Episode length vs reward
    ax = axes[1, 2]
    episode_lengths = [len(ep['actions']) for ep in episodes_data]
    rewards = [ep['total_reward'] for ep in episodes_data]
    colors = ['green' if ep['success'] else 'red' for ep in episodes_data]

    ax.scatter(episode_lengths, rewards, c=colors, s=100, alpha=0.7)
    ax.set_xlabel('Episode Length (steps)')
    ax.set_ylabel('Total Reward')
    ax.set_title('Episode Length vs Reward')
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return fig


def identify_problems(episodes_data):
    """Identify specific behavioral problems"""

    print("\n🔍 BEHAVIORAL PROBLEMS IDENTIFIED:")
    print("="*50)

    problems = []

    # Problem 1: Action bias
    all_actions = []
    for ep_data in episodes_data:
        all_actions.extend(ep_data['actions'])

    action_distribution = [all_actions.count(
        i)/len(all_actions) for i in range(4)]
    action_names = ['Do Nothing', 'Left Engine', 'Main Engine', 'Right Engine']

    print("1. ACTION USAGE:")
    for i, (name, pct) in enumerate(zip(action_names, action_distribution)):
        print(f"   {name}: {pct*100:.1f}%")
        if pct > 0.6:
            problems.append(f"OVERUSES {name} ({pct*100:.1f}%)")
        elif pct < 0.05:
            problems.append(f"NEVER USES {name} ({pct*100:.1f}%)")

    # Problem 2: Landing positions
    final_positions = [(ep['final_state'][0], ep['final_state'][1])
                       for ep in episodes_data]
    in_pad_x = sum(1 for x, y in final_positions if abs(x) < 0.5)
    in_pad_y = sum(1 for x, y in final_positions if y < 0.2)

    print(f"\n2. LANDING POSITIONS:")
    print(f"   X-position OK (within pad): {in_pad_x}/10 episodes")
    print(f"   Y-position OK (low enough): {in_pad_y}/10 episodes")

    if in_pad_x < 5:
        problems.append("POOR HORIZONTAL CONTROL - misses landing pad")
    if in_pad_y < 5:
        problems.append("POOR ALTITUDE CONTROL - doesn't get low enough")

    # Problem 3: Speed control
    final_speeds = []
    for ep_data in episodes_data:
        vx, vy = ep_data['final_state'][2], ep_data['final_state'][3]
        speed = np.sqrt(vx**2 + vy**2)
        final_speeds.append(speed)

    fast_landings = sum(1 for speed in final_speeds if speed > 0.5)
    print(f"\n3. SPEED CONTROL:")
    print(f"   Too fast landings: {fast_landings}/10 episodes")
    print(f"   Average landing speed: {np.mean(final_speeds):.2f}")

    if fast_landings > 5:
        problems.append("TOO FAST - doesn't slow down enough for landing")

    # Problem 4: Episode termination analysis
    timeouts = sum(1 for ep in episodes_data if len(ep['actions']) >= 999)
    crashes = sum(
        1 for ep in episodes_data if ep['terminated'] and not ep['success'])
    successes = sum(1 for ep in episodes_data if ep['success'])

    print(f"\n4. EPISODE OUTCOMES:")
    print(f"   Timeouts: {timeouts}/10")
    print(f"   Crashes: {crashes}/10")
    print(f"   Successes: {successes}/10")

    if timeouts > 3:
        problems.append(
            "FREQUENT TIMEOUTS - indecisive behavior, possible hovering")
    if crashes > 7:
        problems.append("FREQUENT CRASHES - poor landing technique")

    # Summary
    print(f"\n🚨 MAIN PROBLEMS:")
    for i, problem in enumerate(problems, 1):
        print(f"   {i}. {problem}")

    return problems


def main():
    print("🔍 Deep Analysis of Agent Behavior")
    print("This will run 10 episodes and analyze what's going wrong...")

    episodes_data = analyze_agent_behavior()
    problems = identify_problems(episodes_data)

    print(f"\n📊 Generating behavior plots...")
    plot_agent_behavior(episodes_data)

    # Recommendations based on problems
    print(f"\n💡 RECOMMENDED FIXES:")

    action_bias = any("OVERUSES" in p or "NEVER USES" in p for p in problems)
    if action_bias:
        print("   → INCREASE EXPLORATION: Lower epsilon_min to 0.01, slower decay")
        print("   → ADD ACTION ENTROPY BONUS in reward shaping")

    if any("POOR HORIZONTAL" in p for p in problems):
        print("   → IMPROVE GUIDANCE: Stronger potential-based reward for X-position")

    if any("POOR ALTITUDE" in p for p in problems):
        print("   → ALTITUDE CONTROL: Reward descent more strongly")

    if any("TOO FAST" in p for p in problems):
        print("   → SPEED CONTROL: Penalize high speed near ground")

    if any("TIMEOUTS" in p for p in problems):
        print("   → ANTI-HOVER: Stronger penalties for staying stationary")
        print("   → DECISIVENESS: Reward reaching ground quickly")


if __name__ == "__main__":
    main()
