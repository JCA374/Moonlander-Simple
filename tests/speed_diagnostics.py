import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from dqn_agent import DQNAgent


def diagnose_speed_problems(model_path='moonlander_best.pth', episodes=10):
    """
    Detailed analysis of why the agent is coming in too fast
    """
    env = gym.make('LunarLander-v2')
    agent = DQNAgent(8, 4)

    try:
        agent.load(model_path)
        agent.epsilon = 0  # No exploration for diagnosis
    except FileNotFoundError:
        print(f"❌ Model not found: {model_path}")
        return

    print("🔍 SPEED PROBLEM DIAGNOSIS")
    print("="*50)

    all_trajectories = []
    crash_reasons = []

    for episode in range(episodes):
        state, _ = env.reset()
        trajectory = {
            'positions': [],
            'speeds': [],
            'actions': [],
            'vertical_speeds': [],
            'altitudes': [],
            'times': []
        }

        for step in range(1000):
            # Record state
            x, y, vx, vy = state[0], state[1], state[2], state[3]
            speed = np.sqrt(vx**2 + vy**2)

            trajectory['positions'].append((x, y))
            trajectory['speeds'].append(speed)
            trajectory['vertical_speeds'].append(abs(vy))
            trajectory['altitudes'].append(max(0, y))
            trajectory['times'].append(step)

            # Get action and step
            action = agent.act(state)
            trajectory['actions'].append(action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state

            if done:
                # Analyze crash reason
                final_x, final_y = next_state[0], next_state[1]
                final_vx, final_vy = next_state[2], next_state[3]
                final_speed = np.sqrt(final_vx**2 + final_vy**2)

                if terminated:
                    if reward > 0:
                        crash_reasons.append("success")
                    else:
                        if final_speed > 0.8:
                            crash_reasons.append("too_fast")
                        elif abs(final_x) > 0.5:
                            crash_reasons.append("off_pad")
                        else:
                            crash_reasons.append("other_crash")
                else:
                    crash_reasons.append("timeout")

                # Store final metrics
                trajectory['final_speed'] = final_speed
                trajectory['final_position'] = (final_x, final_y)
                trajectory['crash_reason'] = crash_reasons[-1]
                trajectory['episode_length'] = step + 1

                break

        all_trajectories.append(trajectory)

        # Print episode summary
        reason = crash_reasons[-1]
        icon = "✅" if reason == "success" else "❌"
        print(f"{icon} Episode {episode+1}: {reason}, Final speed: {trajectory['final_speed']:.3f}, "
              f"Steps: {trajectory['episode_length']}")

    env.close()

    # ===================================================================
    # DETAILED ANALYSIS
    # ===================================================================

    print(f"\n📊 CRASH REASON BREAKDOWN:")
    reason_counts = {}
    for reason in crash_reasons:
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for reason, count in reason_counts.items():
        print(f"   {reason}: {count}/{episodes} ({count/episodes*100:.1f}%)")

    # Speed analysis
    final_speeds = [t['final_speed'] for t in all_trajectories]
    max_speeds = [max(t['speeds']) for t in all_trajectories]

    print(f"\n🚀 SPEED ANALYSIS:")
    print(f"   Average final speed: {np.mean(final_speeds):.3f}")
    print(f"   Max final speed: {max(final_speeds):.3f}")
    print(f"   Min final speed: {min(final_speeds):.3f}")
    print(f"   Average max speed during episode: {np.mean(max_speeds):.3f}")

    # Dangerous moments analysis
    dangerous_episodes = 0
    for traj in all_trajectories:
        # Check for dangerous speed at low altitude
        for i, (alt, speed) in enumerate(zip(traj['altitudes'], traj['speeds'])):
            if alt < 0.5 and speed > 0.8:  # Dangerous combination
                dangerous_episodes += 1
                break

    print(
        f"   Episodes with dangerous speed at low altitude: {dangerous_episodes}/{episodes}")

    # Action analysis
    all_actions = []
    for traj in all_trajectories:
        all_actions.extend(traj['actions'])

    action_counts = [all_actions.count(i) for i in range(4)]
    action_names = ['Do Nothing', 'Left Engine', 'Main Engine', 'Right Engine']

    print(f"\n🎮 ACTION USAGE:")
    for name, count in zip(action_names, action_counts):
        pct = count / len(all_actions) * 100
        print(f"   {name}: {pct:.1f}%")

    # Critical insight: Action usage near ground
    low_altitude_actions = []
    for traj in all_trajectories:
        for i, alt in enumerate(traj['altitudes']):
            if alt < 0.5 and i < len(traj['actions']):
                low_altitude_actions.append(traj['actions'][i])

    if low_altitude_actions:
        low_action_counts = [low_altitude_actions.count(i) for i in range(4)]
        print(f"\n🚨 ACTION USAGE WHEN LOW (altitude < 0.5):")
        for name, count in zip(action_names, low_action_counts):
            pct = count / len(low_altitude_actions) * \
                100 if low_altitude_actions else 0
            print(f"   {name}: {pct:.1f}%")

    # ===================================================================
    # VISUALIZATION
    # ===================================================================

    create_speed_diagnosis_plots(all_trajectories)

    # ===================================================================
    # RECOMMENDATIONS
    # ===================================================================

    print(f"\n💡 SPECIFIC PROBLEMS IDENTIFIED:")

    if reason_counts.get('too_fast', 0) > episodes * 0.3:
        print("   🚨 PRIMARY ISSUE: Agent consistently approaches too fast")
        print("      → Need aggressive speed control in reward shaping")

    if np.mean(final_speeds) > 0.6:
        print("   🚨 HIGH FINAL SPEEDS: Agent not slowing down for landing")
        print("      → Need altitude-based speed limits")

    main_engine_pct = action_counts[2] / len(all_actions) * 100
    if main_engine_pct > 40:
        print(f"   🚨 MAIN ENGINE OVERUSE: {main_engine_pct:.1f}% usage")
        print("      → Need to discourage main engine when already slow")

    do_nothing_pct = action_counts[0] / len(all_actions) * 100
    if do_nothing_pct > 60:
        print(f"   🚨 TOO PASSIVE: {do_nothing_pct:.1f}% doing nothing")
        print("      → Need to encourage active speed control")

    if low_altitude_actions and low_action_counts[0] / len(low_altitude_actions) > 0.5:
        print("   🚨 PASSIVE WHEN LOW: Not using engines near ground")
        print("      → Need to encourage deceleration when approaching")

    print(f"\n🔧 RECOMMENDED FIXES:")
    print("   1. Use GentleLandingShaper with aggressive speed penalties")
    print("   2. Set strict altitude-based speed limits (0.25 when altitude < 0.3)")
    print("   3. Massive rewards for sustained slow approaches")
    print("   4. Penalize main engine overuse when already slow")
    print("   5. Consider retraining with gentler reward shaping from start")


def create_speed_diagnosis_plots(all_trajectories):
    """Create diagnostic plots showing speed problems"""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Speed Problem Diagnosis', fontsize=16)

    # 1. Speed vs Altitude scatter
    ax = axes[0, 0]
    for traj in all_trajectories:
        colors = ['red' if traj['crash_reason'] == 'too_fast' else 'blue'
                  for _ in traj['speeds']]
        ax.scatter(traj['altitudes'], traj['speeds'], c=colors, alpha=0.6, s=2)

    # Add danger zone
    ax.axhspan(0.8, 3.0, xmin=0, xmax=0.5, alpha=0.3,
               color='red', label='Danger Zone')
    ax.set_xlabel('Altitude')
    ax.set_ylabel('Speed')
    ax.set_title('Speed vs Altitude\n(Red = Speed Crashes)')
    ax.legend()
    ax.grid(True)

    # 2. Final speeds histogram
    ax = axes[0, 1]
    final_speeds = [t['final_speed'] for t in all_trajectories]
    ax.hist(final_speeds, bins=10, alpha=0.7,
            color='orange', edgecolor='black')
    ax.axvline(x=0.5, color='red', linestyle='--', label='Safe Speed Limit')
    ax.set_xlabel('Final Speed')
    ax.set_ylabel('Frequency')
    ax.set_title('Final Landing Speeds')
    ax.legend()

    # 3. Speed over time for crashed episodes
    ax = axes[0, 2]
    crash_trajectories = [
        t for t in all_trajectories if t['crash_reason'] == 'too_fast']
    for i, traj in enumerate(crash_trajectories[:5]):  # Show first 5 crashes
        ax.plot(traj['times'], traj['speeds'], alpha=0.7, label=f'Crash {i+1}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Speed')
    ax.set_title('Speed Progression (Fast Crashes)')
    ax.grid(True)

    # 4. Action usage by altitude
    ax = axes[1, 0]
    altitude_bins = [0, 0.3, 0.6, 1.0, 2.0, 10.0]
    action_names = ['Do Nothing', 'Left', 'Main', 'Right']

    for alt_idx in range(len(altitude_bins)-1):
        actions_in_bin = []
        for traj in all_trajectories:
            for i, alt in enumerate(traj['altitudes']):
                if altitude_bins[alt_idx] <= alt < altitude_bins[alt_idx+1] and i < len(traj['actions']):
                    actions_in_bin.append(traj['actions'][i])

        if actions_in_bin:
            action_counts = [actions_in_bin.count(j) for j in range(4)]
            action_pcts = [c/len(actions_in_bin)*100 for c in action_counts]

            x_pos = alt_idx
            bottom = 0
            for j, (pct, name) in enumerate(zip(action_pcts, action_names)):
                ax.bar(x_pos, pct, bottom=bottom,
                       label=name if alt_idx == 0 else "")
                bottom += pct

    ax.set_xlabel('Altitude Range')
    ax.set_ylabel('Action Usage %')
    ax.set_title('Action Usage by Altitude')
    ax.set_xticks(range(len(altitude_bins)-1))
    ax.set_xticklabels([f'{altitude_bins[i]:.1f}-{altitude_bins[i+1]:.1f}'
                       for i in range(len(altitude_bins)-1)])
    ax.legend()

    # 5. Speed reduction attempts
    ax = axes[1, 1]
    speed_reductions = []
    for traj in all_trajectories:
        reductions = []
        for i in range(1, len(traj['speeds'])):
            if traj['altitudes'][i] < 1.0:  # Only when reasonably low
                reduction = traj['speeds'][i-1] - traj['speeds'][i]
                reductions.append(reduction)
        if reductions:
            speed_reductions.extend(reductions)

    if speed_reductions:
        ax.hist(speed_reductions, bins=20, alpha=0.7,
                color='green', edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', label='No Change')
        ax.set_xlabel('Speed Change per Step')
        ax.set_ylabel('Frequency')
        ax.set_title('Speed Reduction Attempts\n(Positive = Slowing Down)')
        ax.legend()

    # 6. Trajectory overview
    ax = axes[1, 2]
    for i, traj in enumerate(all_trajectories[:5]):
        positions = np.array(traj['positions'])
        color = 'green' if traj['crash_reason'] == 'success' else 'red'
        ax.plot(positions[:, 0], positions[:, 1], color=color, alpha=0.7,
                linewidth=2, label=f"Ep {i+1}")

    # Landing pad
    ax.axhspan(-0.1, 0.1, xmin=0.4, xmax=0.6, alpha=0.3,
               color='yellow', label='Landing Pad')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Flight Trajectories')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    diagnose_speed_problems()
