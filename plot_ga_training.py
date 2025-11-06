"""
Plot GA Training History

Visualizes the training progress of the genetic algorithm.

Usage:
    python plot_ga_training.py --history ga_models/training_history.json
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_history(history_file, save_dir=None):
    """
    Plot training history from JSON file.

    Args:
        history_file: Path to training_history.json
        save_dir: Directory to save plots (None = just display)
    """
    # Load history
    with open(history_file, 'r') as f:
        history = json.load(f)

    generations = history['generation']
    best_fitness = history['best_fitness']
    mean_fitness = history['mean_fitness']
    std_fitness = history['std_fitness']
    best_reward = history['best_reward']
    best_landing_rate = history['best_landing_rate']
    time_per_gen = history['time']

    # Convert to numpy arrays
    generations = np.array(generations)
    best_fitness = np.array(best_fitness)
    mean_fitness = np.array(mean_fitness)
    std_fitness = np.array(std_fitness)
    best_reward = np.array(best_reward)
    best_landing_rate = np.array(best_landing_rate) * 100  # Convert to percentage

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Genetic Algorithm Training Progress', fontsize=16, fontweight='bold')

    # Plot 1: Fitness over generations
    ax1 = axes[0, 0]
    ax1.plot(generations, best_fitness, 'b-', linewidth=2, label='Best Fitness')
    ax1.plot(generations, mean_fitness, 'g--', linewidth=1.5, label='Mean Fitness')
    ax1.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.2,
        color='g',
        label='±1 Std Dev'
    )
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reward over generations
    ax2 = axes[0, 1]
    ax2.plot(generations, best_reward, 'r-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Average Reward')
    ax2.set_title('Best Reward per Generation')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=200, color='g', linestyle='--', alpha=0.5, label='Target (200)')
    ax2.legend()

    # Plot 3: Landing rate over generations
    ax3 = axes[1, 0]
    ax3.plot(generations, best_landing_rate, 'm-', linewidth=2)
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Landing Rate (%)')
    ax3.set_title('Best Landing Rate per Generation')
    ax3.set_ylim([0, 105])
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=90, color='g', linestyle='--', alpha=0.5, label='Excellent (90%)')
    ax3.legend()

    # Plot 4: Time per generation
    ax4 = axes[1, 1]
    ax4.plot(generations, time_per_gen, 'c-', linewidth=1.5, alpha=0.7)
    # Add moving average
    window = min(10, len(time_per_gen) // 5)
    if window > 1:
        moving_avg = np.convolve(time_per_gen, np.ones(window)/window, mode='valid')
        ax4.plot(generations[window-1:], moving_avg, 'b-', linewidth=2, label=f'{window}-gen Moving Avg')
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Time per Generation')
    ax4.grid(True, alpha=0.3)
    if window > 1:
        ax4.legend()

    plt.tight_layout()

    # Save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'training_progress.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    print(f"Total Generations:     {len(generations)}")
    print(f"Best Fitness:          {np.max(best_fitness):.2f}")
    print(f"Best Reward:           {np.max(best_reward):.2f}")
    print(f"Best Landing Rate:     {np.max(best_landing_rate):.1f}%")
    print(f"Final Fitness:         {best_fitness[-1]:.2f}")
    print(f"Final Reward:          {best_reward[-1]:.2f}")
    print(f"Final Landing Rate:    {best_landing_rate[-1]:.1f}%")
    print(f"Total Training Time:   {np.sum(time_per_gen)/60:.1f} minutes")
    print(f"Avg Time per Gen:      {np.mean(time_per_gen):.2f} seconds")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot GA training history")
    parser.add_argument('--history', type=str, default='ga_models/training_history.json',
                        help='Path to training_history.json file')
    parser.add_argument('--save', action='store_true',
                        help='Save plot to file instead of displaying')
    parser.add_argument('--save-dir', type=str, default='ga_models',
                        help='Directory to save plots')

    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"Error: History file not found: {args.history}")
        print("Train a model first using: python train_ga.py")
        return

    save_dir = args.save_dir if args.save else None
    plot_training_history(args.history, save_dir=save_dir)


if __name__ == "__main__":
    main()
