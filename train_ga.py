"""
Genetic Algorithm Training for Lunar Lander

Train a linear controller using CMA-ES (or simple GA) to solve the Lunar Lander task.

Usage:
    python train_ga.py --generations 200 --population 50 --episodes 10
"""

import argparse
import time
import os
import json
import numpy as np
from datetime import datetime

from linear_controller import LinearController
from ga_evaluator import GAEvaluator, parallel_evaluate
from cma_es_optimizer import CMAESOptimizer


class GATrainer:
    """Manages the genetic algorithm training process."""

    def __init__(
        self,
        controller_class,
        num_generations=200,
        population_size=50,
        episodes_per_eval=10,
        sigma=0.5,
        parallel_workers=1,
        save_dir="ga_models",
        log_interval=1,
        eval_interval=10,
        checkpoint_interval=20
    ):
        """
        Initialize GA trainer.

        Args:
            controller_class: Controller class to evolve
            num_generations: Number of generations to run
            population_size: Size of population
            episodes_per_eval: Episodes to average for fitness
            sigma: Initial step size for CMA-ES
            parallel_workers: Number of parallel workers (1 = sequential)
            save_dir: Directory to save models
            log_interval: Generations between logging
            eval_interval: Generations between detailed evaluation
            checkpoint_interval: Generations between checkpoints
        """
        self.controller_class = controller_class
        self.num_generations = num_generations
        self.population_size = population_size
        self.episodes_per_eval = episodes_per_eval
        self.sigma = sigma
        self.parallel_workers = parallel_workers
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.checkpoint_interval = checkpoint_interval

        # Create save directory
        os.makedirs(save_dir, exist_ok=True)

        # Initialize tracking
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'best_reward': [],
            'best_landing_rate': [],
            'time': []
        }

        self.start_time = None
        self.best_fitness_ever = -np.inf
        self.best_params_ever = None
        self.best_metrics_ever = None

    def train(self):
        """Run the genetic algorithm training."""
        print("="*70)
        print("GENETIC ALGORITHM TRAINING - LUNAR LANDER")
        print("="*70)
        print(f"Controller:       {self.controller_class.__name__}")
        print(f"Parameters:       {LinearController().get_num_parameters()}")
        print(f"Generations:      {self.num_generations}")
        print(f"Population:       {self.population_size}")
        print(f"Episodes/eval:    {self.episodes_per_eval}")
        print(f"Sigma:            {self.sigma}")
        print(f"Parallel workers: {self.parallel_workers}")
        print(f"Save directory:   {self.save_dir}")
        print("="*70 + "\n")

        # Initialize optimizer with random starting point
        initial_controller = self.controller_class()
        initial_params = initial_controller.get_parameters()

        self.optimizer = CMAESOptimizer(
            initial_params=initial_params,
            sigma=self.sigma,
            population_size=self.population_size
        )

        print(f"Optimizer initialized (using CMA: {self.optimizer.use_cma})")
        print(f"Actual population size: {self.optimizer.population_size}\n")

        self.start_time = time.time()

        # Training loop
        for generation in range(self.num_generations):
            gen_start_time = time.time()

            # Get candidate solutions
            candidates = self.optimizer.ask()

            # Evaluate candidates
            if self.parallel_workers > 1:
                fitness_list, metrics_list = parallel_evaluate(
                    candidates,
                    self.controller_class,
                    num_episodes=self.episodes_per_eval,
                    num_workers=self.parallel_workers
                )
            else:
                fitness_list, metrics_list = self._sequential_evaluate(candidates)

            # Update optimizer
            self.optimizer.tell(fitness_list)

            # Track best
            gen_best_idx = np.argmax(fitness_list)
            gen_best_fitness = fitness_list[gen_best_idx]
            gen_best_params = candidates[gen_best_idx]
            gen_best_metrics = metrics_list[gen_best_idx]

            if gen_best_fitness > self.best_fitness_ever:
                self.best_fitness_ever = gen_best_fitness
                self.best_params_ever = gen_best_params.copy()
                self.best_metrics_ever = gen_best_metrics.copy()

                # Save best model
                self._save_best_model()

            # Record history
            gen_time = time.time() - gen_start_time
            self.history['generation'].append(generation)
            self.history['best_fitness'].append(gen_best_fitness)
            self.history['mean_fitness'].append(np.mean(fitness_list))
            self.history['std_fitness'].append(np.std(fitness_list))
            self.history['best_reward'].append(gen_best_metrics['avg_reward'])
            self.history['best_landing_rate'].append(gen_best_metrics['landing_rate'])
            self.history['time'].append(gen_time)

            # Logging
            if generation % self.log_interval == 0 or generation == self.num_generations - 1:
                self._log_generation(generation, gen_best_fitness, gen_best_metrics, gen_time)

            # Detailed evaluation
            if generation % self.eval_interval == 0 and generation > 0:
                self._detailed_evaluation(generation)

            # Checkpoint
            if generation % self.checkpoint_interval == 0 and generation > 0:
                self._save_checkpoint(generation)

            # Check stopping criteria
            if self.optimizer.should_stop():
                print("\n" + "="*70)
                print("Optimizer convergence criteria met. Stopping early.")
                print("="*70 + "\n")
                break

        # Final evaluation
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        self._final_evaluation()
        self._save_training_history()

    def _sequential_evaluate(self, candidates):
        """Evaluate candidates sequentially."""
        evaluator = GAEvaluator(num_episodes=self.episodes_per_eval, render=False)
        fitness_list = []
        metrics_list = []

        for params in candidates:
            fitness, metrics = evaluator.evaluate_parameters(
                params,
                self.controller_class,
                verbose=False
            )
            fitness_list.append(fitness)
            metrics_list.append(metrics)

        evaluator.close()
        return fitness_list, metrics_list

    def _log_generation(self, generation, best_fitness, best_metrics, gen_time):
        """Log generation statistics."""
        elapsed = time.time() - self.start_time
        reward = best_metrics['avg_reward']
        landing_rate = best_metrics['landing_rate'] * 100

        print(f"Gen {generation:3d} | "
              f"Fitness: {best_fitness:7.2f} | "
              f"Reward: {reward:7.2f} | "
              f"Landing: {landing_rate:5.1f}% | "
              f"Time: {gen_time:5.2f}s | "
              f"Elapsed: {elapsed/60:5.1f}m")

    def _detailed_evaluation(self, generation):
        """Run detailed evaluation of best controller."""
        print("\n" + "-"*70)
        print(f"DETAILED EVALUATION - Generation {generation}")
        print("-"*70)

        # Create best controller
        controller = self.controller_class()
        controller.set_parameters(self.best_params_ever)

        # Evaluate with more episodes
        evaluator = GAEvaluator(num_episodes=50, render=False)
        fitness, metrics = evaluator.evaluate_controller(controller, verbose=False)
        evaluator.close()

        print(f"Fitness:          {fitness:.2f}")
        print(f"Average Reward:   {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Landing Rate:     {metrics['landing_rate']*100:.1f}%")
        print(f"Crash Rate:       {metrics['crash_rate']*100:.1f}%")
        print(f"Timeout Rate:     {metrics['timeout_rate']*100:.1f}%")
        print(f"Average Length:   {metrics['avg_length']:.1f} steps")
        print("-"*70 + "\n")

    def _final_evaluation(self):
        """Final comprehensive evaluation."""
        print("\nRunning final evaluation with 100 episodes...")

        # Create best controller
        controller = self.controller_class()
        controller.set_parameters(self.best_params_ever)

        # Evaluate
        evaluator = GAEvaluator(num_episodes=100, render=False)
        fitness, metrics = evaluator.evaluate_controller(controller, verbose=False)
        evaluator.close()

        # Display results
        print("\n" + "="*70)
        print("FINAL RESULTS (100 episodes)")
        print("="*70)
        print(f"Best Fitness:     {fitness:.2f}")
        print(f"Average Reward:   {metrics['avg_reward']:.2f} ± {metrics['std_reward']:.2f}")
        print(f"Landing Rate:     {metrics['landing_rate']*100:.1f}%")
        print(f"  Landings:       {metrics['landing_count']}/100")
        print(f"  Crashes:        {metrics['crash_count']}/100")
        print(f"  Timeouts:       {metrics['timeout_count']}/100")
        print(f"Average Length:   {metrics['avg_length']:.1f} steps")
        print(f"Reward Range:     [{metrics['min_reward']:.2f}, {metrics['max_reward']:.2f}]")
        print("\nTotal Training Time: {:.1f} minutes".format((time.time() - self.start_time) / 60))
        print("="*70 + "\n")

        # Show controller analysis
        controller.describe()

    def _save_best_model(self):
        """Save the best model found so far."""
        controller = self.controller_class()
        controller.set_parameters(self.best_params_ever)

        filepath = os.path.join(self.save_dir, "ga_best.npy")
        controller.save(filepath)

    def _save_checkpoint(self, generation):
        """Save a checkpoint."""
        controller = self.controller_class()
        controller.set_parameters(self.best_params_ever)

        filepath = os.path.join(self.save_dir, f"ga_checkpoint_{generation:04d}.npy")
        controller.save(filepath)

    def _save_training_history(self):
        """Save training history to JSON."""
        filepath = os.path.join(self.save_dir, "training_history.json")

        # Convert numpy types to Python types
        history_serializable = {}
        for key, values in self.history.items():
            history_serializable[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]

        with open(filepath, 'w') as f:
            json.dump(history_serializable, f, indent=2)

        print(f"Training history saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train Lunar Lander with Genetic Algorithm")
    parser.add_argument('--generations', type=int, default=200, help='Number of generations')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--episodes', type=int, default=10, help='Episodes per evaluation')
    parser.add_argument('--sigma', type=float, default=0.5, help='Initial step size (CMA-ES)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    parser.add_argument('--save-dir', type=str, default='ga_models', help='Directory to save models')
    parser.add_argument('--log-interval', type=int, default=1, help='Generations between logs')
    parser.add_argument('--eval-interval', type=int, default=10, help='Generations between detailed evaluations')
    parser.add_argument('--checkpoint-interval', type=int, default=20, help='Generations between checkpoints')

    args = parser.parse_args()

    # Create trainer
    trainer = GATrainer(
        controller_class=LinearController,
        num_generations=args.generations,
        population_size=args.population,
        episodes_per_eval=args.episodes,
        sigma=args.sigma,
        parallel_workers=args.parallel,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval
    )

    # Run training
    trainer.train()


if __name__ == "__main__":
    main()
