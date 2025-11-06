"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) Optimizer

This module provides a wrapper for CMA-ES optimization, with fallback
to a simple genetic algorithm if the cma library is not available.

CMA-ES is a state-of-the-art evolutionary algorithm for continuous optimization.
"""

import numpy as np
from typing import Callable, Tuple, Optional, List
import os


class CMAESOptimizer:
    """
    CMA-ES optimizer wrapper with fallback to simple GA.
    """

    def __init__(
        self,
        initial_params: np.ndarray,
        sigma: float = 0.5,
        population_size: Optional[int] = None,
        use_cma_library: bool = True
    ):
        """
        Initialize CMA-ES optimizer.

        Args:
            initial_params: Initial parameter guess
            sigma: Initial standard deviation (step size)
            population_size: Population size (None = auto)
            use_cma_library: Try to use cma library if available
        """
        self.num_params = len(initial_params)
        self.initial_params = np.array(initial_params)
        self.sigma = sigma
        self.generation = 0

        # Try to use cma library
        self.use_cma = False
        if use_cma_library:
            try:
                import cma
                self.cma = cma
                self.use_cma = True
                print("Using CMA-ES library (optimal)")
            except ImportError:
                print("CMA library not found, using simple GA fallback")
                print("To install: pip install cma")

        if self.use_cma:
            # Use CMA-ES library
            options = {
                'popsize': population_size,
                'seed': 42,
                'verbose': -1  # Suppress output, we'll print our own
            }
            # Remove None values
            options = {k: v for k, v in options.items() if v is not None}

            self.es = self.cma.CMAEvolutionStrategy(initial_params, sigma, options)
            self.population_size = self.es.popsize

        else:
            # Use simple GA fallback
            self.population_size = population_size if population_size else max(50, 4 + int(3 * np.log(self.num_params)))
            self.mutation_rate = 0.1
            self.mutation_strength = sigma
            self.crossover_rate = 0.7
            self.elite_size = max(2, self.population_size // 10)

            # Initialize population around initial params
            self.population = [
                initial_params + np.random.randn(self.num_params) * sigma
                for _ in range(self.population_size)
            ]
            self.fitness_history = []

    def ask(self) -> List[np.ndarray]:
        """
        Generate candidate solutions for evaluation.

        Returns:
            List of parameter arrays to evaluate
        """
        if self.use_cma:
            solutions = self.es.ask()
            return [np.array(s) for s in solutions]
        else:
            # Return current population (already generated in tell())
            if self.generation == 0:
                return self.population
            else:
                return self.new_population

    def tell(self, fitness_list: List[float]):
        """
        Update optimizer with fitness values.

        Args:
            fitness_list: Fitness values for each candidate (higher is better)
        """
        if self.use_cma:
            # CMA-ES minimizes by default, so negate fitness
            self.es.tell(self.es.population, [-f for f in fitness_list])
        else:
            # Simple GA: selection, crossover, mutation
            self._simple_ga_step(fitness_list)

        self.generation += 1

    def _simple_ga_step(self, fitness_list: List[float]):
        """Perform one generation of simple GA."""
        fitness_array = np.array(fitness_list)
        self.fitness_history.append({
            'max': np.max(fitness_array),
            'mean': np.mean(fitness_array),
            'std': np.std(fitness_array)
        })

        # Selection: sort by fitness (higher is better)
        sorted_indices = np.argsort(fitness_array)[::-1]
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = fitness_array[sorted_indices]

        # Elitism: keep top individuals
        new_population = sorted_population[:self.elite_size]

        # Selection probabilities (fitness-proportional)
        # Shift fitness to be positive
        shifted_fitness = sorted_fitness - sorted_fitness.min() + 1e-6
        selection_probs = shifted_fitness / shifted_fitness.sum()

        # Generate rest of population through crossover and mutation
        while len(new_population) < self.population_size:
            if np.random.random() < self.crossover_rate and len(new_population) > 1:
                # Crossover
                parent1_idx = np.random.choice(len(sorted_population), p=selection_probs)
                parent2_idx = np.random.choice(len(sorted_population), p=selection_probs)
                parent1 = sorted_population[parent1_idx]
                parent2 = sorted_population[parent2_idx]

                # Blend crossover
                alpha = np.random.random()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                # Just select a parent
                parent_idx = np.random.choice(len(sorted_population), p=selection_probs)
                child = sorted_population[parent_idx].copy()

            # Mutation
            child = self._mutate(child)
            new_population.append(child)

        self.population = new_population
        self.new_population = new_population

    def _mutate(self, params: np.ndarray) -> np.ndarray:
        """Apply Gaussian mutation to parameters."""
        mutated = params.copy()
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                mutated[i] += np.random.randn() * self.mutation_strength
        return mutated

    def get_best(self) -> Tuple[np.ndarray, Optional[float]]:
        """
        Get best solution found so far.

        Returns:
            best_params: Best parameter array
            best_fitness: Best fitness (if available)
        """
        if self.use_cma:
            best = self.es.result.xbest
            fitness = -self.es.result.fbest  # Negate back to original scale
            return np.array(best), fitness
        else:
            # For simple GA, we need to track best separately
            # This is a limitation - would need to track best across generations
            return self.population[0], None

    def should_stop(self) -> bool:
        """Check if optimization should stop."""
        if self.use_cma:
            return self.es.stop()
        else:
            # Simple stopping criteria for GA
            if self.generation < 10:
                return False

            # Check if fitness has plateaued
            if len(self.fitness_history) >= 20:
                recent_max = [h['max'] for h in self.fitness_history[-20:]]
                improvement = recent_max[-1] - recent_max[0]
                if improvement < 1.0:  # Less than 1 point improvement in 20 generations
                    return False  # Don't stop automatically for simple GA

            return False

    def get_stats(self) -> dict:
        """Get current optimization statistics."""
        if self.use_cma:
            return {
                'generation': self.generation,
                'best_fitness': -self.es.result.fbest,
                'sigma': self.es.sigma,
                'evaluations': self.es.result.evaluations
            }
        else:
            if self.fitness_history:
                latest = self.fitness_history[-1]
                return {
                    'generation': self.generation,
                    'best_fitness': latest['max'],
                    'mean_fitness': latest['mean'],
                    'std_fitness': latest['std']
                }
            else:
                return {'generation': self.generation}


if __name__ == "__main__":
    # Test the optimizer with a simple objective function
    print("Testing CMA-ES Optimizer...")

    # Simple test function: minimize distance from target
    target = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def objective(params):
        """Fitness = negative distance from target (higher is better)"""
        distance = np.linalg.norm(params - target)
        return -distance

    # Initialize optimizer
    initial = np.zeros(5)
    optimizer = CMAESOptimizer(initial, sigma=1.0, population_size=10)

    print(f"\nOptimizing {len(target)}-dimensional problem")
    print(f"Target: {target}")
    print(f"Population size: {optimizer.population_size}")
    print(f"Using CMA library: {optimizer.use_cma}\n")

    # Run optimization
    for gen in range(20):
        # Get candidates
        candidates = optimizer.ask()

        # Evaluate
        fitness_list = [objective(c) for c in candidates]

        # Update
        optimizer.tell(fitness_list)

        # Print progress
        stats = optimizer.get_stats()
        if gen % 5 == 0:
            best_params, best_fitness = optimizer.get_best()
            print(f"Generation {gen:3d}: Best fitness = {best_fitness:.4f}, Distance = {-best_fitness:.4f}")

    # Final result
    best_params, best_fitness = optimizer.get_best()
    print(f"\nFinal result:")
    print(f"Best parameters: {best_params}")
    print(f"Target:          {target}")
    print(f"Error:           {np.linalg.norm(best_params - target):.6f}")
