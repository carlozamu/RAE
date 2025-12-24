"""
Test suite for comparing different selection strategies.
Simulates evolutionary runs and compares performance metrics.
"""
import sys
import os
from typing import List, Dict, Any
import random
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Edoardo.Selection.selection import (
    ElitismSelection,
    RankBasedSelection,
    TournamentSelection,
    FitnessProportionateSelection
)
from Edoardo.Generation_Manager.generation_manager import (
    CommaPlusStrategy,
    CommaStrategy,
    HallOfFameStrategy,
    GenerationManager
)


class MockIndividual:
    """Mock individual for testing."""
    def __init__(self, id: int, fitness: float):
        self.id = id
        self.fitness = fitness
    
    def __repr__(self):
        return f"Individual(id={self.id}, fitness={self.fitness:.3f})"


def create_mock_population(size: int, fitness_range: tuple = (0.0, 1.0)) -> List[Dict[str, Any]]:
    """Create a mock population with random fitness values."""
    population = []
    for i in range(size):
        fitness = random.uniform(*fitness_range)
        individual = MockIndividual(id=i, fitness=fitness)
        population.append({
            'member': individual,
            'fitness': fitness
        })
    return population


def simulate_generation(
    population: List[Dict[str, Any]],
    selection_strategy,
    num_offspring: int,
    mutation_rate: float = 0.1
) -> List[Dict[str, Any]]:
    """
    Simulate one generation: select parents, create offspring with mutations.
    
    Args:
        population: Current population
        selection_strategy: Selection strategy to use
        num_offspring: Number of offspring to create
        mutation_rate: Probability of fitness improvement in offspring
    
    Returns:
        List of offspring individuals
    """
    offspring = []
    for _ in range(num_offspring):
        # Select parents
        parents = selection_strategy.select(population, num_parents=2)
        
        # Create offspring (simplified: average fitness with chance of improvement)
        parent_fitnesses = [p['fitness'] for p in parents]
        base_fitness = np.mean(parent_fitnesses)
        
        # Mutation: chance to improve or worsen
        if random.random() < mutation_rate:
            # Improvement
            fitness = min(1.0, base_fitness + random.uniform(0, 0.2))
        else:
            # Slight variation
            fitness = max(0.0, base_fitness + random.uniform(-0.1, 0.1))
        
        new_id = max([ind['member'].id for ind in population]) + len(offspring) + 1
        offspring.append({
            'member': MockIndividual(id=new_id, fitness=fitness),
            'fitness': fitness
        })
    
    return offspring


def run_evolution_experiment(
    selection_strategy,
    survivor_strategy,
    initial_population_size: int = 50,
    num_generations: int = 20,
    num_offspring: int = 50
) -> Dict[str, Any]:
    """
    Run a complete evolutionary experiment with given strategies.
    
    Returns:
        Dictionary with statistics about the run
    """
    # Initialize
    population = create_mock_population(initial_population_size)
    gen_manager = GenerationManager(
        population_size=initial_population_size,
        survivor_strategy=survivor_strategy
    )
    
    best_fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    
    for gen in range(num_generations):
        # Evaluate current population (already has fitness)
        best_fitness = max(ind['fitness'] for ind in population)
        avg_fitness = np.mean([ind['fitness'] for ind in population])
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        # Calculate diversity (std of fitnesses)
        fitnesses = [ind['fitness'] for ind in population]
        diversity = np.std(fitnesses)
        diversity_history.append(diversity)
        
        # Create offspring
        offspring = simulate_generation(
            population,
            selection_strategy,
            num_offspring=num_offspring
        )
        
        # Advance generation
        population = gen_manager.advance_generation(population, offspring)
    
    return {
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'diversity_history': diversity_history,
        'final_best_fitness': best_fitness_history[-1] if best_fitness_history else 0.0,
        'final_avg_fitness': avg_fitness_history[-1] if avg_fitness_history else 0.0,
        'convergence_generation': _find_convergence(best_fitness_history),
        'improvement_rate': _calculate_improvement_rate(best_fitness_history)
    }


def _find_convergence(fitness_history: List[float], threshold: float = 0.001) -> int:
    """Find generation where fitness improvement becomes negligible."""
    for i in range(1, len(fitness_history)):
        if abs(fitness_history[i] - fitness_history[i-1]) < threshold:
            return i
    return len(fitness_history)


def _calculate_improvement_rate(fitness_history: List[float]) -> float:
    """Calculate average improvement per generation."""
    if len(fitness_history) < 2:
        return 0.0
    improvements = [fitness_history[i] - fitness_history[i-1] 
                   for i in range(1, len(fitness_history))]
    return np.mean(improvements) if improvements else 0.0


def compare_selection_strategies(
    strategies: List[tuple],
    num_runs: int = 5,
    num_generations: int = 20
) -> Dict[str, Any]:
    """
    Compare multiple selection strategies across multiple runs.
    
    Args:
        strategies: List of (selection_strategy, survivor_strategy, name) tuples
        num_runs: Number of independent runs per strategy
        num_generations: Number of generations per run
    
    Returns:
        Dictionary with comparison results
    """
    results = defaultdict(list)
    
    print(f"\n{'='*80}")
    print(f"Comparing {len(strategies)} selection strategies")
    print(f"Running {num_runs} independent runs, {num_generations} generations each")
    print(f"{'='*80}\n")
    
    for sel_strategy, surv_strategy, name in strategies:
        print(f"Testing: {name}")
        print(f"  Selection: {sel_strategy.__class__.__name__}")
        print(f"  Survivor: {surv_strategy.__class__.__name__}")
        
        run_results = []
        for run in range(num_runs):
            result = run_evolution_experiment(
                selection_strategy=sel_strategy,
                survivor_strategy=surv_strategy,
                num_generations=num_generations
            )
            run_results.append(result)
        
        # Aggregate statistics
        final_bests = [r['final_best_fitness'] for r in run_results]
        final_avgs = [r['final_avg_fitness'] for r in run_results]
        convergence_gens = [r['convergence_generation'] for r in run_results]
        improvement_rates = [r['improvement_rate'] for r in run_results]
        
        results[name] = {
            'final_best_mean': np.mean(final_bests),
            'final_best_std': np.std(final_bests),
            'final_avg_mean': np.mean(final_avgs),
            'final_avg_std': np.std(final_avgs),
            'convergence_mean': np.mean(convergence_gens),
            'improvement_rate_mean': np.mean(improvement_rates),
            'all_runs': run_results
        }
        
        print(f"  Final Best Fitness: {np.mean(final_bests):.4f} ± {np.std(final_bests):.4f}")
        print(f"  Final Avg Fitness: {np.mean(final_avgs):.4f} ± {np.std(final_avgs):.4f}")
        print(f"  Avg Convergence Gen: {np.mean(convergence_gens):.1f}")
        print(f"  Avg Improvement Rate: {np.mean(improvement_rates):.6f}\n")
    
    return dict(results)


def main():
    """Main test function comparing top selection strategies."""
    
    # Define strategies to test
    # Top 3 combinations based on literature and NEAT compatibility
    strategies = [
        # 1. Tournament + CommaPlus (good balance, widely used)
        (TournamentSelection(tournament_size=3), CommaPlusStrategy(), "Tournament + CommaPlus"),
        
        # 2. Rank-based + CommaPlus (robust, maintains diversity)
        (RankBasedSelection(selection_pressure=2.0), CommaPlusStrategy(), "Rank-based + CommaPlus"),
        
        # 3. Elitism + HallOfFame (preserves best solutions)
        (ElitismSelection(elite_size=5), HallOfFameStrategy(hall_of_fame_size=10), "Elitism + HallOfFame"),
        
        # Bonus: Tournament + Comma (more explorative)
        (TournamentSelection(tournament_size=3), CommaStrategy(), "Tournament + Comma"),
    ]
    
    # Run comparison
    results = compare_selection_strategies(
        strategies=strategies,
        num_runs=5,
        num_generations=20
    )
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY - Best Strategies by Final Best Fitness:")
    print(f"{'='*80}")
    
    sorted_strategies = sorted(
        results.items(),
        key=lambda x: x[1]['final_best_mean'],
        reverse=True
    )
    
    for rank, (name, stats) in enumerate(sorted_strategies, 1):
        print(f"{rank}. {name}")
        print(f"   Best Fitness: {stats['final_best_mean']:.4f} ± {stats['final_best_std']:.4f}")
        print(f"   Avg Fitness: {stats['final_avg_mean']:.4f} ± {stats['final_avg_std']:.4f}")
        print(f"   Convergence: {stats['convergence_mean']:.1f} generations")
        print()
    
    return results


if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    results = main()
    
    print("\n" + "="*80)
    print("RECOMMENDATION:")
    print("="*80)
    print("For NEAT-based prompt evolution, consider:")
    print("1. Tournament + CommaPlus: Good balance of exploration/exploitation")
    print("2. Rank-based + CommaPlus: More robust to fitness scaling")
    print("3. Elitism + HallOfFame: Best for preserving top solutions")
    print("="*80)

