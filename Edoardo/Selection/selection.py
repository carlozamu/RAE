"""
Selection strategies for evolutionary algorithms.
Each strategy selects parents from a population based on different criteria.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import random
import numpy as np


class SelectionStrategy(ABC):
    """
    Base class for all selection strategies.
    Works with individuals stored as dictionaries with 'member' and 'fitness' keys.
    """
    
    @abstractmethod
    def select(self, population: List[Dict[str, Any]], num_parents: int) -> List[Dict[str, Any]]:
        """
        Select parents from the population.
        
        Args:
            population: List of individuals, each as dict with 'member' and 'fitness' keys
            num_parents: Number of parents to select
            
        Returns:
            List of selected parent dictionaries
        """
        pass
    
    def _validate_population(self, population: List[Dict[str, Any]]) -> None:
        """Validate that population has required structure."""
        if not population:
            raise ValueError("Population cannot be empty")
        if not all('fitness' in ind and 'member' in ind for ind in population):
            raise ValueError("Each individual must have 'member' and 'fitness' keys")


class ElitismSelection(SelectionStrategy):
    """
    Elitism selection: Always selects the top N individuals by fitness.
    Preserves the best individuals across generations.
    """
    
    def __init__(self, elite_size: int = 1):
        """
        Args:
            elite_size: Number of top individuals to always select (default: 1)
        """
        self.elite_size = elite_size
    
    def select(self, population: List[Dict[str, Any]], num_parents: int) -> List[Dict[str, Any]]:
        """
        Selects top individuals first, then fills remaining slots randomly from top performers.
        """
        self._validate_population(population)
        
        # Sort by fitness (descending: higher is better)
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        
        # Always include top elite_size individuals
        selected = sorted_pop[:min(self.elite_size, num_parents)]
        
        # Fill remaining slots from top performers (top 50% of population)
        remaining = num_parents - len(selected)
        if remaining > 0:
            top_half = sorted_pop[:max(1, len(sorted_pop) // 2)]
            if len(top_half) > remaining:
                selected.extend(random.sample(top_half, remaining))
            else:
                selected.extend(top_half)
                # If still need more, sample with replacement from top half
                while len(selected) < num_parents:
                    selected.append(random.choice(top_half))
        
        return selected[:num_parents]


class RankBasedSelection(SelectionStrategy):
    """
    Rank-based selection: Assigns selection probability based on rank rather than raw fitness.
    More robust to fitness scaling issues and maintains selection pressure.
    """
    
    def __init__(self, selection_pressure: float = 2.0):
        """
        Args:
            selection_pressure: Controls how much better ranks are favored (default: 2.0)
                               Higher values = stronger bias toward top ranks
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[Dict[str, Any]], num_parents: int) -> List[Dict[str, Any]]:
        """
        Selects parents based on rank-based probabilities.
        """
        self._validate_population(population)
        
        # Sort by fitness (descending: higher is better)
        sorted_pop = sorted(population, key=lambda x: x['fitness'], reverse=True)
        n = len(sorted_pop)
        
        # Assign ranks (1 = best, n = worst)
        # Calculate selection probabilities using linear ranking
        # P(rank i) = (2 - s) / n + 2 * (n - i + 1) * (s - 1) / (n * (n - 1))
        # where s = selection_pressure, and we use (n - i + 1) to favor better ranks
        # This ensures rank 1 (best) gets highest probability, rank n (worst) gets lowest
        probabilities = []
        for i, rank in enumerate(range(1, n + 1)):
            # Reverse rank: rank 1 (best) should get highest probability
            # Use (n - rank + 1) so best rank gets value n, worst rank gets value 1
            reversed_rank = n - rank + 1
            prob = (2 - self.selection_pressure) / n + 2 * reversed_rank * (self.selection_pressure - 1) / (n * (n - 1)) if n > 1 else 1.0
            probabilities.append(prob)
        
        # Normalize probabilities
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        
        # Select based on probabilities
        selected = np.random.choice(
            len(sorted_pop),
            size=num_parents,
            replace=True,
            p=probabilities
        )
        
        return [sorted_pop[int(idx)] for idx in selected]


class TournamentSelection(SelectionStrategy):
    """
    Tournament selection: Randomly selects k individuals and picks the best from each tournament.
    Simple, effective, and maintains diversity.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Args:
            tournament_size: Number of individuals competing in each tournament (default: 3)
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[Dict[str, Any]], num_parents: int) -> List[Dict[str, Any]]:
        """
        Selects parents through tournament competitions.
        """
        self._validate_population(population)
        
        selected = []
        for _ in range(num_parents):
            # Randomly select tournament_size individuals
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            # Select the best from the tournament (Max fitness)
            winner = max(tournament, key=lambda x: x['fitness'])
            selected.append(winner)
        
        return selected


class FitnessProportionateSelection(SelectionStrategy):
    """
    Fitness-proportionate selection (Roulette Wheel): 
    Selection probability proportional to fitness.
    Works well when fitness values are positive and well-scaled.
    """
    
    def __init__(self, use_adjusted_fitness: bool = False):
        """
        Args:
            use_adjusted_fitness: If True, uses adjusted_fitness from individual (for NEAT compatibility)
        """
        self.use_adjusted_fitness = use_adjusted_fitness
    
    def select(self, population: List[Dict[str, Any]], num_parents: int) -> List[Dict[str, Any]]:
        """
        Selects parents with probability proportional to fitness.
        """
        self._validate_population(population)
        
        # Get fitness values
        if self.use_adjusted_fitness:
            fitnesses = [ind.get('adjusted_fitness', ind['fitness']) for ind in population]
        else:
            fitnesses = [ind['fitness'] for ind in population]
        
        # Calculate probabilities
        total_fitness = sum(fitnesses)
        if total_fitness == 0:
            # If all fitnesses are zero, use uniform selection
            probabilities = [1.0 / len(population)] * len(population)
        else:
            probabilities = [f / total_fitness for f in fitnesses]

        # Select based on probabilities
        selected = np.random.choice(
            len(population),
            size=num_parents,
            replace=True,
            p=probabilities
        )
        
        return [population[int(idx)] for idx in selected]

