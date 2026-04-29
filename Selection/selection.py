"""
Selection logic for the Reasoning Agent Engine (RAE).
Handles Parent Selection using a Rank-Based approach to mitigate non-linear LLM fitness scores.
"""
from abc import ABC, abstractmethod
from typing import List, Protocol, TypeVar
import numpy as np

# 1. Type Safety Protocol
class Evolvable(Protocol):
    """Protocol defining the minimum requirements for a genome in the selection process."""
    fitness: float

T = TypeVar('T', bound=Evolvable)

# 2. Base Architecture
class SelectionStrategy(ABC):
    """Base class establishing the contract for parent selection."""
    
    @abstractmethod
    def select(self, population: List[T], num_parents: int) -> List[T]:
        """Selects a specified number of parents from the population."""
        pass
    
    def _validate_population(self, population: List[T]) -> None:
        """Ensures the population is valid and compatible."""
        if not population:
            raise ValueError("Population cannot be empty.")
        if not all(hasattr(ind, 'fitness') for ind in population):
            raise ValueError("Each individual in the population must have a 'fitness' attribute.")

# 3. The Concrete Implementation
class RankBasedSelection(SelectionStrategy):
    """
    Rank-based parent selection.
    Sorts the population by fitness (highest to lowest). Assigns selection 
    probabilities linearly based on rank rather than raw score.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Args:
            selection_pressure: Controls the statistical bias towards top performers.
                Range: [1.0, 2.0]. 
                1.0 = purely random selection (no pressure).
                2.0 = maximum deterministic bias toward Rank 1.
        """
        if not (1.0 <= selection_pressure <= 2.0):
            raise ValueError("Selection pressure must be between 1.0 and 2.0")
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[T], num_parents: int) -> List[T]:
        self._validate_population(population)
        
        # Sort descending: Highest fitness = index 0 (Rank 1)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        n = len(sorted_pop)
        
        # Edge case: If population is 1, just return clones of it
        if n == 1:
            return [sorted_pop[0]] * num_parents
            
        # Calculate stochastic linear ranking probabilities
        probabilities = []
        for i in range(n):
            rank = i + 1  # 1-indexed rank
            reversed_rank = n - rank + 1  # Invert so Rank 1 gets the highest multiplier
            
            prob = (2 - self.selection_pressure) / n + \
                   (2 * reversed_rank * (self.selection_pressure - 1)) / (n * (n - 1))
            probabilities.append(prob)
        
        # Convert to numpy array and normalize to resolve floating-point inaccuracies
        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()
        
        # Sample parents with replacement
        selected_indices = np.random.choice(
            n,
            size=num_parents,
            replace=True,
            p=probabilities
        )
        
        return [sorted_pop[idx] for idx in selected_indices]
    