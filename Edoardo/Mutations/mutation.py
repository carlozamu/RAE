from abc import ABC
from typing import List, Tuple, Callable


class MutationOperator(ABC):
    def __init__(self):
        mutations : List[Tuple[float, Callable]] = [] # List of (probability, mutation_function) tuples

    def mutate(self, genome):
        for prob, mutation_func in self.mutations:
            if self._should_mutate(prob):
                genome = mutation_func(genome)
        return genome

    def _should_mutate(self, probability: float) -> bool:
        import random
        return random.random() < probability