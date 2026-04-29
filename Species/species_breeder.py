"""
Species Breeder Module.
Handles the micro-level reproduction of a single isolated species.
Applies Elitism, Rank-Based Parent Selection, Crossover, and Mutation.
"""
from typing import List
from Genome.agent_genome import AgentGenome
from Selection.selection import RankBasedSelection
from Crossover.crossover import Crossover
from Mutations.mutator import Mutator
import math

class SpeciesBreeder:
    def __init__(self, 
                 selector: RankBasedSelection, 
                 mutator: Mutator,
                 elitism_ratio: float):
        """
        Args:
            selector: The RankBasedSelection instance (singleton) to pick parents.
            mutator: The Mutator instance (singleton) handling mutations.
            elitism_ratio: Percentage of target size reserved for pure clones. 
                           At 0.2, species < 5 get 0 elites, 5-9 get 1, 10-14 get 2, etc.
        """
        self.selector = selector
        self.mutator = mutator
        self.elitism_ratio = elitism_ratio

    async def breed_next_generation(self, 
                                    current_species_members: List[AgentGenome], 
                                    target_size: int) -> List[AgentGenome]:
        """
        Creates the next generation for a specific species.
        """
        if target_size <= 0:
            return []
            
        if not current_species_members:
            raise ValueError("Cannot breed an empty species.")

        # 1. Sort current members by fitness (Descending: Highest is best)
        sorted_members = sorted(current_species_members, key=lambda x: x.fitness, reverse=True)
        
        next_generation: List[AgentGenome] = []

        # 2. Elitism: Clone the absolute best individuals directly
        num_elites = math.floor(target_size * self.elitism_ratio)
        num_elites = min(num_elites, len(sorted_members), target_size) # Cap it safely
        
        for i in range(num_elites):
            elite_clone = sorted_members[i].copy()
            next_generation.append(elite_clone)

        # 3. Calculate remaining slots for children
        num_children_needed = target_size - len(next_generation)
        
        if num_children_needed <= 0:
            return next_generation

        # 4. Generate Children via Selection, Crossover, and Mutation
        for _ in range(num_children_needed):
            # Select 2 parents using Rank-Based Selection
            parents = self.selector.select(current_species_members, num_parents=2)
            p1, p2 = parents[0], parents[1]
            
            # Crossover
            child = Crossover.create_offspring(p1, p2)
            
            # Mutate
            child = await self.mutator.mutate(child) 
            
            # Reset the child's fitness so it must be evaluated in the macro loop
            child.fitness = 0.0 
            
            next_generation.append(child)

        return next_generation