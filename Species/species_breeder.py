"""
Species Breeder Module.
Handles the micro-level reproduction of a single isolated species.
Applies Elitism, Rank-Based Parent Selection, Crossover, and Mutation.
"""
import asyncio
from typing import List

import numpy as np
from Genome.agent_genome import AgentGenome
from Selection.selection import RankBasedSelection
from Crossover.crossover import Crossover
from Mutations.mutator import Mutator
import math

class SpeciesBreeder:
    def __init__(self, 
                 selector: RankBasedSelection, 
                 mutator: Mutator,
                 elitism_ratio: float = 0.2):
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
                                        target_size: int,
                                        generation: int) -> List[AgentGenome]:
        """
        Creates the next generation for a specific species concurrently.
        """
        if target_size <= 0:
            return []
            
        if not current_species_members:
            raise ValueError("Cannot breed an empty species.")

        # 1. Sort current members by fitness (Descending: Highest is best)
        sorted_members = sorted(current_species_members, key=lambda x: x.accuracy, reverse=True)
        
        next_generation: List[AgentGenome] = []

        # 2. Elitism: Clone the absolute best individuals directly
        num_elites = math.floor(len(current_species_members) * self.elitism_ratio)
        num_elites = min(num_elites, len(sorted_members), target_size) # Cap it safely
        
        for i in range(num_elites):
            elite_clone = sorted_members[i].copy()
            next_generation.append(elite_clone)

        # 3. Calculate remaining slots for children
        num_children_needed = target_size - len(next_generation)
        
        if num_children_needed <= 0:
            return next_generation

        # --- CONCURRENCY UPGRADE ---
        
        # Define a single asynchronous pipeline for a child
        async def generate_single_child() -> AgentGenome:
            # Select 2 parents
            parents = self.selector.select(current_species_members, num_parents=2)
            p1, p2 = parents[0], parents[1]

            # Crossover or Clone
            if np.random.rand() > 0.25:
                child = Crossover.create_offspring(p1, p2)
            else:
                child = p1.copy() if p1.accuracy >= p2.accuracy else p2.copy()

            # Mutate
            child = await self.mutator.mutate(child, current_generation=generation) 

            # Reset fitness
            child.fitness = 0.0 
            child.accuracy = 0.0

            # Evaluate
            child.evaluated = False

            return child

        # Create a list of tasks, but don't run them sequentially
        child_tasks = [generate_single_child() for _ in range(num_children_needed)]
        
        # Fire them all off concurrently and wait for all to finish
        mutated_children = await asyncio.gather(*child_tasks)
        
        # Add them to the next generation pool
        next_generation.extend(mutated_children)

        return next_generation