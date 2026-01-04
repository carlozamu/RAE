"""
Generation management and survivor selection strategies for evolutionary algorithms.
Handles the transition between generations and determines which individuals survive.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from collections import deque
import hashlib


class SurvivorSelectionStrategy(ABC):
    """
    Base class for survivor selection strategies.
    Determines which individuals from current and offspring populations survive to next generation.
    """
    
    @abstractmethod
    def select_survivors(
        self,
        current_population: List[Dict[str, Any]],
        offspring_population: List[Dict[str, Any]],
        population_size: int
    ) -> List[Dict[str, Any]]:
        """
        Select survivors from current and offspring populations.
        
        Args:
            current_population: Current generation individuals (dict with 'member' and 'fitness')
            offspring_population: Newly created offspring (dict with 'member' and 'fitness')
            population_size: Target population size for next generation
            
        Returns:
            List of survivors for next generation
        """
        pass


class CommaPlusStrategy(SurvivorSelectionStrategy):
    """
    (μ+λ) strategy: Selects best individuals from both parents and offspring.
    Preserves best solutions but may slow down exploration.
    """
    
    def select_survivors(
        self,
        current_population: List[Dict[str, Any]],
        offspring_population: List[Dict[str, Any]],
        population_size: int
    ) -> List[Dict[str, Any]]:
        """
        Combines current and offspring, then selects top N by fitness.
        """
        # Combine both populations
        combined = current_population + offspring_population
        
        # Sort by fitness (ascending) and select top N
        sorted_combined = sorted(combined, key=lambda x: x['fitness'], reverse=False)
        
        return sorted_combined[:population_size]


class CommaStrategy(SurvivorSelectionStrategy):
    """
    (μ,λ) strategy: Selects survivors only from offspring population.
    More explorative, forces continuous improvement.
    Requires λ >= μ (more offspring than parents).
    """
    
    def select_survivors(
        self,
        current_population: List[Dict[str, Any]],
        offspring_population: List[Dict[str, Any]],
        population_size: int
    ) -> List[Dict[str, Any]]:
        """
        Selects top N from offspring only (parents are discarded).
        """
        if len(offspring_population) < population_size:
            raise ValueError(
                f"Comma strategy requires at least {population_size} offspring, "
                f"but only {len(offspring_population)} were generated"
            )
        
        # Sort offspring by fitness and select top N
        sorted_offspring = sorted(offspring_population, key=lambda x: x['fitness'], reverse=False)
        
        return sorted_offspring[:population_size]


class HallOfFameStrategy(SurvivorSelectionStrategy):
    """
    Hall of Fame strategy: Maintains a separate archive of best-ever individuals.
    Combines current best with historical best individuals.
    Prevents loss of good solutions found in earlier generations.
    """
    
    def __init__(self, hall_of_fame_size: int = 10):
        """
        Args:
            hall_of_fame_size: Maximum number of best-ever individuals to maintain
        """
        self.hall_of_fame_size = hall_of_fame_size
        self.hall_of_fame: List[Dict[str, Any]] = []
    
    @staticmethod
    def _get_individual_signature(individual: Dict[str, Any]) -> str:
        """
        Generate a unique signature for an individual based on its genome structure.
        Uses a hash of the genome structure (nodes and connections) rather than object id.
        
        Args:
            individual: Individual dict with 'member' (Phenotype) key
            
        Returns:
            String signature that uniquely identifies the genome structure
        """
        member = individual.get('member', individual)
        
        # Try to access the genome from the phenotype
        # Phenotype typically has a .genome attribute
        try:
            if hasattr(member, 'genome'):
                genome = member.genome
            elif hasattr(member, '__dict__'):
                # Fallback: try to find genome in attributes
                genome = getattr(member, 'genome', member)
            else:
                genome = member
            
            # Create signature from genome structure
            # Use nodes and connections to create a unique identifier
            if hasattr(genome, 'nodes') and hasattr(genome, 'connections'):
                # Sort node IDs for consistent hashing
                node_ids = sorted(genome.nodes.keys()) if isinstance(genome.nodes, dict) else []
                
                # Create connection signatures (preserve direction, sort for consistency)
                connections = []
                if hasattr(genome, 'connections') and genome.connections:
                    for conn in genome.connections:
                        if hasattr(conn, 'in_node') and hasattr(conn, 'out_node'):
                            # Preserve direction: (in_node, out_node) - don't sort!
                            # Also include enabled status if available
                            enabled = getattr(conn, 'enabled', True)
                            conn_tuple = (str(conn.in_node), str(conn.out_node), enabled)
                            connections.append(conn_tuple)
                    # Sort connections list for consistent hashing (but preserve direction in each tuple)
                    connections.sort()
                
                # Create hashable signature
                signature_parts = (
                    tuple(node_ids),
                    tuple(sorted(connections)),
                    str(genome.start_node_id) if hasattr(genome, 'start_node_id') else None,
                    str(genome.end_node_id) if hasattr(genome, 'end_node_id') else None
                )
                signature_str = str(signature_parts)
                # Use hash for efficiency (MD5 is fine for this use case)
                return hashlib.md5(signature_str.encode()).hexdigest()
            else:
                # Fallback: use fitness + a hash of the object representation
                fitness = individual.get('fitness', 0)
                obj_repr = str(member)
                return hashlib.md5(f"{fitness}_{obj_repr}".encode()).hexdigest()
        except Exception:
            # Ultimate fallback: use fitness + object id (less reliable but better than nothing)
            fitness = individual.get('fitness', 0)
            obj_id = id(member)
            return f"{fitness}_{obj_id}"
    
    def select_survivors(
        self,
        current_population: List[Dict[str, Any]],
        offspring_population: List[Dict[str, Any]],
        population_size: int
    ) -> List[Dict[str, Any]]:
        """
        Updates hall of fame, then combines with current/offspring to select survivors.
        """
        # Update hall of fame with best from current and offspring
        combined = current_population + offspring_population
        all_individuals = combined + self.hall_of_fame
        
        # Sort all by fitness and update hall of fame
        sorted_all = sorted(all_individuals, key=lambda x: x['fitness'], reverse=False)
        self.hall_of_fame = sorted_all[:self.hall_of_fame_size]
        
        # Select survivors: mix of best from combined population and hall of fame
        # Strategy: take top (population_size - hall_of_fame_size) from combined,
        # then add hall of fame members
        num_from_combined = max(0, population_size - len(self.hall_of_fame))
        sorted_combined = sorted(combined, key=lambda x: x['fitness'], reverse=False)
        
        survivors = sorted_combined[:num_from_combined]
        
        # Add hall of fame members (avoid duplicates using structural signatures)
        survivor_signatures = {self._get_individual_signature(s) for s in survivors}
        for hof_member in self.hall_of_fame:
            if len(survivors) >= population_size:
                break
            hof_signature = self._get_individual_signature(hof_member)
            if hof_signature not in survivor_signatures:
                survivors.append(hof_member)
                survivor_signatures.add(hof_signature)
        
        # If still need more, fill from combined
        if len(survivors) < population_size:
            for ind in sorted_combined:
                if len(survivors) >= population_size:
                    break
                ind_signature = self._get_individual_signature(ind)
                if ind_signature not in survivor_signatures:
                    survivors.append(ind)
                    survivor_signatures.add(ind_signature)
        
        return survivors[:population_size]
    
    def get_hall_of_fame(self) -> List[Dict[str, Any]]:
        """Returns the current hall of fame."""
        return self.hall_of_fame.copy()


class GenerationManager:
    """
    Manages generation transitions, population evolution, and survivor selection.
    Coordinates selection strategies with generation management.
    """
    
    def __init__(
        self,
        population_size: int,
        survivor_strategy: SurvivorSelectionStrategy,
        generation: int = 0
    ):
        """
        Args:
            population_size: Target size for each generation
            survivor_strategy: Strategy for selecting survivors (CommaPlus, Comma, or HallOfFame)
            generation: Starting generation number
        """
        self.population_size = population_size
        self.survivor_strategy = survivor_strategy
        self.current_generation = generation
        self.generation_history: List[List[Dict[str, Any]]] = []
    
    def advance_generation(
        self,
        current_population: List[Dict[str, Any]],
        offspring_population: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Advances to next generation by selecting survivors.
        
        Args:
            current_population: Current generation individuals
            offspring_population: Newly created offspring
            
        Returns:
            Next generation population
        """
        # Store current generation in history
        self.generation_history.append(current_population.copy())
        
        # Select survivors using the chosen strategy
        next_population = self.survivor_strategy.select_survivors(
            current_population=current_population,
            offspring_population=offspring_population,
            population_size=self.population_size
        )
        
        # Advance generation counter
        self.current_generation += 1
        
        return next_population
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Returns statistics about the evolution process.
        """
        if not self.generation_history:
            return {
                'current_generation': self.current_generation,
                'total_generations': 0,
                'best_fitness_history': []
            }
        
        best_fitness_history = []
        for gen in self.generation_history:
            if gen:
                best_fitness = min(ind['fitness'] for ind in gen)
                best_fitness_history.append(best_fitness)
        
        return {
            'current_generation': self.current_generation,
            'total_generations': len(self.generation_history),
            'best_fitness_history': best_fitness_history,
            'population_size': self.population_size
        }
    
    def reset(self, generation: int = 0) -> None:
        """Reset the generation manager to initial state."""
        self.current_generation = generation
        self.generation_history = []
        if isinstance(self.survivor_strategy, HallOfFameStrategy):
            self.survivor_strategy.hall_of_fame = []

