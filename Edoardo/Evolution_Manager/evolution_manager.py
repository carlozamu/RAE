from Edoardo.Crossover.crossover import Crossover
from Edoardo.Mutations.mutator import get_dynamic_config
from Edoardo.Species.species import Species
from Edoardo.Fitness.fitness import Fitness
from Edoardo.Selection.selection import SelectionStrategy
from Edoardo.Generation_Manager.generation_manager import SurvivorSelectionStrategy
from typing import Dict, List, Optional, Any
import random
import hashlib


class EvolutionManager:
    def __init__(self, 
                 selection_strategy: SelectionStrategy, 
                 survivor_strategy: SurvivorSelectionStrategy, 
                 num_parents: int = 2,
                 per_species_hof_size: int = 10,
                 hof_parent_ratio: float = 0.2): #! THIS CAN BE CHANGED AS HYPERPARAMETER, 20% from HoF the rest from the current specie
        """Manages the evolution process across generations and species.
        :param selection_strategy: Strategy to select parents.
        :param survivor_strategy: Strategy to select survivors (e.g. CommaPlus).
        :param num_parents: Number of parents to use for creating offspring.
        :param per_species_hof_size: Maximum number of best individuals to maintain per species in Hall of Fame.
        :param hof_parent_ratio: Ratio of parents to select from Hall of Fame (0.0 to 1.0).
        """
        self.current_generation_index = 0  # Index of the current generation
        self.species = []
        self.selection_strategy = selection_strategy
        self.survivor_strategy = survivor_strategy
        self.num_parents = num_parents
        self.per_species_hof_size = per_species_hof_size
        self.hof_parent_ratio = hof_parent_ratio
        self.species_hall_of_fame: Dict[Species, List[Dict[str, Any]]] = {}

    def create_new_generation(self):
        """
        Creates a new generation of individuals based on the fitness of the current generation.
        Uses the injected survivor_strategy to select the next generation for each species.
        """
        # Update Hall of Fame from current generation
        self._update_hall_of_fame()

        # TODO: Is there a better way to calculate total average fitness?
        total_fitness = 0
        total_individuals = 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                total_fitness += species.cumulative_fitness(self.current_generation_index)
                total_individuals += species.member_count(self.current_generation_index)
        average_fitness = total_fitness / total_individuals if total_individuals > 0 else 0

        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                # Calculate number of offspring to create
                new_species_count = int(
                    species.cumulative_fitness(self.current_generation_index) / average_fitness if average_fitness > 0 else species.member_count(
                        self.current_generation_index))

                # Create offspring
                offsprings = self.create_offsprings(species, new_species_count)

                # Get current generation members for this species
                current_members = species.generations[self.current_generation_index - species.generation_offset] if (
                        self.current_generation_index >= species.generation_offset and self.current_generation_index - species.generation_offset < len(
                    species.generations)) else []

                # Convert offspring Phenotypes to dict format
                offspring_dicts = [{"member": child, "fitness": Fitness.evaluate(child)} for child in offsprings]

                # Determine target size (maintain population size)
                target_size = len(current_members) if current_members else len(offspring_dicts)

                # Select survivors using strategy
                selected_individuals_dicts = self.survivor_strategy.select_survivors(
                    current_population=current_members,
                    offspring_population=offspring_dicts,
                    population_size=target_size
                )

                # Extract members
                selected_individuals = [item['member'] for item in selected_individuals_dicts]

                # Add selected individuals to next generation
                species.add_members(selected_individuals, generation=self.current_generation_index + 1)

        # Advance generation index
        self.current_generation_index += 1

    def get_active_species_count(self):
        """
        Counts the number of species that have members in the current generation.
        """
        count = 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                count += 1
        return count

    def get_latest_generation(self):
        """
        Retrieves all members from the latest generation across all species.
        """
        members = []
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                members.extend(species.get_all_members())
        return None

    @staticmethod
    def _get_individual_signature(individual: Dict[str, Any]) -> str:
        """
        Generate a unique signature for an individual based on its genome structure.
        """
        member = individual.get('member', individual)
        try:
            if hasattr(member, 'genome'):
                genome = member.genome
            elif hasattr(member, '__dict__'):
                genome = getattr(member, 'genome', member)
            else:
                genome = member
            
            if hasattr(genome, 'nodes') and hasattr(genome, 'connections'):
                node_ids = sorted(genome.nodes.keys()) if isinstance(genome.nodes, dict) else []
                # Simple structural hash
                return hashlib.md5(f"{node_ids}_{len(genome.connections)}".encode()).hexdigest()
            else:
                fitness = individual.get('fitness', 0)
                obj_repr = str(member)
                return hashlib.md5(f"{fitness}_{obj_repr}".encode()).hexdigest()
        except Exception:
            fitness = individual.get('fitness', 0)
            obj_id = id(member)
            return f"{fitness}_{obj_id}"

    def _update_hall_of_fame(self):
        """
        Updates the per-species Hall of Fame with the best individuals from the current generation.
        """
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                # Get all members from current generation
                generation = self.current_generation_index
                relative_generation = generation - species.generation_offset
                
                if relative_generation < 0 or relative_generation >= len(species.generations):
                    continue
                
                members = species.generations[relative_generation]
                if not members:
                    continue
                
                # Sort by fitness (descending)
                sorted_members = sorted(members, key=lambda x: x['fitness'], reverse=True)
                
                existing_hof = self.species_hall_of_fame.get(species, [])
                all_candidates = existing_hof + sorted_members
                
                # Remove duplicates
                seen_signatures = set()
                unique_candidates = []
                for candidate in all_candidates:
                    signature = self._get_individual_signature(candidate)
                    if signature not in seen_signatures:
                        seen_signatures.add(signature)
                        unique_candidates.append(candidate)
                
                sorted_unique = sorted(unique_candidates, key=lambda x: x['fitness'], reverse=True)
                self.species_hall_of_fame[species] = sorted_unique[:self.per_species_hof_size]

    def create_offsprings(self, species, num_offsprings: int):
        """
        Creates num_offsprings offspring from a given species.
        """
        offsprings = []
        for _ in range(num_offsprings):
            healthy_child = False
            child = None
            while not healthy_child:
                parents = self.select_parents(species, self.num_parents)
                if len(parents) < self.num_parents:
                    break
                child = Crossover.create_offspring(parents[0].genome, parents[1].genome)
                # TODO: add mutations
                new_species = True
                next_generation = self.current_generation_index + 1
                for s in self.species:
                    if s.belongs_to_species(child,
                                            next_generation) and s != species and s.generation_offset == next_generation:
                        healthy_child = True
                        new_species = False
                        s.add_members([child], generation=next_generation)
                        break
                    if s.belongs_to_species(child, next_generation) and s != species:
                        healthy_child = False
                        new_species = False
                        break
                    elif s.belongs_to_species(child, next_generation) and s == species:
                        healthy_child = True
                        new_species = False
                        offsprings.append(child)
                        break
                if new_species:
                    healthy_child = True
                    self.species.append(Species([child], generation=next_generation))
        return offsprings

    def select_parents(self, species: Species, num_parents: int) -> List:
        """
        Select parents from a species using the injected selection strategy AND Hall of Fame.

        :param species: The species to select parents from
        :param num_parents: Total number of parents to select
        :return: List of Phenotype objects to use as parents
        """
        # Get HoF members
        hof_members = self.species_hall_of_fame.get(species, [])
        
        # Calculate split
        num_hof_parents = int(num_parents * self.hof_parent_ratio) if self.per_species_hof_size > 0 and hof_members else 0
        num_species_parents = num_parents - num_hof_parents
        
        selected_parents = []
        
        # 1. Select from HoF (Randomly, as a simple diversity injection)
        if num_hof_parents > 0:
            hof_selected = random.sample(hof_members, min(num_hof_parents, len(hof_members)))
            selected_parents.extend([member['member'] for member in hof_selected])

        # 2. Select from Current Species (using Strategy)
        if num_species_parents > 0:
            relative_gen = self.current_generation_index - species.generation_offset
            if 0 <= relative_gen < len(species.generations):
                population = species.generations[relative_gen]
            else:
                population = []

            if population:
                selected_dicts = self.selection_strategy.select(population, num_species_parents)
                selected_parents.extend([item['member'] for item in selected_dicts])

        # Fill potential gaps if not enough parents found
        while len(selected_parents) < num_parents:
            # Fallback source: combined HoF + Population or just Population
            # If we missed parents, likely population was too small or something failed.
            # Try grabbing random valid member
            relative_gen = self.current_generation_index - species.generation_offset
            if 0 <= relative_gen < len(species.generations) and species.generations[relative_gen]:
                 selected_parents.append(random.choice(species.generations[relative_gen])['member'])
            elif hof_members:
                 selected_parents.append(random.choice(hof_members)['member'])
            else:
                break
        
        return selected_parents[:num_parents]