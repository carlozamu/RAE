from Crossover.crossover import Crossover
from Utils.LLM import LLM
from Genome.agent_genome import AgentGenome
from Mutations.mutator import Mutator
from Phenotype.phenotype import Phenotype
from Species.species import Species
from Fitness.fitness import Fitness
from Selection.selection import SelectionStrategy
from Generation_Manager.generation_manager import SurvivorSelectionStrategy
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import random
import hashlib


class EvolutionManager:
    def __init__(self, 
                 selection_strategy: SelectionStrategy, 
                 survivor_strategy: SurvivorSelectionStrategy, 
                 mutator: Mutator,
                 fitness_evaluator: Fitness,
                 llm_client: LLM,
                 initial_population: List[Phenotype],
                 dataset_manager: Any, # Can be CLUTTRManager or generic
                 num_parents: int = 2,
                 per_species_hof_size: int = 10,
                 hof_parent_ratio: float = 0.2, #! THIS CAN BE CHANGED AS HYPERPARAMETER, 20% from HoF the rest from the current specie
                 top_r = 10, # Top R individuals considered for selection in fallback mechanism for species
                 c1 = 1.0,  # Coefficient for excess genes in species compatibility
                 c2 = 1.0,  # Coefficient for disjoint genes in species compatibility
                 c3 = 0.4,  # Coefficient for average node differences in species compatibility
                 protection_base = 3,  # Base number of generations for protection of species
                 adjust_rate_protected_species = 1.5,  # Adjustment rate for protected species
                 compatibility_threshold = 3.0  # Threshold for species compatibility
    ):
        """Manages the evolution process across generations and species.
        :param selection_strategy: Strategy to select parents.
        :param survivor_strategy: Strategy to select survivors (e.g. CommaPlus).
        :param num_parents: Number of parents to use for creating offspring.
        :param per_species_hof_size: Maximum number of best individuals to maintain per species in Hall of Fame.
        :param hof_parent_ratio: Ratio of parents to select from Hall of Fame (0.0 to 1.0).
        :param dataset_manager: Object capable of providing get_batch(size).
        """
        Species.top_r = top_r
        Species.c1 = c1
        Species.c2 = c2
        Species.c3 = c3
        Species.protection_base = protection_base
        Species.adjust_rate_protected_species = adjust_rate_protected_species
        Species.compatibility_threshold = compatibility_threshold
        self.current_generation_index = 0  # Index of the current generation
        self.species: List[Species] = self._get_initial_species(initial_population) ##FLAG##
        #TODO: initialize the species given a population
        self.selection_strategy = selection_strategy
        self.survivor_strategy = survivor_strategy
        self.mutator = mutator
        self.num_parents = num_parents
        self.per_species_hof_size = per_species_hof_size
        self.hof_parent_ratio = hof_parent_ratio
        self.llm_client = llm_client   
        self.fitness_evaluator = fitness_evaluator
        self.dataset_manager = dataset_manager
    
    def _get_initial_species(self, initial_population: List[Phenotype]) -> List[Species]:
        """
        Initializes species from the initial population.
        Creates a first species with the first individual, then assigns others to existing species or creates new ones.
        :param initial_population: List of Phenotype individuals to speciate.
        :return: List of Species objects.
        """
        species_list = []
        first_species = Species([initial_population[0]], generation=self.current_generation_index, max_hof_size=self.per_species_hof_size)
        initial_population = initial_population[1:]
        species_list.append(first_species)
        for individual in initial_population:
            added_to_species = False
            for species in species_list:
                if species.belongs_to_species(individual.genome, self.current_generation_index):
                    species.add_members([individual], generation=self.current_generation_index)
                    added_to_species = True
                    break
            if not added_to_species:
                new_species = Species([individual], generation=self.current_generation_index, max_hof_size=self.per_species_hof_size)
                species_list.append(new_species)
        return species_list


    def get_problem_pool(self, size: int = 3) -> List[Dict[str, str]]:
        """
        Returns a list of problems from the dataset manager.
        """
        return self.dataset_manager.get_batch(size=size)
    
    def _compute_normalized_species_counts(self, average_fitness: float, total_individuals: int) -> Dict[Species, int]:
        """
        Computes normalized offspring counts for each species based on average fitness.
        :param average_fitness: Average fitness across all species.
        :param total_individuals: Total number of individuals across all species.
        :return: Dictionary mapping Species to their normalized offspring counts.
        """
        species_counts = {}
        for species in self.species:
            species_counts[species] = 0
            if species.last_generation_index() == self.current_generation_index:
                # Calculate number of offspring to create
                new_species_count = species.adjusted_offspring_count(average_fitness, self.current_generation_index)
                species_counts[species] = new_species_count
        normalized_species_counts = {}
        total_counts = sum(species_counts.values())
        for species, count in species_counts.items():
            if total_counts > 0:
                normalized_species_counts[species] = int((count / total_counts) * total_individuals)
            else:
                normalized_species_counts[species] = 0
        return normalized_species_counts

    async def create_new_generation(self)-> list[Tuple[str, Phenotype]]: # tuple(species_name, individual):
        """
        Creates a new generation of individuals based on the fitness of the current generation.
        Uses the injected survivor_strategy to select the next generation for each species.
        """
        # Update Hall of Fame for all active species
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                species.update_hall_of_fame()

        # Generate Problem Pool for this generation
        # We use the same pool for all species to ensure fair comparison
        problem_pool = self.get_problem_pool(size=3) # Configurable size

        # TODO: Is there a better way to calculate total average fitness?
        total_fitness = 0
        total_individuals = 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                total_fitness += species.cumulative_fitness(self.current_generation_index)
                total_individuals += species.member_count(self.current_generation_index)
        average_fitness = total_fitness / total_individuals if total_individuals > 0 else 0
        species_counts = self._compute_normalized_species_counts(average_fitness, total_individuals)
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                new_species_count = species_counts.get(species, 0)
                if new_species_count > 0:
                    # Create offspring
                    offsprings = self.create_offsprings(species, new_species_count)

                    # Get current generation members for this species
                    current_members = species.get_all_members_from_generation(self.current_generation_index)
                    current_phenotypes = [m['member'] for m in current_members]

                    # Evaluate ALL candidates (Current + Offsprings) on the NEW problem pool
                    # This ensures fairness as the problem set changes/rotates.
                    all_candidates = current_phenotypes + offsprings
                    self.fitness_evaluator.evaluate_population(all_candidates, problem_pool)

                    # Convert offspring Phenotypes to dict format (now with updated fitness)
                    offspring_dicts = [{"member": child, "fitness": child.fitness} for child in offsprings]
                    
                    # Update current_members dicts with new fitness
                    # Note: This updates the dicts used for selection, effectively re-evaluating parents.
                    # We create NEW dicts to avoid modifying the historical record in species.generations if that's preferred,
                    # BUT for CommaPlus/Elitism, we need comparable fitness.
                    current_pop_dicts = [{"member": p, "fitness": p.fitness} for p in current_phenotypes]

                    # Determine target size (keep total population constant)
                    target_size = len(offsprings)

                    # Select survivors using strategy
                    selected_individuals_dicts = self.survivor_strategy.select_survivors(
                        current_population=current_pop_dicts,
                        offspring_population=offspring_dicts,
                        population_size=target_size
                    )

                    # Extract members
                    selected_individuals = [item['member'] for item in selected_individuals_dicts]

                    # Add selected individuals to next generation
                    species.add_members(selected_individuals, generation=self.current_generation_index + 1)

        # Advance generation index
        self.current_generation_index += 1

        return self.get_latest_generation()

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
                species_members = species.get_all_members_from_generation(self.current_generation_index)
                members.extend([(species.id, member) for member in species_members])
        return members

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
                child = Crossover.create_offspring(parents[0].genome, parents[1].genome)
                # Calculate dynamic mutation probabilities
                child_mutation_config = self.mutator.get_dynamic_config(
                    generation=self.current_generation_index, 
                    parent_node_count=len(child.nodes)
                )
                #mutate offspring
                child: AgentGenome = asyncio.run(self.mutator.mutate(genome=child, runtime_config=child_mutation_config))
                child = Phenotype(genome=child, llm_client=self.llm_client, fitness_evaluator=self.fitness_evaluator)
                #TODO: switch to Phenotype
                new_species = True
                next_generation = self.current_generation_index + 1
                for s in self.species:
                    if s.last_generation_index() >= self.current_generation_index:
                        if s.belongs_to_species(child, self.current_generation_index) and s != species and s.generation_offset == next_generation:
                            # child belongs to newly created species along another individual
                            healthy_child = True
                            new_species = False
                            s.add_members([child], generation=next_generation)
                            break
                        if s.belongs_to_species(child, self.current_generation_index) and s != species:
                            # child belongs to another existing species. Abort and retry
                            healthy_child = False
                            new_species = False
                            break
                        elif s.belongs_to_species(child, self.current_generation_index) and s == species:
                            # child belongs to the same species as parents
                            healthy_child = True
                            new_species = False
                            offsprings.append(child)
                            break
                if new_species:
                    # child creates a new species
                    healthy_child = True
                    self.species.append(Species([child], generation=next_generation, max_hof_size=self.per_species_hof_size))
        return offsprings

    def select_parents(self, species: Species, num_parents: int) -> List:
        """
        Select parents from a species using the injected selection strategy AND Hall of Fame.

        :param species: The species to select parents from
        :param num_parents: Total number of parents to select
        :return: List of Phenotype objects to use as parents
        """
        # Get HoF members directly from the Species
        hof_members = species.hall_of_fame
        
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