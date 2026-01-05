import math
from tqdm import tqdm
from Crossover.crossover import Crossover
from Utils.MarkDownLogger import md_logger
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
                 c3 = 1.0,  # Coefficient for different edges count in species compatibility
                 c4 = 0.5,  # Coefficient for average node differences in species compatibility
                 protection_base = 3,  # Base number of generations for protection of species
                 adjust_rate_protected_species = 1.5,  # Adjustment rate for protected species
                 compatibility_threshold = 0.51  # Threshold for species compatibility
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
        Species.c4 = c4
        Species.protection_base = protection_base
        Species.adjust_rate_protected_species = adjust_rate_protected_species
        Species.compatibility_threshold = compatibility_threshold
        self.current_generation_index = 0  # Index of the current generation
        self.per_species_hof_size = per_species_hof_size
        self.species: List[Species] = self._get_initial_species(initial_population) 
        self.selection_strategy = selection_strategy
        self.survivor_strategy = survivor_strategy
        self.mutator = mutator
        self.num_parents = num_parents
        self.hof_parent_ratio = hof_parent_ratio
        self.llm_client = llm_client   
        self.fitness_evaluator = fitness_evaluator
        self.dataset_manager = dataset_manager

        #print(f"   🌿 Evolution Manager initialized with {len(self.species)} species.")
    
    def _get_initial_species(self, initial_population: List[Phenotype]) -> List[Species]:
        """
        Initializes species from the initial population.
        Creates a first species with the first individual, then assigns others to existing species or creates new ones.
        Includes safety checks and debug logging.
        """
        print(f"   🌱 Speciating Initial Population of {len(initial_population)} individuals...")
        
        if not initial_population:
            print("   ⚠️ Warning: Initial population is empty. returning empty species list.")
            return []

        species_list = []
        
        # Create the first anchor species
        # Note: Passing list [initial_population[0]] because Species expects a list
        first_species = Species(
            [initial_population[0]], 
            generation=self.current_generation_index, 
            max_hof_size=self.per_species_hof_size
        )
        species_list.append(first_species)
        
        # Iterate through the rest
        remaining_pop = initial_population[1:]
        
        for i, individual in enumerate(remaining_pop):
            added_to_species = False
            
            # Try to fit into existing species
            for species in species_list:
                # belongs_to_species typically checks genetic distance
                if species.belongs_to_species(individual, self.current_generation_index):
                    species.add_members([individual], generation=self.current_generation_index)
                    added_to_species = True
                    break
            
            # If no match, create new species
            if not added_to_species:
                new_species = Species(
                    [individual], 
                    generation=self.current_generation_index, 
                    max_hof_size=self.per_species_hof_size
                )
                species_list.append(new_species)
        
        print(f"   ✅ Initial Speciation Complete. Created {len(species_list)} distinct species.")
        return species_list

    def get_problem_pool(self) -> List[Dict[str, str]]:
        """
        Returns a list of problems from the dataset manager.
        """
        return self.dataset_manager.get_batch()
    
    def _compute_normalized_species_counts(self, average_fitness: float, total_individuals: int) -> Dict[Species, int]:
        """
        Computes normalized offspring counts for each species based on average fitness.
        :param average_fitness: Average fitness across all species.
        :param total_individuals: Total number of individuals across all species.
        :return: Dictionary mapping Species to their normalized offspring counts.
        """
        species_counts = {}
        for species in self.species:
            species_counts[species] = 0.0
            if species.last_generation_index() >= self.current_generation_index:
                # Calculate number of offspring to create
                new_species_count = species.adjusted_offspring_count(average_fitness, self.current_generation_index)
                species_counts[species] = new_species_count
        normalized_species_counts = {}
        total_counts = sum(species_counts.values())
        for species, count in species_counts.items():
            if total_counts > 0:
                normalized_species_counts[species] = round((count / total_counts) * 30)
            else:
                normalized_species_counts[species] = 0
        md_logger.log_header(f"Species Offspring Allocation for Generation {self.current_generation_index + 1}")
        for species, count in normalized_species_counts.items():
            md_logger.log_event(f"Species {species.id}: Allocated Offspring = {count} with nonormalized count {species_counts[species]:.4f}")

        # compute total allocated
        total_allocated = sum(normalized_species_counts.values())
        if total_allocated < 30:
            # Distribute remaining slots randomly
            species_list = list(normalized_species_counts.keys())
            while total_allocated < 30:
                chosen_species = random.choice(species_list)
                normalized_species_counts[chosen_species] += 1
                total_allocated += 1
        # recompute total allocated
        total_allocated = sum(normalized_species_counts.values())
        md_logger.log_event(f"Total Allocated Offspring after adjustment: {total_allocated}")
    
        return normalized_species_counts
        
    async def create_new_generation(self) -> list[Tuple[int, Phenotype]]:
        """
        Creates a new generation of individuals based on the fitness of the current generation.
        Uses the injected survivor_strategy to select the next generation for each species.
        Includes robust error handling and debug logging.
        """
        print(f"\n--- 🧬 Evolution Manager: Starting Generation {self.current_generation_index + 1} ---")
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                species.update_hall_of_fame()
        problem_pool: List[Dict[str, str]] = self.get_problem_pool()
        total_fitness = 0.0
        total_individuals = 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                fitness = species.cumulative_fitness(self.current_generation_index)
                if not math.isinf(fitness):
                    total_fitness += fitness
                    total_individuals += species.member_count(self.current_generation_index)
        average_fitness = (total_fitness / total_individuals) if total_individuals > 0 else 0
        print(f"      -> Global Avg Fitness: {average_fitness:.4f}| Total Fitness: {total_fitness} | Total Pop: {total_individuals}")
        species_counts = self._compute_normalized_species_counts(average_fitness, total_individuals)
        active_species_count = 0
        pbar = tqdm(total=30, desc=f"Gen {self.current_generation_index+1} - Starting Offsprings Creation", unit="child")
        species_tuple = tuple(self.species)  # To avoid modification during iteration
        for species in species_tuple:
            # Skip species that died out previously
            if species.last_generation_index() != self.current_generation_index:
                continue
            new_species_target_count = species_counts.get(species, 1)
            # If the species is extinct due to poor performance
            if new_species_target_count <= 0:
                print(f"   💀 Species {species.id} has gone extinct (0 slots allocated).")
                continue
            active_species_count += 1
            print(f"      Creating {new_species_target_count} offspring...")
            offsprings = await self.create_offsprings(species, new_species_target_count, problem_pool=problem_pool, pbar=pbar)
            
            if not offsprings:
                print(f"      ⚠️ Warning: No offspring produced for species {species.id}. Skipping.")
                continue
            current_members = species.get_all_members_from_generation(self.current_generation_index)
            current_phenotypes = [item['member'] for item in current_members]
            
            all_candidates = current_phenotypes + offsprings
            await self.fitness_evaluator.evaluate_population(population=all_candidates, problem_pool=problem_pool)
            offspring_dicts = [{"member": child, "fitness": child.genome.fitness} for child in offsprings]
            current_pop_dicts = [{"member": p, "fitness": p.genome.fitness} for p in current_phenotypes]
            selected_individuals_dicts = self.survivor_strategy.select_survivors(
                current_population=current_pop_dicts,
                offspring_population=offspring_dicts,
                population_size=len(offsprings)
            )
            selected_individuals = [item['member'] for item in selected_individuals_dicts]
            print(f"      ✅ Selected {len(selected_individuals)} survivors for next gen.")
            species.add_members(selected_individuals, generation=self.current_generation_index + 1)

        pbar.close()
        
        if active_species_count == 0:
            print("   ⚠️ WARNING: All species have gone extinct!")
            # Optional: Re-seed population here?
        
        self.current_generation_index += 1
        print(f"--- ✅ Generation {self.current_generation_index} Finalized ---\n")

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

    def get_latest_generation(self) -> list[Tuple[int, Phenotype]]:
        """
        Retrieves all members from the latest generation across all species.
        """
        members: List[Tuple[int, Phenotype]] = []
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                species_members = species.get_all_members_from_generation(self.current_generation_index)
                members.extend([(species.id, member['member']) for member in species_members])
        return members

    @staticmethod
    def _get_individual_signature(individual: Dict[str, Any]) -> str:
        """
        Generate a unique signature for an individual based on its genome structure.
        """
        member = individual.get('member', individual)
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
       



    async def create_offsprings(self, species, num_offsprings: int,  problem_pool:List[Dict[str, str]], pbar: tqdm=None)-> List[Phenotype]:
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
                child_mutation_config: dict = self.mutator.get_dynamic_config(
                    generation=self.current_generation_index, 
                    parent_node_count=len(child.nodes)
                )
                
                #mutate offspring
                mutated_child: AgentGenome = await self.mutator.mutate(genome=child, runtime_config=child_mutation_config)
                child_phenotype = Phenotype(genome=mutated_child, llm_client=self.llm_client)
                await self.fitness_evaluator._update_fitness(problems_pool=problem_pool, phenotype=child_phenotype)
                new_species = True
                next_generation = self.current_generation_index + 1
                current_species = tuple(self.species)
                for s in current_species:
                    # only work with active species
                    if s.last_generation_index() >= self.current_generation_index:
                        if s.belongs_to_species(child_phenotype, self.current_generation_index) and s == species:
                            # child belongs to the same species as parents
                            healthy_child = True
                            new_species = False
                            offsprings.append(child_phenotype)
                            break
                        elif s.belongs_to_species(child_phenotype, self.current_generation_index) and s != species and s.generation_offset == next_generation:
                            # child belongs to newly created species along another individual
                            healthy_child = True
                            new_species = False
                            s.add_members([child_phenotype], generation=next_generation)
                            break
                        elif s.belongs_to_species(child_phenotype, self.current_generation_index) and s != species:
                            # child belongs to another existing species. Abort and retry
                            healthy_child = False
                            new_species = False
                            #break
                if new_species:
                    # child creates a new species
                    healthy_child = True
                    self.species.append(Species([child_phenotype], generation=next_generation, max_hof_size=self.per_species_hof_size))
            if pbar:
                pbar.update(1)
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
    