from tqdm.notebook import tqdm
from Crossover.crossover import Crossover
from Utils.utilities import md_logger
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
                 c3 = 0.4,  # Coefficient for different edges count in species compatibility
                 c4 = 0.2,  # Coefficient for average node differences in species compatibility
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

    def get_problem_pool(self, size: int = 3) -> List[Dict[str, str]]:
        """
        Returns a list of problems from the dataset manager.
        """
        return self.dataset_manager.get_batch(size=size)
    
    def _compute_normalized_species_counts(self, average_fitness: float, total_individuals: int) -> Dict[Species, int]:
        """
        Computes normalized offspring counts for each species based on average fitness.
        Ensures the total next generation size matches the target (total_individuals).
        """
        species_counts_raw = {}
        
        # 1. Calculate Raw Target Counts
        for species in self.species:
            # Initialize to 0 safety
            species_counts_raw[species] = 0
            
            if species.last_generation_index() == self.current_generation_index:
                count = species.adjusted_offspring_count(average_fitness, self.current_generation_index)
                species_counts_raw[species] = count

        # 2. Normalize to fill Total Population
        normalized_species_counts = {}
        total_raw_counts = sum(species_counts_raw.values())
        
        # Debug: Print raw distribution
        # print(f"      Raw Allocations: {raw_debug} (Sum: {total_raw_counts})")

        current_allocated_sum = 0

        for species, count in species_counts_raw.items():
            if total_raw_counts > 0:
                # Proportional allocation
                norm_count = int((count / total_raw_counts) * total_individuals)
            else:
                # If everyone failed (total_raw=0), split evenly? or kill all?
                # Fallback: Even split among active species
                active_count = len([s for s in species_counts_raw if species_counts_raw[s] >= 0]) # approximate
                norm_count = total_individuals // active_count if active_count > 0 else 0
            
            normalized_species_counts[species] = norm_count
            current_allocated_sum += norm_count

        # 3. Handle Rounding Errors (Remainders)
        # Example: If we have 50 slots but calculated sum is 48, we need to add 2.
        remainder = total_individuals - current_allocated_sum
        
        if remainder > 0:
            print(f"      ⚠️ Rounding Gap: {remainder} unallocated slots. Distributing to top performers...")
            # Simple strategy: Give 1 extra slot to the first N species in the list until remainder is gone
            # (Ideally, sort by fitness first, but order is often implicit)
            active_species = [s for s in normalized_species_counts if normalized_species_counts[s] > 0]
            if not active_species:
                active_species = list(normalized_species_counts.keys())
            
            for i in range(remainder):
                target_species = active_species[i % len(active_species)]
                normalized_species_counts[target_species] += 1
        
        return normalized_species_counts
        
    async def create_new_generation(self) -> list[Tuple[int, Phenotype]]:
        """
        Creates a new generation of individuals based on the fitness of the current generation.
        Uses the injected survivor_strategy to select the next generation for each species.
        Includes robust error handling and debug logging.
        """
        print(f"\n--- 🧬 Evolution Manager: Starting Generation {self.current_generation_index + 1} ---")
        
        # 1. Update Hall of Fame
        # ---------------------------------------------------------
        print("   🏆 Updating Hall of Fame...")
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                species.update_hall_of_fame()

        # 2. Generate Problem Pool
        # ---------------------------------------------------------
        print("   🧪 Generating new Problem Pool...")
        
        # We use the same pool for all species to ensure fair comparison
        problem_pool: List[Dict[str, str]] = self.get_problem_pool(size=3)
        print(f"      -> Pool size: {len(problem_pool)}")
        # 3. Calculate Fitness Stats & Allocate Slots (Speciation)
        # ---------------------------------------------------------
        print("   📊 Calculating Fitness Statistics & Allocating Slots...")
        
        total_fitness = 0.0
        total_individuals = 0
        print(f"Starting from index {self.current_generation_index}, calculating total fitness across {len(self.species)} species")
        
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                total_fitness += species.cumulative_fitness(self.current_generation_index)
                total_individuals += species.member_count(self.current_generation_index)
        
        average_fitness = total_fitness / total_individuals if total_individuals > 0 else 0
        print(f"      -> Global Avg Fitness: {average_fitness:.4f}| Total Fitness: {total_fitness} | Total Pop: {total_individuals}")

        # This determines how many babies each species is allowed to have
        species_counts = self._compute_normalized_species_counts(average_fitness, total_individuals)

        # 4. Process Each Species
        # ---------------------------------------------------------
        active_species_count = 0
        pbar = tqdm(total=50, desc=f"Gen {self.current_generation_index+1} Breeding", unit="child")
        for species in self.species:
            # Skip species that died out previously
            if species.last_generation_index() != self.current_generation_index:
                continue
            
            # Retrieve allocated slot count
            new_species_target_count = species_counts.get(species, 0)
            
            # If the species is extinct due to poor performance
            if new_species_target_count <= 0:
                print(f"   💀 Species {species.id} has gone extinct (0 slots allocated).")
                continue

            print(f"\n   🦕 Processing Species {species.id} (Target: {new_species_target_count})...")
            active_species_count += 1

            pbar.set_description(f"Breeding Species {species.id}")

            # A. Create Offspring
            # ---------------------------------------
            # We usually generate exactly the target count, or slightly more if we want selection pressure
            print(f"      Creating {new_species_target_count} offspring...")
            offsprings = await self.create_offsprings(species, new_species_target_count, problem_pool=problem_pool, pbar=pbar)
            
            if not offsprings:
                print(f"      ⚠️ Warning: No offspring produced for species {species.id}. Skipping.")
                continue

            # B. Gather Candidates (Parents + Children)
            # ---------------------------------------
            current_members = species.get_all_members_from_generation(self.current_generation_index)
            current_phenotypes = [item['member'] for item in current_members]
            
            all_candidates = current_phenotypes + offsprings
            print(f"      Candidates: {len(current_phenotypes)} Parents + {len(offsprings)} Offspring = {len(all_candidates)} Total")

            # C. Evaluate EVERYONE on the NEW Problem Pool
            # ---------------------------------------
            # This is crucial: Parents might have high fitness from an "easy" previous batch.
            # Re-evaluating them on the new batch ensures fair competition with children.
            print("      Evaluating all candidates on new problem pool...")
            await self.fitness_evaluator.evaluate_population(population=all_candidates, problem_pool=problem_pool)

            # D. Prepare Data for Selection Strategy
            # ---------------------------------------
            offspring_dicts = [{"member": child, "fitness": child.genome.fitness} for child in offsprings]
            
            # Parent dicts need updated fitness from the re-evaluation above
            current_pop_dicts = [{"member": p, "fitness": p.genome.fitness} for p in current_phenotypes]

            # E. Select Survivors
            # ---------------------------------------
            # For Comma Strategy (Children replace parents): target_size is just the allocated count
            # For Plus Strategy (Best of both): target_size is also the allocated count
            print("      Selecting survivors...")
            selected_individuals_dicts = self.survivor_strategy.select_survivors(
                current_population=current_pop_dicts,
                offspring_population=offspring_dicts,
                population_size=new_species_target_count
            )

            selected_individuals = [item['member'] for item in selected_individuals_dicts]
            print(f"      ✅ Selected {len(selected_individuals)} survivors for next gen.")

            # F. Add to Next Generation Storage
            # ---------------------------------------
            species.add_members(selected_individuals, generation=self.current_generation_index + 1)

        pbar.close()

        # 5. Finalize Generation
        # ---------------------------------------------------------
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
       



    async def create_offsprings(self, species, num_offsprings: int, problem_pool:List[Dict[str, str]], pbar: tqdm=None)-> List[Phenotype]:
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
                child: AgentGenome = await self.mutator.mutate(genome=child, runtime_config=child_mutation_config)
                child_phenotype = Phenotype(genome=child, llm_client=self.llm_client)
                await self.fitness_evaluator._update_fitness(problems_pool=problem_pool, phenotype=child_phenotype)
                new_species = True
                next_generation = self.current_generation_index + 1
                for s in self.species:
                    if s.last_generation_index() >= self.current_generation_index:
                        if s.belongs_to_species(child_phenotype, self.current_generation_index) and s != species and s.generation_offset == next_generation:
                            # child belongs to newly created species along another individual
                            healthy_child = True
                            new_species = False
                            s.add_members([child_phenotype], generation=next_generation)
                            break
                        if s.belongs_to_species(child_phenotype, self.current_generation_index) and s != species:
                            # child belongs to another existing species. Abort and retry
                            healthy_child = False
                            new_species = False
                            break
                        elif s.belongs_to_species(child_phenotype, self.current_generation_index) and s == species:
                            # child belongs to the same species as parents
                            healthy_child = True
                            new_species = False
                            offsprings.append(child_phenotype)
                            break
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
    