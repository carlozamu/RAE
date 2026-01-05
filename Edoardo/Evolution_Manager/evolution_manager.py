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
                 compatibility_threshold = 2.51  # Threshold for species compatibility
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
        
    # Inside evolution_manager.py

    async def create_new_generation(self) -> list[Tuple[int, Phenotype]]:
        print(f"\n--- 🧬 Evolution Manager: Starting Generation {self.current_generation_index + 1} ---")
        
        # 1. Update HoF and clean empty species
        self.species = [s for s in self.species if s.member_count(self.current_generation_index) > 0]
        for species in self.species:
            species.update_hall_of_fame()

        problem_pool = self.get_problem_pool()

        # 2. Safe Fitness Calculation (Handling inf)
        total_adjusted_fitness = 0.0
        valid_species = []
        
        # We invert fitness here because you want to minimize Loss.
        # Higher 'inverse_fitness' = More offspring
        for s in self.species:
            avg_loss = s.cumulative_fitness(self.current_generation_index) / max(1, s.member_count(self.current_generation_index))
            
            # Clamp infinite loss to a large number (e.g., 100.0) to prevent math errors
            if avg_loss == float('inf') or avg_loss > 1000:
                avg_loss = 1000.0
                
            # Fitness formula: 1 / avg_loss
            # (Low loss = High fitness score for allocation)
            s_fitness_score = 1.0 / max(1e-9, avg_loss) 
            
            # Store for allocation calculation
            s.temp_fitness_score = s_fitness_score
            total_adjusted_fitness += s_fitness_score
            valid_species.append(s)

        # 3. Allocate Slots (Exact Count)
        target_pop_size = 30
        allocated_slots = {}
        total_allocated = 0
        
        # First pass: Integers
        for s in valid_species:
            share = (s.temp_fitness_score / total_adjusted_fitness) * target_pop_size
            slots = max(1, int(round(share))) # Ensure at least 1 if it survived this far
            allocated_slots[s.id] = slots
            total_allocated += slots

        # Second pass: Force sum to exactly 30
        # If we have too many, remove from the worst performing (highest loss)
        sorted_species = sorted(valid_species, key=lambda x: x.temp_fitness_score) # Ascending fitness (Worst first)
        
        while total_allocated > target_pop_size:
            for s in sorted_species:
                if allocated_slots[s.id] > 1: # Don't kill species completely yet
                    allocated_slots[s.id] -= 1
                    total_allocated -= 1
                    if total_allocated == target_pop_size: break
            # If still over, force remove from random
            if total_allocated > target_pop_size:
                rid = random.choice([s.id for s in valid_species if allocated_slots[s.id] > 0])
                allocated_slots[rid] -= 1
                total_allocated -= 1

        # If we have too few, add to best performing
        while total_allocated < target_pop_size:
            best_s = sorted_species[-1] # Best is last
            allocated_slots[best_s.id] += 1
            total_allocated += 1

        print(f"   📊 Allocation complete. Total Slots: {total_allocated}")

        # 4. Breeding & Selection Loop
        next_gen_population: List[Phenotype] = []
        
        pbar = tqdm(total=target_pop_size, desc="Breeding & Selecting")
        
        for species in valid_species:
            slots = allocated_slots.get(species.id, 0)
            if slots == 0: continue
                
            # A. Breed
            # Breed slightly more than allocated slots (e.g., 1.5x) to ensure selection pressure
            # so the survivor strategy has candidates to discard.
            num_children = slots
            offspring = await self.create_offsprings(species, num_children, problem_pool, pbar)
            
            # B. Get Current Members
            current_members = [x['member'] for x in species.get_all_members_from_generation(self.current_generation_index)]
            
            # C. Re-evaluate Parents on new pool (Fairness)
            # We must re-eval parents so they share the same fitness baseline as children
            await self.fitness_evaluator.evaluate_population(current_members, problem_pool)
            
            # D. Survivor Selection
            # We pass the split populations so the strategy can decide if parents are eligible (Plus) or not (Comma)
            survivors_dicts = self.survivor_strategy.select_survivors(
                current_population=[{"member": m, "fitness": m.genome.fitness} for m in current_members],
                offspring_population=[{"member": m, "fitness": m.genome.fitness} for m in offspring],
                population_size=slots # STRICTLY enforce this size
            )
            
            survivors = [x['member'] for x in survivors_dicts]
            next_gen_population.extend(survivors)
            
        pbar.close()
        
        # 5. Global Speciation (The "Clean Up")
        print(f"   🧩 Speciating {len(next_gen_population)} survivors...")
        
        self.current_generation_index += 1
        next_gen_idx = self.current_generation_index
        
        # [FIX 1] Iterate over a COPY of the species list ([:])
        # This ensures that if we append new species during the loop, 
        # we don't iterate over them immediately or break the iterator.
        
        # We also need to handle new species created *during* this loop.
        # If an individual doesn't fit existing species, it creates a new one.
        # That new one should be available for SUBSEQUENT individuals in this same batch.
        
        active_species_pool = self.species[:] # Snapshot of "Old" species

        for individual in next_gen_population:
            placed = False
            
            # Check against the Snapshot (Established Species)
            for s in active_species_pool:
                if s.belongs_to_species(individual): # No generation needed, uses stored Rep
                    s.add_members([individual], generation=next_gen_idx)
                    placed = True
                    break
            
            # If not placed in old species, check against NEWLY created species
            # (This groups orphans together instead of making 1 species per orphan)
            if not placed:
                # Check against species added to self.species during this loop
                # We iterate only the new ones
                newly_added = self.species[len(active_species_pool):]
                for s in newly_added:
                    if s.belongs_to_species(individual):
                        s.add_members([individual], generation=next_gen_idx)
                        placed = True
                        break

            if not placed:
                # Create completely new species
                # The first member becomes the implicit representative via __init__
                new_s = Species([individual], generation=next_gen_idx, max_hof_size=self.per_species_hof_size)
                self.species.append(new_s)
                
        # Remove species that received no members in this new generation
        self.species = [s for s in self.species if s.member_count(next_gen_idx) > 0]
        
        # --- ADD THIS BLOCK ---
        for s in self.species:
            s.elect_new_representative()
        # ----------------------

        print(f"   ✅ Generation {next_gen_idx} Complete. Active Species: {len(self.species)}")
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

    async def create_offsprings(self, species, num_offsprings: int, problem_pool: List[Dict[str, str]], pbar: tqdm = None) -> List[Phenotype]:
        """
        Creates num_offsprings from a species. 
        Does NOT assign them to species yet. strictly breeding.
        """
        offsprings = []
        # Safety break to prevent infinite loops if parents are incompatible
        attempts = 0
        max_attempts = num_offsprings * 5 

        while len(offsprings) < num_offsprings and attempts < max_attempts:
            attempts += 1
            
            # 1. Select Parents & Crossover
            parents = self.select_parents(species, self.num_parents)
            if len(parents) < 2:
                # Fallback if species is too small
                child_genome = parents[0].genome.copy() # Clone
            else:
                child_genome = Crossover.create_offspring(parents[0].genome, parents[1].genome)

            # 2. Mutation
            child_mutation_config = self.mutator.get_dynamic_config(
                generation=self.current_generation_index, 
                parent_node_count=len(child_genome.nodes)
            )
            mutated_genome = await self.mutator.mutate(genome=child_genome, runtime_config=child_mutation_config)
            
            # 3. Create Phenotype & Evaluate
            child_phenotype = Phenotype(genome=mutated_genome, llm_client=self.llm_client)
            
            # Evaluate immediately
            await self.fitness_evaluator._update_fitness(problems_pool=problem_pool, phenotype=child_phenotype)
            
            offsprings.append(child_phenotype)
            
            if pbar:
                pbar.update(1)
                
        return offsprings

    def select_parents(self, species: Species, num_parents: int) -> List[Phenotype]:
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
        
        selected_parents: List[Phenotype] = []
        
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
    