"""
Speciation Engine Module.
Handles the macro-level ecology of the Reasoning Agent Engine (RAE).
Manages dynamic thresholding, Explicit Fitness Sharing, and resource allocation.
"""
from typing import List
from Genome.agent_genome import AgentGenome
from Species.species import Species
from Species.species_breeder import SpeciesBreeder

MAX_SHIFT = 100.0 # Maximum allowed change to the compatibility threshold per generation to prevent oscillation
INITIAL_THRESHOLD = 5.0

class SpeciationEngine:
    def __init__(self, 
                 breeder: SpeciesBreeder,
                 target_population_size: int = 50,
                 target_species_count: int = 4, # Single target for the P-Controller
                 dropoff_age: int = 10,
                 proportional_step: float = 0.5
                 ):
        """
        Args:
            breeder: The micro-layer instance that handles intra-species mating.
            target_population_size: Total allowed agents across all species.
            target_species_count: The ideal number of species to maintain.
            dropoff_age: Generations a species can survive without improving max fitness.
        """
        self.breeder = breeder
        self.target_population_size = target_population_size
        self.target_species_count = target_species_count
        self.dropoff_age = dropoff_age
        self.proportional_step = proportional_step

        self.species_list: List[Species] = []
        self.compatibility_threshold = INITIAL_THRESHOLD # Starting point, will be dynamically tuned
        self.species_id_counter = 0

    async def step_generation(self, generation: int) -> List[AgentGenome]:
        """
        Takes the current evaluated population, processes speciation, allocates resources,
        and returns the completely new, unevaluated next generation.
        """
        # --- 1. THE THERMOSTAT (Dynamic Thresholding) ---
        self._adjust_compatibility_threshold()

        # --- 2. STAGNATION CULLING WITH EXTINCTION FAILSAFE ---
        active_species = []
        stagnant_species = []
        
        for s in self.species_list:
            s.update_stagnation()
            if s.generations_without_improvement >= self.dropoff_age:
                stagnant_species.append(s)
            else:
                active_species.append(s)

        # THE NEAT FAILSAFE: Never drop below 2 species due to culling
        if len(active_species) < min(2, len(self.species_list)):
            # Sort the dying species by their historical best performance
            stagnant_species.sort(key=lambda x: x.max_fitness_ever, reverse=True)
            
            # Calculate how many we need to spare to maintain exactly 2 active species
            needed = 2 - len(active_species)
            spared_species = stagnant_species[:needed]
            
            for s in spared_species:
                s.generations_without_improvement = 0  # Grant a grace period
                active_species.append(s)
                print(f"⚠️ Failsafe Triggered: Spared Species {s.id} from mass extinction.")

        self.species_list = active_species

        # --- 3. GLOBAL ELITISM ---
        # Pool every single member across all species into a flat list
        all_global_members = [member for s in self.species_list for member in s.members]
        
        # Sort by fitness descending to find the absolute best
        all_global_members.sort(key=lambda x: x.fitness, reverse=True)
        
        # Slice the top 3 (or fewer, if the population is extremely small)
        global_elites = all_global_members[:3]
        num_elites = len(global_elites)

        # --- 4. EXPLICIT FITNESS SHARING (Resource Allocation) ---
        # Pass the number of elites so the allocator knows how many slots are left
        species_target_sizes = self._calculate_offspring_allocation(num_elites)

        # --- 5. BREEDING (Calling the Micro-Layer) ---
        next_generation_global: List[AgentGenome] = []

        # Immediately clone the top 3 global elites into the next generation untouched
        for elite in global_elites:
            next_generation_global.append(elite.copy())
        
        for species in self.species_list:
            target_size = species_target_sizes.get(species.id, 0)
            if target_size > 0:
                # Delegate to the micro-layer
                offspring = await self.breeder.breed_next_generation(species.members, target_size, generation=generation)
                next_generation_global.extend(offspring)

        # --- 6. PREPARE FOR NEXT GENERATION ---
        # Pick new representatives and clear the buckets
        self.species_list = [s for s in self.species_list if species_target_sizes.get(s.id, 0) > 0]
        for s in self.species_list:
            s.update_representative()

        return next_generation_global

    def _speciate_population(self, population: List[AgentGenome]):
        """Routes each genome into the appropriate species bucket."""

        new_species_list: list[Species] = []

        for genome in population:
            distances = []
            found_species = False
            for species in self.species_list:
                distance = species.compatibility_distance(genome)
                distances.append(f"{distance:.2f}")
                if distance < self.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break
            
            for species in new_species_list:
                distance = species.compatibility_distance(genome)
                distances.append(f"{distance:.2f}")
                if distance < self.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break
            
            # If it doesn't fit anywhere, create a new niche
            if not found_species:
                print(f"New Species Detected: {distances} | Threshold: {self.compatibility_threshold:.2f}")
                new_species = Species(representative=genome, species_id=self.species_id_counter)
                self.species_id_counter += 1
                new_species_list.append(new_species)

        self.species_list.extend(new_species_list)

        # Sort species by fitness
        self.species_list.sort(key=lambda x: x.get_average_fitness(), reverse=True)
                
        # Remove empty species (can happen if a representative's niche dies out)
        self.species_list = [s for s in self.species_list if len(s.members) > 0]

    def _adjust_compatibility_threshold(self):
        """
        Proportional Thermostat: Smoothly tunes the threshold to maintain the target species count.
        Avoids oscillation by scaling the adjustment to the size of the error.
        """
        num_species = len(self.species_list)
        
        # 1. Calculate how far we are from the ideal target (e.g., 4)
        error = num_species - self.target_species_count
    
        # 3. Calculate adjustment
        adjustment = 0
        adjustment = self.proportional_step * error
        shift = max(-MAX_SHIFT, min(MAX_SHIFT, adjustment))
        
        # 4. Apply adjustment (with a hard floor to prevent the threshold from reaching 0 or negatives)
        self.compatibility_threshold = self.compatibility_threshold + shift
        
        # Optional: Print for debugging so you can watch the P-Controller work
        # print(f"   [Thermostat] Species: {num_species} (Target: {self.target_species_count}) | Error: {error} | Adjustment: {adjustment:+.3f} | New Threshold: {self.compatibility_threshold:.3f}")

    def _calculate_offspring_allocation(self, num_elites: int) -> dict:
        """Determines exactly how many of the remaining slots each species gets."""
        if not self.species_list:
            return {}

        avg_fitnesses = {s.id: s.get_average_fitness() for s in self.species_list}
        total_average = sum(avg_fitnesses.values())

        allocation = {}
        allocated_so_far = 0
        
        # The total number of slots available for actual breeding
        offspring_target = max(0, self.target_population_size - num_elites)

        # Edge Case: Everyone scored 0. Distribute evenly.
        if total_average == 0:
            slots_per = offspring_target // len(self.species_list) if self.species_list else 0
            for s in self.species_list:
                allocation[s.id] = slots_per
                allocated_so_far += slots_per
        else:
            # 2. Allocate proportionally
            for s in self.species_list:
                exact_allocation = (avg_fitnesses[s.id] / total_average) * offspring_target
                granted_slots = int(exact_allocation) # Floor it
                allocation[s.id] = granted_slots
                allocated_so_far += granted_slots

        # 3. Handle leftover slots due to rounding (give them to the best performing species)
        leftovers = offspring_target - allocated_so_far
        if leftovers > 0:
            # Sort species by average fitness, descending
            sorted_species_ids = sorted(avg_fitnesses, key=avg_fitnesses.get, reverse=True)
            for i in range(leftovers):
                lucky_species = sorted_species_ids[i % len(sorted_species_ids)]
                allocation[lucky_species] += 1

        return allocation
     