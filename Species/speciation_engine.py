"""
Speciation Engine Module.
Handles the macro-level ecology of the Reasoning Agent Engine (RAE).
Manages dynamic thresholding, Explicit Fitness Sharing, and resource allocation.
"""
from typing import List
from Genome.agent_genome import AgentGenome
from Species.species import Species
from Species.species_breeder import SpeciesBreeder

class SpeciationEngine:
    def __init__(self, 
                 breeder: SpeciesBreeder,
                 target_population_size: int = 40,
                 target_species_count: int = 4, # Single target for the P-Controller
                 dropoff_age: int = 10,
                 proportioanl_step: float = 0.075
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
        self.proportioanl_step = proportioanl_step

        self.species_list: List[Species] = []
        self.compatibility_threshold = 2.0 # Starting point, will be dynamically tuned
        self.species_id_counter = 0

    async def step_generation(self, evaluated_population: List[AgentGenome]) -> List[AgentGenome]:
        """
        Takes the current evaluated population, processes speciation, allocates resources,
        and returns the completely new, unevaluated next generation.
        """
        # --- 1. SPECIATION (Routing the Population) ---
        self._speciate_population(evaluated_population)

        # --- 2. THE THERMOSTAT (Dynamic Thresholding) ---
        self._adjust_compatibility_threshold()

        # --- 3. STAGNATION CULLING WITH EXTINCTION FAILSAFE ---
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

        # --- 4. EXPLICIT FITNESS SHARING (Resource Allocation) ---
        species_target_sizes = self._calculate_offspring_allocation()

        # --- 5. BREEDING (Calling the Micro-Layer) ---
        next_generation_global: List[AgentGenome] = []
        
        for species in self.species_list:
            target_size = species_target_sizes.get(species.id, 0)
            if target_size > 0:
                # Delegate to the micro-layer we built previously!
                offspring = await self.breeder.breed_next_generation(species.members, target_size)
                next_generation_global.extend(offspring)

        # --- 6. PREPARE FOR NEXT GENERATION ---
        # Pick new representatives and clear the buckets
        self.species_list = [s for s in self.species_list if species_target_sizes.get(s.id, 0) > 0]
        for s in self.species_list:
            s.update_representative()

        return next_generation_global

    def _speciate_population(self, population: List[AgentGenome]):
        """Routes each genome into the appropriate species bucket."""
        for genome in population:
            found_species = False
            for species in self.species_list:
                if species.compatibility_distance(genome) < self.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break
            
            # If it doesn't fit anywhere, create a new niche
            if not found_species:
                new_species = Species(representative=genome, species_id=self.species_id_counter)
                self.species_id_counter += 1
                self.species_list.append(new_species)
                
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
        
        # 2. Proportional gain (Kp). 
        # 0.075 means an error of 2 species results in a 0.15 threshold shift.
        proportional_step = self.proportioanl_step 
        
        # 3. Calculate adjustment
        adjustment = error * proportional_step
        
        # 4. Apply adjustment (with a hard floor to prevent the threshold from reaching 0 or negatives)
        self.compatibility_threshold = max(0.3, self.compatibility_threshold + adjustment)
        
        # Optional: Print for debugging so you can watch the P-Controller work
        # print(f"   [Thermostat] Species: {num_species} (Target: {self.target_species_count}) | Error: {error} | Adjustment: {adjustment:+.3f} | New Threshold: {self.compatibility_threshold:.3f}")

    def _calculate_offspring_allocation(self) -> dict:
        """Determines exactly how many of the 50 slots each species gets."""
        if not self.species_list:
            return {}

        # 1. Calculate the shared average fitness of each species
        avg_fitnesses = {s.id: s.get_average_fitness() for s in self.species_list}
        total_average = sum(avg_fitnesses.values())

        allocation = {}
        allocated_so_far = 0

        # Edge Case: Everyone scored 0. Distribute evenly.
        if total_average == 0:
            slots_per = self.target_population_size // len(self.species_list)
            for s in self.species_list:
                allocation[s.id] = slots_per
                allocated_so_far += slots_per
        else:
            # 2. Allocate proportionally
            for s in self.species_list:
                exact_allocation = (avg_fitnesses[s.id] / total_average) * self.target_population_size
                granted_slots = int(exact_allocation) # Floor it
                allocation[s.id] = granted_slots
                allocated_so_far += granted_slots

        # 3. Handle leftover slots due to rounding (give them to the best performing species)
        leftovers = self.target_population_size - allocated_so_far
        if leftovers > 0:
            # Sort species by average fitness, descending
            sorted_species_ids = sorted(avg_fitnesses, key=avg_fitnesses.get, reverse=True)
            for i in range(leftovers):
                lucky_species = sorted_species_ids[i % len(sorted_species_ids)]
                allocation[lucky_species] += 1

        return allocation
    