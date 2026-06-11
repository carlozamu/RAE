"""
Speciation Engine Module.
Handles the macro-level ecology of the Reasoning Agent Engine (RAE).
Manages dynamic thresholding, Explicit Fitness Sharing, and resource allocation.
"""
from typing import List
from Genome.agent_genome import AgentGenome
from Species.species import Species
from Species.species_breeder import SpeciesBreeder

# --- UPDATED CONSTANTS FOR JACCARD SIMILARITY SCALE (0.0 to 1.0) ---
MAX_SHIFT = 0.1           # Max 10% shift per generation to prevent oscillation
INITIAL_THRESHOLD = 0.32   # 0.45 Distance = requires 46% similarity to join
MIN_THRESHOLD = 0.15       # Hard floor to prevent infinite speciation
MAX_THRESHOLD = 0.85       # Hard ceiling

class SpeciationEngine:
    def __init__(self, 
                 breeder: SpeciesBreeder,
                 target_population_size: int = 50,
                 target_species_count: int = 5, 
                 dropoff_age: int = 15,
                 proportional_step: float = 0.035 # Lowered step size for 0-1 scale stability
                 ):
        self.breeder = breeder
        self.target_population_size = target_population_size
        self.target_species_count = target_species_count
        self.dropoff_age = dropoff_age
        self.proportional_step = proportional_step

        self.species_list: List[Species] = []
        self.compatibility_threshold = INITIAL_THRESHOLD 
        
        # Start at 1, because 0 is strictly reserved for the Primordial Soup
        self.species_id_counter = 1 

    async def step_generation(self, generation: int) -> List[AgentGenome]:
        """
        Takes the current evaluated population, processes speciation, allocates resources,
        and returns the completely new, unevaluated next generation.
        """
        # --- 1. THE THERMOSTAT (Dynamic Thresholding) ---
        # Assuming you updated this to pass generation or removed the arg as discussed
        self._adjust_compatibility_threshold(generation=generation)

        # --- 2. STAGNATION CULLING WITH EXTINCTION FAILSAFE ---
        mature_active = []
        stagnant_this_turn = []
        
        for s in self.species_list:
            if not s.alive:
                continue # Skip species that are already dead
                
            s.update_stagnation()
            if s.generations_without_improvement >= self.dropoff_age and s.id != 0: # Never kill the Primordial Soup
                s.alive = False # Kill instead of removing
                stagnant_this_turn.append(s)
            else:
                mature_active.append(s)

        # THE NEAT FAILSAFE 
        # Ensure we don't try to mandate 2 species if the engine hasn't even created 2 yet
        total_mature_ever_created = len([s for s in self.species_list if s.id != 0])
        if len(mature_active) < min(2, total_mature_ever_created):
            stagnant_this_turn.sort(key=lambda x: x.max_fitness_ever, reverse=True)
            needed = min(2, total_mature_ever_created) - len(mature_active)
            spared_species = stagnant_this_turn[:needed]
            
            for s in spared_species:
                s.alive = True # Resurrect the spared species
                s.generations_without_improvement = 0  
                print(f"⚠️ Failsafe Triggered: Spared Species {s.id} from mass extinction.")

        # --- 3. GLOBAL ELITISM ---
        # Only pull members from ALIVE species
        all_global_members = [member for s in self.species_list if s.alive for member in s.members]
        all_global_members.sort(key=lambda x: (x.accuracy, x.fitness), reverse=True)
        
        global_elites = all_global_members[:3]
        num_elites = len(global_elites)

        # --- 4. EXPLICIT FITNESS SHARING ---
        species_target_sizes = self._calculate_offspring_allocation(num_elites)

        # --- 5. BREEDING ---
        next_generation_global: List[AgentGenome] = []

        for elite in global_elites:
            next_generation_global.append(elite.copy())
        
        for species in self.species_list:
            if not species.alive:
                continue # Do not breed dead species
                
            target_size = species_target_sizes.get(species.id, 0)
            if target_size > 0:
                offspring = await self.breeder.breed_next_generation(species.members, target_size, generation=generation)
                next_generation_global.extend(offspring)

        # --- 6. PREPARE FOR NEXT GENERATION ---
        # Instead of deleting species that got 0 slots, we kill them
        for s in self.species_list:
            if s.alive:
                target_size = species_target_sizes.get(s.id, 0)
                if target_size == 0 and s.id != 0:
                    s.alive = False # Died out due to lack of resources
                else:
                    s.update_representative()

        return next_generation_global

    def _speciate_population(self, population: List[AgentGenome]):
        # 1. We look at ALL species (even dead ones) to find the best similarity
        # We categorize the soup separately
        primordial_soup = next((s for s in self.species_list if s.id == 0), None)

        for genome in population:
            best_species = None
            best_similarity = self.compatibility_threshold # Only accept if > threshold

            # --- SEARCH PHASE ---
            # Compare against ALL species (alive or dead)
            for species in self.species_list:
                if species.id == 0: continue # Never compare against the Soup
                
                similarity = species.compatibility_distance(genome)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_species = species
            
            # --- ROUTING PHASE ---
            if best_species:
                # We found a match!
                best_species.add_member(genome)

                # Logic: If it was dead, resurrect it and update representative
                if not best_species.alive:
                    best_species.alive = True
                    best_species.representative = genome
                    #print(f"Species {best_species.id} resurrected by individual.")
            else:
                # No match found above threshold
                genome_size = len(genome.nodes) + sum(1 for c in genome.connections.values() if c.enabled)
                
                if genome_size <= 1:
                    # It's a 1-node graph -> Dump into Primordial Soup
                    if primordial_soup is None:
                        primordial_soup = Species(representative=genome, species_id=0)
                        self.species_list.append(primordial_soup)
                    else:
                        primordial_soup.add_member(genome)
                        primordial_soup.alive = True # Soup wakes up
                else:
                    # Complex graph -> New species
                    new_species = Species(representative=genome, species_id=self.species_id_counter)
                    self.species_id_counter += 1
                    self.species_list.append(new_species)

        # --- CLEANUP PHASE ---
        # Mark as dead if they have no members
        for s in self.species_list:
            if len(s.members) == 0:
                s.alive = False

    def _adjust_compatibility_threshold(self, generation: int = 2):
        if generation > 4:
            mature_species_count = len([s for s in self.species_list  if len(s.members) > 0 and s.alive])
            
            error = mature_species_count - self.target_species_count
            adjustment = self.proportional_step * error
            shift = max(-MAX_SHIFT, min(MAX_SHIFT, adjustment))
            
            # Apply adjustment with hard mathematical floors and ceilings
            new_threshold = self.compatibility_threshold - shift
            self.compatibility_threshold = max(MIN_THRESHOLD, min(MAX_THRESHOLD, new_threshold))

    def _calculate_offspring_allocation(self, num_elites: int) -> dict:
        """Determines exactly how many of the remaining slots each species gets."""
        if not self.species_list:
            return {}

        # 1. Filter out dead species
        active_species = [s for s in self.species_list if s.alive]
        if not active_species:
            return {}

        avg_fitnesses = {s.id: s.get_average_fitness() for s in active_species}
        total_average = sum(avg_fitnesses.values())

        allocation = {s.id: 0 for s in active_species}
        
        # The total number of slots available for actual breeding
        offspring_target = max(0, self.target_population_size - num_elites)
        
        if offspring_target == 0:
            return allocation

        # --- EDGE CASE: Everyone scored 0 ---
        # Distribute evenly, but prioritize the youngest species first to protect them.
        if total_average == 0:
            sorted_species = sorted(active_species, key=lambda s: s.age) 
            allocated = 0
            while allocated < offspring_target:
                for s in sorted_species:
                    if allocated >= offspring_target:
                        break
                    allocation[s.id] += 1
                    allocated += 1
            return allocation

        sorted_species = sorted(active_species, key=lambda s: avg_fitnesses[s.id])
        
        remaining_target = offspring_target
        remaining_fitness = total_average
        
        for s in sorted_species:
            if remaining_target <= 0:
                break # Hard stop if we run out of physical slots
                
            # Calculate proportional share from the REMAINING pool
            if remaining_fitness > 0:
                exact_allocation = (avg_fitnesses[s.id] / remaining_fitness) * remaining_target
                granted = int(exact_allocation) # Floor it
            else:
                granted = 0
                
            granted = min(granted, remaining_target)
            
            allocation[s.id] = granted
            
            # Deduct from the pools so the next species gets a mathematically adjusted slice
            remaining_target -= granted
            remaining_fitness -= avg_fitnesses[s.id]

        # Failsafe for rounding remainders (give to the most performant, which is the last in the list)
        if remaining_target > 0:
            best_species_id = sorted_species[-1].id
            allocation[best_species_id] += remaining_target

        return allocation
    