"""
1) Population initialization (simple same prompt, single node, 50 individuals, fitness computed on one individual and assigned to all)

2) Loop until termination condition (fitness under threshold (@DEV) or max time (X hours/ k-pressed) reached):

2) Loop until termination condition (fitness under threshold (@DEV) or max time (X hours/ k-pressed) reached):
    2.1) Loop for each species:
            - Selection (intra-species)
            - Crossover
            - Mutation
            - Fitness evaluation
    2.2) Average fitness calculation per species
    2.3) Replacement (keep 50 individuals top for each generation)

Variabili globali:
    next innovation number (incremental integer)
    generation index (incremental integer)
    species age (incremental integer per species)


Cose da loggare:
    generation number
    best-worse fitness per generation
    average fitness per species per generation
    average number of nodes and connections per individual per generation
    species count per generation
    paragone con prompt engineering

Cose da fare:
evolution manager todo
    mutazione genome max(0.1,0.9^(generation_number+1))
    mutazione gene 1/NUMERO DI NODI DEL GENOMA
    
    selection fix 

    genotype/phenotype
    new species handling


Report
    1) Abstract + Introduction (probem + why NEAT)
    2) Dataset + benchmarking
    2) Selection + Fitnes
    3) Crossover + Mutation
    3) Speciation + Evolution Management
    4) Experiments + Results
"""
from Edoardo.Fitness.fitness import Fitness
from Edoardo.Mutations.mutator import Mutator
from Edoardo.Data.cluttr import CLUTTRManager
from Edoardo.Data.cot import CoTManager
from Utils.utilities import _get_next_innovation_number
from Utils.LLM import LLM
from ERA.init_pop import initialize_population
from Edoardo.Evolution_Manager.evolution_manager import EvolutionManager
from Edoardo.Selection.selection import TournamentSelection
from Edoardo.Generation_Manager.generation_manager import CommaPlusStrategy

# Initialize LLM client endpoint, exposes get_embeddings and generate_text methods
llm_client = LLM()
print("LLM client initialized.")

USE_REASONING = True # Toggle to enable/disable reasoning evaluation
fitness_evaluator = Fitness(llm=llm_client, use_reasoning=USE_REASONING)
mutator = Mutator(breeder_llm_client=llm_client)
# Choose Dataset based on Reasoning flag
if USE_REASONING:
    print("Reasoning Evaluation ENABLED. Using CoT-Collection.")
    dataset_manager = CoTManager(split="train")
else:
    print("Reasoning Evaluation DISABLED. Using CLUTTR Dataset.")
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")

print("Fitness, Mutator & Dataset Manager initialized.")

# Initialize population
starting_prompt = "You are an expert reasoning AI. Given the input, provide a detailed and accurate response following the instructions."
population = initialize_population(num_individuals=50, prompt=starting_prompt)
_get_next_innovation_number()  # Initialize global innovation number tracker, now it becomes 0, next time the function will be called it will return 1
print(f"Initialized population with {len(population)} individuals with fitness of {population[0].fitness}.")

# Strategy Configuration
selection_strategy = TournamentSelection(tournament_size=3)    # <----- SET as HYPERPARAMETER?
survivor_strategy = CommaPlusStrategy(elite_size=2) # Elitism   # <----- SET as HYPERPARAMETER?

# Initialize Evolution Manager
evolution_manager = EvolutionManager(
    selection_strategy=selection_strategy,
    survivor_strategy=survivor_strategy,
    mutator=mutator,
    fitness_evaluator=fitness_evaluator,
    dataset_manager=dataset_manager,
    num_parents=2,
    per_species_hof_size=5,
    hof_parent_ratio=0.2
)

# TODO: Add the actual loop calling evolution_manager.create_new_generation()
print("Evolution Manager initialized.")


# Main Loop ------------ TODO: DA SISTEMARE
import asyncio
import time
from Edoardo.Species.species import Species

# Configuration for Stop Criteria
MAX_GENERATIONS = 50
MAX_TIME_SECONDS = 3600 * 10 # 10 Hours
TARGET_LOSS = 0.20 # Based on threshold analysis (Perfect ~0.15)

async def run_evolution():
    print("\n--- Starting Evolution ---")
    start_time = time.time()
    
    # 1. Bootstrap EvolutionManager with Initial Population
    # Create the first species from the initialized population
    initial_species = Species(
        members=population, 
        generation=0, 
        selection_strategy=selection_strategy,
        max_hof_size=5
    )
    evolution_manager.species.append(initial_species)
    
    # 2. Evaluate Initial Population
    # We need baseline fitness before the first evolution step
    print("Evaluating initial population...")
    initial_problems = evolution_manager.get_problem_pool(size=3)
    fitness_evaluator.evaluate_population(population, initial_problems)
    
    # Log Initial Stats
    best_init = min(p.fitness for p in population)
    avg_init = sum(p.fitness for p in population) / len(population)
    print(f"Gen 0 (Initial): Best Loss={best_init:.4f}, Avg Loss={avg_init:.4f}")

    # 3. Evolution Loop
    while True:
        current_gen = evolution_manager.current_generation_index
        
        # Check Stop Conditions
        # Note: We check fitness achieved in the PREVIOUS step (or initial step)
        # We need to get stats from the manager or species
        
        # Get global best fitness across all species
        best_loss = 1.0
        for s in evolution_manager.species:
            if s.last_generation_index() == current_gen:
                mems = s.get_all_members_from_generation(current_gen)
                if mems:
                    # mems is list of dict {'member': ..., 'fitness': ...}
                    s_best = min(m['fitness'] for m in mems)
                    if s_best < best_loss:
                        best_loss = s_best
        
        # Logging
        # print(f"Gen {current_gen} Completed. Global Best Loss: {best_loss:.4f}")
        
        # Stop Criteria
        if best_loss <= TARGET_LOSS:
            print(f"\nSUCCESS: Target Loss ({TARGET_LOSS}) reached! Best: {best_loss:.4f}")
            break
            
        if (time.time() - start_time) > MAX_TIME_SECONDS:
            print(f"\nSTOP: Maximum time limit ({MAX_TIME_SECONDS}s) reached.")
            break
            
        if current_gen >= MAX_GENERATIONS:
            print(f"\nSTOP: Maximum generations ({MAX_GENERATIONS}) reached.")
            break
            
        # Run Next Generation
        print(f"Running Generation {current_gen + 1}...")
        await evolution_manager.create_new_generation()
        
        # Post-Generation Stats Logging (Optional)
        # Statistics are updated inside create_new_generation usually
        
    print("--- Evolution Finished ---")

# Run the async loop
if __name__ == "__main__":
    asyncio.run(run_evolution())