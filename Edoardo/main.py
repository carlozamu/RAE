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
from math import inf
from Fitness.fitness import Fitness
from Mutations.mutator import Mutator
from Data.cluttr import CLUTTRManager
from Data.cot import CoTManager
from Utils.utilities import _get_next_innovation_number
from Utils.LLM import LLM
from ERA.init_pop import initialize_population
from Evolution_Manager.evolution_manager import EvolutionManager
from Selection.selection import TournamentSelection
from Generation_Manager.generation_manager import CommaPlusStrategy

# Initialize LLM client endpoint, exposes get_embeddings and generate_text methods
llm_client = LLM()
print("LLM client initialized.")

USE_REASONING = False # Toggle to enable/disable reasoning evaluation
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
initial_problems_pool = dataset_manager.get_batch(size=3)
starting_prompt = "You are an expert reasoning AI. Given the input, provide a detailed and accurate response following the instructions."
population = initialize_population(num_individuals=50, prompt=starting_prompt, problems_pool=initial_problems_pool, llm_client=llm_client, fitness_evaluator=fitness_evaluator)
_get_next_innovation_number()  # Initialize global innovation number tracker, now it becomes 0, next time the function will be called it will return 1
print(f"Initialized population with {len(population)} individuals with fitness of {population[0].genome.fitness}.")

# Strategy Configuration
selection_strategy = TournamentSelection(tournament_size=5)    # <----- SET as HYPERPARAMETER?
survivor_strategy = CommaPlusStrategy(elite_size=3) # Elitism   # <----- SET as HYPERPARAMETER?

# Initialize Evolution Manager
evolution_manager = EvolutionManager(
    selection_strategy=selection_strategy,
    survivor_strategy=survivor_strategy,
    mutator=mutator,
    fitness_evaluator=fitness_evaluator,
    llm_client=llm_client,
    initial_population=population,
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
MAX_GENERATIONS = 500
MAX_TIME_SECONDS = 3600 * 10 # 10 Hours
TARGET_LOSS = 0.20 # Based on threshold analysis (Perfect ~0.15)

async def run_evolution():
    print("\n--- Starting Evolution ---")
    start_time = time.time()
    
    # 3. Evolution Loop
    while True:
        new_gen = evolution_manager.create_new_generation()
        
        # Check Stop Conditions
        # Note: We check fitness achieved in the current step
        
        # Get global best fitness across all species
        best_loss = inf
        worse_loss = -inf
        average_loss = 0.0
        total_individuals = 0
        current_gen = evolution_manager.current_generation_index
        for species_id, individual in new_gen: ##FLAG## statistiche
            total_individuals += 1
            average_loss += individual.genome.fitness

            if individual.genome.fitness < best_loss:
                best_loss = individual.genome.fitness

            if individual.genome.fitness > worse_loss:
                worse_loss = individual.genome.fitness

        average_loss /= total_individuals if total_individuals > 0 else 1
        
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