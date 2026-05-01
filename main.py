"""
ERA: Evolving Reasoning Agents
Main Execution Script
"""
import asyncio
import time
import gc
import torch
import subprocess

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Phenotype.phenotype import Phenotype
from Mutations.mutator import Mutator
from Data.cluttr import CLUTTRManager
from Utils.utilities import _get_next_innovation_number, log_generation_to_json, plot_complexity_vs_fitness
from Utils.LLM import LLM
from ERA.init_pop import initialize_population

# --- New Architectural Modules ---
from Selection.selection import RankBasedSelection
from Species.species_breeder import SpeciesBreeder
from Species.speciation_engine import SpeciationEngine

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" # Ensure this matches your local Ollama or vLLM setup
BASE_URL = "http://localhost:8000"  # Ensure this matches your local Ollama or vLLM setup
MAX_GENERATIONS = 500
MAX_TIME_SECONDS = 3600 * 10 # 10 Hours
TARGET_FITNESS = 95.0        # Higher is better (Max is 100.0)
STARTING_PROMPT = "You are an expert in reasoning. Given the input, provide the accurate response to the following problem."
NUM_INDIVIDUALS = 50  
TARGET_SPECIES = 4
DROPOFF_AGE = 8 # Generations a species can survive without improving max fitness  
BATCH_SIZE = 50 # Number of problems each individual is evaluated on per generation
SELECTION_PRESSURE = 1.5
ELITISM_RATIO = 0.2
PROPORTIONAL_STEP = 0.075 # P-Controller gain for dynamic thresholding


def force_cleanup():
    """Releases GPU memory."""
    print("\n🧹 Performing Memory Cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU Memory Released.")

async def run_evolution():
    print("\n--- Initializing ERA System ---")
    
    # 1. Initialize API and Core Tools
    llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL) 
    fitness_evaluator = Fitness(llm=llm_client, use_reasoning=True)
    mutator = Mutator(breeder_llm_client=llm_client)
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    
    # 2. Population Initialization
    initial_problems_pool = dataset_manager.get_batch(batch_size=BATCH_SIZE)
    _get_next_innovation_number() # Reset global innovation tracker

    print("🌱 Seeding Minimal Population...")
    evaluated_population = await initialize_population(
        num_individuals=NUM_INDIVIDUALS, 
        prompt=STARTING_PROMPT, 
        problems_pool=initial_problems_pool, 
        llm_client=llm_client, 
        fitness_evaluator=fitness_evaluator
    )
    
    print(f"Initialized population: {len(evaluated_population)} clones. Best Gen 0 Fitness: {evaluated_population[0].genome.fitness:.4f}")

    # 3. Setup the Micro and Macro Layers
    # Micro-Layer: Parent selection and breeding
    selector = RankBasedSelection(selection_pressure=SELECTION_PRESSURE)
    breeder = SpeciesBreeder(selector=selector, mutator=mutator, elitism_ratio=ELITISM_RATIO)
    
    # Macro-Layer: Ecology, speciation, and resource allocation
    speciation_engine = SpeciationEngine(
        breeder=breeder,
        target_population_size=NUM_INDIVIDUALS,
        target_species_count=TARGET_SPECIES,
        dropoff_age=DROPOFF_AGE,
        proportioanl_step=PROPORTIONAL_STEP
    )

    print("✅ Evolution Engine Ready.")

    # 4. Main Evolution Loop
    print("\n🚀 Starting Evolution Loop...")
    start_time = time.time()
    generation_idx = 0
    encountered_species = {}  # Dictionary of (species_id, hex color) for plotting

    while generation_idx < MAX_GENERATIONS:
        # A. Logging and Analytics for the Current (Evaluated) Generation
        best_fit = float('-inf')
        worse_fit = float('inf')
        total_fit = 0.0
        total_nodes = 0
        total_edges = 0
        
        for phenotype in evaluated_population:
            fit = phenotype.genome.fitness
            total_fit += fit
            total_nodes += len(phenotype.genome.nodes)
            total_edges += len(phenotype.genome.connections)

            if fit > best_fit: best_fit = fit
            if fit < worse_fit: worse_fit = fit

        avg_fit = total_fit / NUM_INDIVIDUALS
        avg_nodes = total_nodes / NUM_INDIVIDUALS
        avg_edges = total_edges / NUM_INDIVIDUALS
        
        print(f"\n--- Generation {generation_idx} ---")
        print(f"Fitness -> Avg: {avg_fit:.2f} | Best: {best_fit:.2f} | Worst: {worse_fit:.2f}")
        print(f"Structure -> Avg Nodes: {avg_nodes:.1f} | Avg Edges: {avg_edges:.1f}")
        
        # B. Utilities (Plotting & File Logging)
        log_generation_to_json(evaluated_population, generation_idx) 
        plot_path = plot_complexity_vs_fitness(evaluated_population, generation_idx, encountered_species) 
        # (Optional: Only pop open the viewer every 10 generations to avoid spamming the screen)
        #if generation_idx % 10 == 0: subprocess.Popen(['xdg-open', plot_path]) 
        subprocess.Popen(['xdg-open', plot_path]) 

        # C. Stop Criteria Checks
        if best_fit >= TARGET_FITNESS:
            print(f"\n🏆 SUCCESS: Target Fitness ({TARGET_FITNESS}) reached! Final Best: {best_fit:.4f}")
            break
        if (time.time() - start_time) > MAX_TIME_SECONDS:
            print(f"\n🛑 STOP: Maximum time limit ({MAX_TIME_SECONDS}s) reached.")
            break

        # D. THE GENETIC STEP (Macro + Micro Layers)
        print(f"🧬 Speciating and Breeding Generation {generation_idx + 1}...")
        
        genomes_to_breed = [p.genome for p in evaluated_population]
        unevaluated_next_gen_genomes = await speciation_engine.step_generation(genomes_to_breed)
        # wrap the new genomes into Phenotypes with empty fitness for the next evaluation step
        unevaluated_next_gen: list[Phenotype] = [Phenotype(genome=g, llm_client=llm_client) for g in unevaluated_next_gen_genomes]
        
        print(f"📊 Active Species count for next gen: {len(speciation_engine.species_list)}")
        print(f"🌡️ Current Compatibility Threshold: {speciation_engine.compatibility_threshold:.2f}")

        # E. THE EVALUATION STEP
        print("🧪 Fetching new problem pool and evaluating offspring...")
        current_problem_pool = dataset_manager.get_batch(batch_size=BATCH_SIZE)
        
        # Next generation evaluates on the NEW batch of problems
        await fitness_evaluator.evaluate_population(unevaluated_next_gen, current_problem_pool)
        
        # F. Prepare for next iteration
        evaluated_population = unevaluated_next_gen
        generation_idx += 1

    print("--- Evolution Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_evolution())
    finally:
        force_cleanup()