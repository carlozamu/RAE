"""
Cose da loggare:
    generation number
    best-worse fitness per generation
    average fitness per species per generation
    average number of nodes and connections per individual per generation
    species count per generation
    paragone con prompt engineering

ERA: Evolving Reasoning Agents
Main Execution Script
"""
import asyncio
import time
import sys
import gc
import torch
import traceback
from math import inf
import subprocess # Needed to spawn the image viewer process

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Mutations.mutator import Mutator
from Data.cluttr import CLUTTRManager
from Data.cot import CoTManager
from Utils.utilities import _get_next_innovation_number, plot_complexity_vs_fitness
from Utils.MarkDownLogger import md_logger
from Utils.LLM import LLM
from ERA.init_pop import initialize_population
from Evolution_Manager.evolution_manager import EvolutionManager
from Selection.selection import TournamentSelection
from Generation_Manager.generation_manager import CommaPlusStrategy
from Species.species import Species

# --- Configuration ---
USE_REASONING = False       # Toggle to enable/disable reasoning evaluation
MAX_GENERATIONS = 500
MAX_TIME_SECONDS = 3600 * 10 # 10 Hours
TARGET_LOSS = 0.20          # Stop if loss drops below this

def force_cleanup():
    """
    Forces Python and PyTorch to release all GPU memory.
    Call this on shutdown or error.
    """
    print("\n🧹 Performing Memory Cleanup...")
    
    # 1. Force Python Garbage Collector
    gc.collect()
    
    # 2. Clear PyTorch Cache (if local tensors were created)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    
    print("✅ GPU Memory Released.")

async def run_evolution():
    print("\n--- Initializing ERA System ---")
    
    # 1. Initialize Components
    # Initialize LLM
    # NOTE: Ensure your LLM class loads the Embedder on CPU to avoid VRAM conflict!
    llm_client = LLM() 
    print("LLM client initialized.")

    fitness_evaluator = Fitness(llm=llm_client, use_reasoning=USE_REASONING)
    mutator = Mutator(breeder_llm_client=llm_client)

    # Dataset Selection
    if USE_REASONING:
        print("Reasoning Evaluation ENABLED. Using CoT-Collection.")
        dataset_manager = CoTManager(split="train")
    else:
        print("Reasoning Evaluation DISABLED. Using CLUTTR Dataset.")
        dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")

    # 2. Population Initialization
    initial_problems_pool = dataset_manager.get_batch()
    starting_prompt = "You are an expert reasoning AI. Given the input, provide a detailed and accurate response following the instructions."
    
    print("🌱 Seeding Population...")
    population = await initialize_population(
        num_individuals=30, 
        prompt=starting_prompt, 
        problems_pool=initial_problems_pool, 
        llm_client=llm_client, 
        fitness_evaluator=fitness_evaluator
    )
    
    # Reset/Init innovation counter
    _get_next_innovation_number() 
    print(f"Initialized population: {len(population)} agents. Best Gen 0 Fitness: {population[0].genome.fitness:.4f}")

    # 3. Evolution Strategy Setup
    selection_strategy = TournamentSelection(tournament_size=7)  
    survivor_strategy = CommaPlusStrategy()

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
    print("Evolution Manager Ready.")

    # 4. Main Evolution Loop
    print("\n🚀 Starting Evolution Loop...")
    start_time = time.time()
    x=True
    encountered_species: dict[(int, str)] = {}  # Dictionary of (species_id, hex color)
    while x==True:
        # A. Create Next Generation (Includes Selection, Mutation, Evaluation)
        # This returns the NEW population list (Species list or individual list depending on your manager return)
        new_gen = await evolution_manager.create_new_generation()

        current_gen_idx = evolution_manager.current_generation_index
        plot_path = plot_complexity_vs_fitness(generation_data=new_gen, generation_idx=current_gen_idx, species_colors_registry=encountered_species)
        subprocess.Popen(['xdg-open', plot_path])
        
        # B. Statistics Calculation
        current_gen = evolution_manager.current_generation_index
        
        best_loss = inf
        worse_loss = -inf
        total_loss = 0.0
        total_individuals = 0
        
        for item in new_gen:
            # Handle both list formats safely
            individual = item[1]
            
            fit = individual.genome.fitness
            total_individuals += 1
            total_loss += fit

            if fit < best_loss: best_loss = fit
            if fit > worse_loss: worse_loss = fit

        average_loss = total_loss / total_individuals if total_individuals > 0 else 0.0
        
        # C. Logging to Console
        print(f"Gen {current_gen} | Avg Loss: {average_loss:.4f} | Best: {best_loss:.4f} | Worst: {worse_loss:.4f}")

        # D. Stop Criteria Checks
        if best_loss <= TARGET_LOSS:
            print(f"\n🏆 SUCCESS: Target Loss ({TARGET_LOSS}) reached! Final Best: {best_loss:.4f}")
            break
            
        if (time.time() - start_time) > MAX_TIME_SECONDS:
            print(f"\n🛑 STOP: Maximum time limit ({MAX_TIME_SECONDS}s) reached.")
            break
            
        if current_gen >= MAX_GENERATIONS:
            print(f"\n🛑 STOP: Maximum generations ({MAX_GENERATIONS}) reached.")
            break

    print("--- Evolution Finished ---")

# --- Entry Point with Safety Wrapper ---
if __name__ == "__main__":
    try:
        asyncio.run(run_evolution())
    except KeyboardInterrupt:
        print("\n\n👋 Manual Interruption Detected (CTRL+C).")
    finally:
        # This ALWAYS runs, ensuring VRAM is freed
        force_cleanup()