"""
ERA: Evolving Reasoning Agents
Main Execution Script
"""
import asyncio
import time
import subprocess

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Genome.agent_genome import AgentGenome
from Phenotype.phenotype import Phenotype
from Mutations.mutator import Mutator
from Data.cluttr import CLUTTRManager
from Utils.utilities import HistoryTracker, log_generation_to_markdown, log_and_print, clear_log_file, Plotter, force_cleanup
from Utils.LLM import LLM
from Utils.baseline_exportable import evaluate_baseline_batch
from ERA.init_pop import initialize_population
from Selection.selection import RankBasedSelection
from Species.species_breeder import SpeciesBreeder
from Species.speciation_engine import SpeciationEngine

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" # Ensure this matches your local Ollama or vLLM setup
BASE_URL = "http://localhost:8000"  # Ensure this matches your local Ollama or vLLM setup
MAX_GENERATIONS = 500
MAX_TIME_SECONDS = 3600 * 10 # 10 Hours
TARGET_FITNESS = 95.0        # Higher is better (Max is 100.0)
STARTING_PROMPT = "Task: State only the one kinship word (from the posible answers) that describes the family relationship."
NUM_INDIVIDUALS = 50
TARGET_SPECIES = 4
BATCH_SIZE = 50 # Number of problems each individual is evaluated on per generation
SELECTION_PRESSURE = 1.5
ELITISM_RATIO = 0.2

async def run_evolution():
    clear_log_file() # Start with a clean slate for logging
    log_and_print("\n--- Initializing ERA System ---")
    
    # 1. Initialize API and Core Tools
    llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL) 
    fitness_evaluator = Fitness(llm=llm_client, use_reasoning=False)
    mutator = Mutator(breeder_llm_client=llm_client)
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    
    # 2. Setup the Micro and Macro Layers
    # Micro-Layer: Parent selection and breeding
    selector = RankBasedSelection(selection_pressure=SELECTION_PRESSURE)
    breeder = SpeciesBreeder(selector=selector, mutator=mutator, elitism_ratio=ELITISM_RATIO)
    plotter = Plotter()
    history_manager = HistoryTracker()
    generation_idx = 0
    
    # Macro-Layer: Ecology, speciation, and resource allocation
    speciation_engine = SpeciationEngine(
        breeder=breeder,
        target_population_size=NUM_INDIVIDUALS,
        target_species_count=TARGET_SPECIES,
    )
    
    # 3. Population Initialization
    initial_problems_pool = dataset_manager.get_batch(batch_size=BATCH_SIZE)
    log_and_print("🌱 Seeding Minimal Population...")
    start_init_time = time.time()
    first_gen = await initialize_population(
        num_individuals=NUM_INDIVIDUALS, 
        prompt=STARTING_PROMPT, 
        problems_pool=initial_problems_pool, 
        llm_client=llm_client, 
        fitness_evaluator=fitness_evaluator
    )
    log_and_print(f"🌡️ Current Compatibility Threshold: {speciation_engine.compatibility_threshold:.2f}")
    evaluated_population: list[AgentGenome] = [individual.genome for individual in first_gen]
    speciation_engine._speciate_population(evaluated_population)
    log_and_print("✅ Evolution Engine Ready.")
    init_duration = time.time() - start_init_time

    # Baseline Evaluations for Generation 0
    zero_shot_stats = await evaluate_baseline_batch(
            baseline_name="Zero-Shot Baseline",
            problem_batch=initial_problems_pool,
            llm_client=llm_client,
            fitness=fitness_evaluator,
            prompt_builder_func=CLUTTRManager.build_prompt_clutrr_baseline
    )
    few_shots_stats = await evaluate_baseline_batch(
        baseline_name="Few-Shot Baseline",
        problem_batch=initial_problems_pool,
        llm_client=llm_client,
        fitness=fitness_evaluator,
        prompt_builder_func=CLUTTRManager.build_prompt_clutrr_few_shots
    )

    # Log initial generation summary and plot
    log_and_print(f"\n📊 Generation 0 Summary:")
    log_and_print(f"Baseline Zero-Shot run in {zero_shot_stats['execution_time']:.2f}s with {zero_shot_stats['accuracy']:.2f}% accuracy & {zero_shot_stats['fitness']:.2f} fitness.")
    log_and_print(f"Baseline Few-Shots run in {few_shots_stats['execution_time']:.2f}s with {few_shots_stats['accuracy']:.2f}% accuracy & {few_shots_stats['fitness']:.2f} fitness.")
    log_and_print(f"ERA initialized in {init_duration:.2f}s. with {first_gen[0].genome.fitness:.2f} fitness and {first_gen[0].genome.accuracy:.2f}% accuracy.") 
    log_and_print("-"*30)
    history_manager.record_generation(speciation_engine.species_list, zero_shot_stats, few_shots_stats)
    plot_path = plotter.plot_accuracy_vs_tokens(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
    plot_2_path = plotter.plot_fitness_vs_complexity(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
    subprocess.Popen(['xdg-open', plot_path])
    subprocess.Popen(['xdg-open', plot_2_path]) 

    # 4. Main Evolution Loop
    log_and_print("\n🚀 Starting Evolution Loop...")
    start_time = time.time()

    while generation_idx < MAX_GENERATIONS:
        gen_x_starting_time = time.time()
        # A. THE GENETIC STEP (Macro + Micro Layers)
        log_and_print(f"🧬 Speciating and Breeding Generation {generation_idx + 1}...")
        start_breed_time = time.time()
        unevaluated_next_gen_genomes = await speciation_engine.step_generation(generation=generation_idx)
        breed_duration = time.time() - start_breed_time
        log_and_print(f"⏱️ Breeding completed in {breed_duration:.2f} seconds. Generated {len(unevaluated_next_gen_genomes)} offspring.")
        unevaluated_next_gen: list[Phenotype] = [Phenotype(genome=g, llm_client=llm_client) for g in unevaluated_next_gen_genomes]
        log_and_print(f"📊 Active Species count for generation {generation_idx + 1}: {len(speciation_engine.species_list)}")
        log_and_print(f"🌡️ Current Compatibility Threshold: {speciation_engine.compatibility_threshold:.2f}")

        # B. THE EVALUATION STEP
        log_and_print(f"🧪 Fetching new {BATCH_SIZE} problem pool and evaluating {len(unevaluated_next_gen)} offspring...")
        current_problem_pool = dataset_manager.get_batch(batch_size=BATCH_SIZE)
        start_eval_time = time.time()
        await fitness_evaluator.evaluate_population(unevaluated_next_gen, current_problem_pool)
        eval_duration = time.time() - start_eval_time

        # C. THE BASELINE EVALUATION STEP
        zero_shot_stats = await evaluate_baseline_batch(
            baseline_name="Zero-Shot Baseline",
            problem_batch=current_problem_pool,
            llm_client=llm_client,
            fitness=fitness_evaluator,
            prompt_builder_func=CLUTTRManager.build_prompt_clutrr_baseline
        )
        few_shots_stats = await evaluate_baseline_batch(
            baseline_name="Few-Shot Baseline",
            problem_batch=current_problem_pool,
            llm_client=llm_client,
            fitness=fitness_evaluator,
            prompt_builder_func=CLUTTRManager.build_prompt_clutrr_few_shots
        )
        
        # D. Prepare for next iteration
        evaluated_population: list[AgentGenome] = [individual.genome for individual in unevaluated_next_gen]
        speciation_engine._speciate_population(evaluated_population) # Speciate the population for the logs
        generation_idx += 1 # Increment generation counter

        # E. Logging and Analytics for the Current (Evaluated) Generation
        best_accuracy = fitness_evaluator.best_accuracy
        avg_accuracy = fitness_evaluator.avg_accuracy
        best_fit = log_generation_to_markdown(
            speciation_engine.species_list, 
            best_accuracy, 
            avg_accuracy, 
            zero_shot_stats, 
            few_shots_stats, 
            generation_idx,
            eval_duration
        )
        log_and_print(f"\n📊 Generation {generation_idx} Summary:")
        log_and_print(f"Baseline Zero-Shot run in {zero_shot_stats['execution_time']:.2f}s with {zero_shot_stats['accuracy']:.2f}% accuracy & {zero_shot_stats['fitness']:.2f} fitness.")
        log_and_print(f"Baseline Few-Shots run in {few_shots_stats['execution_time']:.2f}s with {few_shots_stats['accuracy']:.2f}% accuracy & {few_shots_stats['fitness']:.2f} fitness.")
        log_and_print(f"ERA run in {eval_duration:.2f}s. with {best_accuracy:.2f}% accuracy, {best_fit:.2f} best fitness.") 
        gen_x_duration = time.time() - gen_x_starting_time
        log_and_print(f"⏱️ Generation {generation_idx} completed in {gen_x_duration:.2f} seconds.")
        log_and_print("-"*30)

        # F. Plot Generation for visual analysis
        history_manager.record_generation(speciation_engine.species_list, zero_shot_stats, few_shots_stats)
        plot_path = plotter.plot_accuracy_vs_tokens(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        plot_2_path = plotter.plot_fitness_vs_complexity(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        subprocess.Popen(['xdg-open', plot_path])
        subprocess.Popen(['xdg-open', plot_2_path]) 

        # G. Stop Criteria Checks
        if best_fit >= TARGET_FITNESS:
            log_and_print(f"\n🏆 SUCCESS: Target Fitness ({TARGET_FITNESS}) reached! Final Best: {best_fit:.4f}")
            break
        if (time.time() - start_time) > MAX_TIME_SECONDS:
            log_and_print(f"\n🛑 STOP: Maximum time limit ({MAX_TIME_SECONDS}s) reached.")
            break

    log_and_print("--- Evolution Finished ---")

if __name__ == "__main__":
    try:
        asyncio.run(run_evolution())
    finally:
        force_cleanup()