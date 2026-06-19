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
from Data.clutrr import CLUTTRManager
from Utils.utilities import log_generation_to_markdown, log_and_print, clear_log_file, Plotter, force_cleanup, HistoryTracker
from Utils.LLM import LLM
from Utils.baseline_exportable import evaluate_baseline_batch
from ERA.init_pop import initialize_population
from Selection.selection import RankBasedSelection
from Species.species_breeder import SpeciesBreeder
from Species.speciation_engine import SpeciationEngine
from Z_Baselines.run_ERA_best import run_ERA_best_individual

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" 
BASE_URL = "http://localhost:8000"  
MAX_GENERATIONS = 50
MAX_TIME_SECONDS = 3600 * 15 # 15 Hours
TARGET_FITNESS = 50.0        # Higher is better (Max is 100.0)
STARTING_PROMPT = "Task: State only the one kinship word (from the posible answers) that describes the family relationship."
NUM_INDIVIDUALS = 50
TARGET_SPECIES = 5

# --- Main Execution ---
async def run_evolution():
    # 1. Initialize API and Core Tools (Required for both new runs and rehydrated runs)
    llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL) 
    fitness_evaluator = Fitness(llm=llm_client, use_reasoning=False)
    mutator = Mutator(breeder_llm_client=llm_client)
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    dataset = dataset_manager.get_or_create_curated_dataset() 
    
    # 2. Setup the Micro Layer (Instantiates fresh LLM sockets)
    selector = RankBasedSelection()
    breeder = SpeciesBreeder(selector=selector, mutator=mutator)
    plotter = Plotter()
    history_manager = HistoryTracker()
    
    start_time = time.time()

    # ==========================================
    # 3. STATE RECOVERY & MACRO LAYER SETUP
    # ==========================================
    checkpoint_state = history_manager.load_checkpoint()

    if checkpoint_state:
        log_and_print("\n🔄 Valid Checkpoint Found. Rehydrating State...")
        
        # Restore variables from disk
        generation_idx = checkpoint_state["generation_idx"]
        speciation_engine = checkpoint_state["speciation_engine"]
        zero_shot_stats = checkpoint_state["zero_shot_stats"]
        few_shots_stats = checkpoint_state["few_shots_stats"]
        
        # CRITICAL: Re-inject the newly instanced breeder (with active LLM client) into the restored engine
        speciation_engine.breeder = breeder
        
        log_and_print(f"✅ Evolution successfully resumed. Targeting Generation {generation_idx}")
        
    else:
        log_and_print("\n--- Initializing ERA System ---")
        log_and_print("\n🌱 No Checkpoint Found. Seeding Minimal Population...")
        clear_log_file() # Start with a clean slate for logging
    
        generation_idx = 0
        
        # Initialize Macro-Layer from scratch
        speciation_engine = SpeciationEngine(
            breeder=breeder,
            target_population_size=NUM_INDIVIDUALS,
            target_species_count=TARGET_SPECIES,
        )
        
        start_init_time = time.time()
        first_gen = await initialize_population(
            num_individuals=NUM_INDIVIDUALS, 
            prompt=STARTING_PROMPT, 
            problems_pool=dataset, 
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
                problem_batch=dataset,
                llm_client=llm_client,
                fitness=fitness_evaluator,
                prompt_builder_func=CLUTTRManager.build_prompt_clutrr_baseline
        )
        few_shots_stats = await evaluate_baseline_batch(
            baseline_name="Few-Shot Baseline",
            problem_batch=dataset,
            llm_client=llm_client,
            fitness=fitness_evaluator,
            prompt_builder_func=CLUTTRManager.build_prompt_clutrr_few_shots
        )

        # Log initial generation summary and plot
        log_and_print(f"\n📊 Generation 0 Summary:")
        log_and_print(f"Baseline Zero-Shot run in {zero_shot_stats['execution_time']:.2f}s with {zero_shot_stats['accuracy']:.2f}% accuracy & {zero_shot_stats['fitness']:.2f} fitness.")
        log_and_print(f"Baseline Few-Shots run in {few_shots_stats['execution_time']:.2f}s with {few_shots_stats['accuracy']:.2f}% accuracy & {few_shots_stats['fitness']:.2f} fitness.")
        log_and_print(f"ERA initialized in {init_duration:.2f}s. with {first_gen[0].genome.accuracy:.2f}% accuracy and {first_gen[0].genome.fitness:.2f} fitness.") 
        log_and_print("-" * 30)
        
        history_manager.record_generation(speciation_engine.species_list, zero_shot_stats, few_shots_stats)
        
        plot_path = plotter.plot_accuracy_vs_tokens(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        plot_2_path = plotter.plot_fitness_vs_complexity(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        plot_3_path = plotter.plot_accuracy_by_species(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        subprocess.Popen(['xdg-open', plot_path])
        subprocess.Popen(['xdg-open', plot_2_path]) 
        subprocess.Popen(['xdg-open', plot_3_path])
        
        # Save Generation 0 to disk before entering loop
        generation_idx = 1
        history_manager.save_checkpoint(generation_idx, speciation_engine, zero_shot_stats, few_shots_stats)

    # ==========================================
    # 4. MAIN EVOLUTION LOOP
    # ==========================================
    log_and_print("\n🚀 Starting Evolution Loop...")

    while generation_idx <= MAX_GENERATIONS:
        gen_x_starting_time = time.time()
        
        # A. THE GENETIC STEP (Macro + Micro Layers)
        log_and_print(f"\n🧬 Speciating and Breeding Generation {generation_idx}...")
        start_breed_time = time.time()
        unevaluated_next_gen_genomes = await speciation_engine.step_generation(generation=generation_idx)
        breed_duration = time.time() - start_breed_time
        log_and_print(f"⏱️ Breeding completed in {breed_duration:.2f} seconds. Generated {len(unevaluated_next_gen_genomes)} offspring.")
        
        unevaluated_next_gen: list[Phenotype] = [Phenotype(genome=g, llm_client=llm_client) for g in unevaluated_next_gen_genomes]
        active_species_count = sum(+1 for s in speciation_engine.species_list if s.alive)
        log_and_print(f"📊 Active Species count for generation {generation_idx}: {active_species_count}")
        log_and_print(f"🌡️ Current Compatibility Threshold: {speciation_engine.compatibility_threshold:.2f}")

        # B. THE EVALUATION STEP
        log_and_print(f"🧪 Fetching new {len(dataset)} problem pool and evaluating {len(unevaluated_next_gen)} offspring...")
        start_eval_time = time.time()
        await fitness_evaluator.evaluate_population(unevaluated_next_gen, dataset)
        eval_duration = time.time() - start_eval_time
        
        # C. PREPARE FOR NEXT ITERATION
        evaluated_population: list[AgentGenome] = [individual.genome for individual in unevaluated_next_gen]
        speciation_engine._speciate_population(evaluated_population) 

        # D. LOGGING & ANALYTICS
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
        log_and_print("-" * 30)

        # E. PLOTTING
        history_manager.record_generation(speciation_engine.species_list, zero_shot_stats, few_shots_stats)
        plot_path = plotter.plot_accuracy_vs_tokens(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        plot_2_path = plotter.plot_fitness_vs_complexity(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        plot_3_path = plotter.plot_accuracy_by_species(speciation_engine.species_list, zero_shot_stats, few_shots_stats, generation_idx)
        subprocess.Popen(['xdg-open', plot_path])
        subprocess.Popen(['xdg-open', plot_2_path]) 
        subprocess.Popen(['xdg-open', plot_3_path])

        # F. CHECKPOINTING 
        # We save the state mapping to the NEXT generation so a crash or stop 
        # picks up immediately without repeating work.
        generation_idx += 1
        history_manager.save_checkpoint(generation_idx, speciation_engine, zero_shot_stats, few_shots_stats)

        # G. STOP CRITERIA CHECKS
        if best_fit >= TARGET_FITNESS:
            log_and_print(f"\n🏆 SUCCESS: Target Fitness ({TARGET_FITNESS}) reached! Final Best: {best_fit:.4f}")
            break
        if (time.time() - start_time) > MAX_TIME_SECONDS:
            log_and_print(f"\n🛑 STOP: Maximum time limit ({MAX_TIME_SECONDS}s) reached.")
            break

    log_and_print("--- Evolution Finished ---")
    force_cleanup()

    # find individual with the highest accuracy
    best_accuracy_individual = max(evaluated_population, key=lambda x: x.genome.accuracy)
    best_acc_phenotype = Phenotype(genome=best_accuracy_individual, llm_client=llm_client)
    print(f"🏆 Final Best Accuracy: {best_accuracy_individual.genome.accuracy:.4f}")
    await run_ERA_best_individual(dataset_manager, fitness_evaluator, best_acc_phenotype)

    # find individual with the highest fitness
    best_fitness_individual = max(evaluated_population, key=lambda x: x.genome.fitness)
    if best_fitness_individual.id != best_accuracy_individual.id:
        best_fit_phenotype = Phenotype(genome=best_fitness_individual, llm_client=llm_client)
        print(f"🏆 Final Best Fitness: {best_fitness_individual.genome.fitness:.4f}")
        await run_ERA_best_individual(dataset_manager, fitness_evaluator, best_fit_phenotype)

if __name__ == "__main__":
    try:
        asyncio.run(run_evolution())
    finally:
        force_cleanup()