"""
ERA: Evolving Reasoning Agents
Baseline Execution Script (No Primer + Logging)
"""
import asyncio
from tqdm.asyncio import tqdm
import time

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Data.clutrr import CLUTTRManager
from Utils.utilities import log_and_print
from Phenotype.phenotype import Phenotype

# --- Configuration ---
MAX_CONCURRENT_REQUESTS = 45
async def run_ERA_best_individual(dataset_manager:CLUTTRManager, fitness:Fitness, era_best: Phenotype):
    print("\n--- Initializing ERA Best ---")
    start_time = time.time()
    
    # llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL)
    # dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    # fitness = Fitness(llm=llm_client)
    
    # node_0 = PromptNode(
    #     name="Node 0", 
    #     instruction="Determine the primary relationship established by the given kinship word.", 
    #     innovation_number=0
    # )
    # node_1 = PromptNode(
    #     name="Node 1", 
    #     instruction="As a familial researcher, state the kinship word and respond with exactly one word.", 
    #     innovation_number=1
    # )
    # node_6 = PromptNode(
    #     name="Node 6", 
    #     instruction="Analyze the provided context and extract the specific variables required for the final execution. Prioritize clarity and conciseness in your response.",
    #     innovation_number=6
    # )
    # node_3 = PromptNode(
    #     name="Node 3", 
    #     instruction="As a family historian, identify the specific familial connection implied by the provided context.",
    #     innovation_number=3
    # )


    # nodes_dict = {
    #     0: node_0,
    #     1: node_1,
    #     3: node_3,
    #     6: node_6
    # }

    # connection_1 = Connection(
    #     input_node_in=node_6.innovation_number, 
    #     output_node_in=node_3.innovation_number, 
    #     enabled=True
    # )
    # connection_2 = Connection(
    #     input_node_in=node_6.innovation_number, 
    #     output_node_in=node_0.innovation_number, 
    #     enabled=True
    # )
    # connection_3 = Connection(
    #     input_node_in=node_3.innovation_number, 
    #     output_node_in=node_0.innovation_number, 
    #     enabled=True
    # )
    # connection_4 = Connection(
    #     input_node_in=node_0.innovation_number, 
    #     output_node_in=node_1.innovation_number, 
    #     enabled=True
    # )


    # connection_dict = {
    #     connection_1.innovation_number: connection_1,
    #     connection_2.innovation_number: connection_2,
    #     connection_3.innovation_number: connection_3,
    #     connection_4.innovation_number: connection_4
    # }

    # winner_genome = AgentGenome(
    #     nodes_dict=nodes_dict,
    #     connections_dict=connection_dict,
    #     start_node_innovation_number=6,
    #     end_node_innovation_number=1
    # )

    # phenotype = Phenotype(
    #     genome=winner_genome,
    #     llm_client=llm_client
    # )

    # 2. Fetch the ENTIRE dataset
    print("Fetching the COMPLETE dataset for stratified baseline...")
    initial_problems_pool = dataset_manager.get_entire_dataset_stratified(dataset_manager.build_prompt_clutrr)
    #initial_problems_pool = dataset_manager.get_full_split()

    print(f"Starting Baseline evaluation with {MAX_CONCURRENT_REQUESTS} concurrent workers...\n")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    order = era_best.genome.get_execution_order()
    
    async def sem_task(semaphore, fitness, individual, problem, execution_order):
        async with semaphore:
            return await fitness._evaluate_single_problem(
                individual=individual, 
                problem=problem, 
                execution_order=execution_order
            )

    # Then update your tasks list construction:
    tasks = [
        sem_task(semaphore, fitness, era_best, problem, order)
        for problem in initial_problems_pool
    ]
    
    # 3. Fire tasks
    results = await tqdm.gather(*tasks, desc="Evaluating Problems")

    # 4. Stratified Aggregation & Word Count Tracking
    stratified_stats = {}
    total_correct = 0
    total_score = 0.0
    
    # NEW: Variables for tracking word counts
    total_1_word = 0
    total_2_word = 0
    total_words_generated = 0

    for idx, (is_correct, score, answer_length) in enumerate(results):
        length = initial_problems_pool[idx]['metadata']['reasoning_length']
        
        # Update Word Count Metrics
        total_words_generated += answer_length
        if answer_length == 1:
            total_1_word += 1
        elif answer_length == 2:
            total_2_word += 1
        
        # Update Stratified Metrics
        if length not in stratified_stats:
            stratified_stats[length] = {"correct": 0, "total": 0}
            
        stratified_stats[length]["total"] += 1
        total_score += score
        
        if is_correct:
            stratified_stats[length]["correct"] += 1
            total_correct += 1

    execution_time = time.time() - start_time

    # 5. Output the Stratified Report
    total_problems = len(initial_problems_pool)
    
    log_and_print("\n" + "="*50)
    log_and_print("🎯 STRATIFIED BASELINE REPORT")
    log_and_print("="*50)
    log_and_print(f"Execution Time: {execution_time:.2f} seconds")
    log_and_print(f"Total Problems Evaluated: {total_problems}\n")

    # Sort the dictionary by reasoning length to print in order
    for length in sorted(stratified_stats.keys()):
        stats = stratified_stats[length]
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        log_and_print(f"Level {length:02d} Hops: Accuracy {acc:05.2f}% ({stats['correct']}/{stats['total']})")

    log_and_print("-" * 50)
    overall_accuracy = (total_correct / total_problems) * 100
    overall_fitness = total_score / total_problems
    average_length = total_words_generated / total_problems
    
    log_and_print(f"Overall Dataset Accuracy: {overall_accuracy:.2f}%")
    log_and_print(f"Overall Average Fitness:  {overall_fitness:.4f}")
    log_and_print("-" * 50)
    log_and_print(f"Average Answer Length:    {average_length:.2f} words")
    log_and_print(f"Exactly 1-Word Answers:   {total_1_word} ({(total_1_word/total_problems)*100:.1f}%)")
    log_and_print(f"Exactly 2-Word Answers:   {total_2_word} ({(total_2_word/total_problems)*100:.1f}%)")
    log_and_print("=" * 50)
