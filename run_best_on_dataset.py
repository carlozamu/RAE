"""
ERA: Evolving Reasoning Agents
Baseline Execution Script (No Primer + Logging)
"""
import asyncio
from tqdm.asyncio import tqdm
import time
import gc
from typing import Dict
import torch

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Data.clutrr import CLUTTRManager
from Gene.connection import Connection
from Gene.gene import PromptNode
from Genome.agent_genome import AgentGenome
from Phenotype.phenotype import Phenotype
from Utils.LLM import LLM

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" 
BASE_URL = "http://localhost:8000"  
MAX_CONCURRENT_REQUESTS = 50 

def force_cleanup():
    print("\n🧹 Performing Memory Cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU Memory Released.")

async def _evaluate_single_problem(individual:Phenotype, problem:Dict, execution_order:list[tuple[PromptNode, list[int]]], fitness:Fitness, semaphore:asyncio.Semaphore):
    """Worker function with index tracking for debug printing."""
    async with semaphore:

        try:
            response = await individual.run(problem=problem['question'], execution_order=execution_order)
            token_used = response['stats'].get('total_tokens', 0)
        except Exception as e:
            print(f"Error executing baseline prompt: {e}")
            response = ""
            token_used = 0

        # Clean the generated answer for word counting and logging
        clean_ans:str = response['answer'].strip()
        
        # 1. Get the exact word count
        answer_length = len(clean_ans.split())

        expected = problem['answer']
        is_correct = False
        
        if problem.get('task_type') == 'cluttr':
            mapped_response = CLUTTRManager.map_to_relation(clean_ans.lower())
            is_correct = (mapped_response == expected)
        else:
            is_correct = (clean_ans.lower() == expected.strip().lower())

        score = fitness.calculator.compute_score(
            is_correct=is_correct,
            token_count=token_used
        )
        
        return is_correct, score, answer_length

async def run_baseline():
    print("\n--- Initializing ERA Baseline ---")
    start_time = time.time()
    
    llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL)
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    fitness = Fitness(llm=llm_client)
    
    # 2. Fetch the ENTIRE dataset
    print("Fetching the COMPLETE dataset for stratified baseline...")
    initial_problems_pool = dataset_manager.get_entire_dataset_stratified(dataset_manager.build_prompt_clutrr)
    #initial_problems_pool = dataset_manager.get_full_split()

    print(f"Starting Baseline evaluation with {MAX_CONCURRENT_REQUESTS} concurrent workers...\n")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    node_77 = PromptNode(
        name="Node 77", 
        instruction="Restrict the investigation to the elements explicitly mentioned, avoiding tangential explorations.",
        innovation_number=77
    )
    node_127 = PromptNode(
        name="Node 127", 
        instruction="Assume you are a data analyst specializing in predictive modeling. Focus solely on identifying the data points - numerical, textual, or otherwise - that demonstrably shift the core concept's trajectory, and present only that information in a concise, statistically-driven format.",
        innovation_number=127
    )
    node_92 = PromptNode(
        name="Node 92", 
        instruction="Recognize the central claim or point being made within the text.",
        innovation_number=92
    )
    node_84 = PromptNode(
        name="Node 84", 
        instruction="Craft a single, declarative sentence that summarizes the argument.",
        innovation_number=84
    )

    nodes_dict = {
        77: node_77,
        127: node_127,
        92: node_92,
        84: node_84
    }

    connection_1 = Connection(
        input_node_in=node_77.innovation_number, 
        output_node_in=node_127.innovation_number, 
        enabled=True
    )
    connection_2 = Connection(
        input_node_in=node_77.innovation_number, 
        output_node_in=node_92.innovation_number, 
        enabled=True
    )
    connection_3 = Connection(
        input_node_in=node_127.innovation_number, 
        output_node_in=node_92.innovation_number, 
        enabled=True
    )
    connection_4 = Connection(
        input_node_in=node_92.innovation_number, 
        output_node_in=node_84.innovation_number, 
        enabled=True
    )

    connection_dict = {
        connection_1.innovation_number: connection_1,
        connection_2.innovation_number: connection_2,
        connection_3.innovation_number: connection_3,
        connection_4.innovation_number: connection_4
    }

    winner_genome = AgentGenome(
        nodes_dict=nodes_dict,
        connections_dict=connection_dict,
        start_node_innovation_number=77,
        end_node_innovation_number=84
    )

    phenotype = Phenotype(
        genome=winner_genome,
        llm_client=llm_client
    )

    order = winner_genome.get_execution_order()
    
    tasks = [
        _evaluate_single_problem(individual=phenotype, problem=problem, execution_order=order, fitness=fitness, semaphore=semaphore)
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
    
    print("\n" + "="*50)
    print("🎯 STRATIFIED BASELINE REPORT")
    print("="*50)
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Total Problems Evaluated: {total_problems}\n")

    # Sort the dictionary by reasoning length to print in order
    for length in sorted(stratified_stats.keys()):
        stats = stratified_stats[length]
        acc = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"Level {length:02d} Hops: Accuracy {acc:05.2f}% ({stats['correct']}/{stats['total']})")

    print("-" * 50)
    overall_accuracy = (total_correct / total_problems) * 100
    overall_fitness = total_score / total_problems
    average_length = total_words_generated / total_problems
    
    print(f"Overall Dataset Accuracy: {overall_accuracy:.2f}%")
    print(f"Overall Average Fitness:  {overall_fitness:.4f}")
    print("-" * 50)
    print(f"Average Answer Length:    {average_length:.2f} words")
    print(f"Exactly 1-Word Answers:   {total_1_word} ({(total_1_word/total_problems)*100:.1f}%)")
    print(f"Exactly 2-Word Answers:   {total_2_word} ({(total_2_word/total_problems)*100:.1f}%)")
    print("=" * 50)

if __name__ == "__main__":
    try:
        asyncio.run(run_baseline())
    finally:
        force_cleanup()