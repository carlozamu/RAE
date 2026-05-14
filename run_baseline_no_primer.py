"""
ERA: Evolving Reasoning Agents
Baseline Execution Script (No Primer + Logging)
"""
import asyncio
import time
import gc
import torch

# --- Internal Modules ---
from Fitness.fitness import Fitness
from Data.cluttr import CLUTTRManager
from Utils.LLM import LLM

# --- Configuration ---
MODEL_NAME = "google/gemma-3-1b-it" 
BASE_URL = "http://localhost:8000"  
TOTAL_PROBLEMS = 500
MAX_CONCURRENT_REQUESTS = 50 

def force_cleanup():
    print("\n🧹 Performing Memory Cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU Memory Released.")

async def _evaluate_single_problem(idx, problem, dataset_manager, llm_client, fitness, semaphore):
    """Worker function with index tracking for debug printing."""
    async with semaphore:
        story = problem['metadata']['story']
        query = problem['metadata']['query']
        prompt = dataset_manager.build_prompt_clutrr_baseline_no_primer(story, query)

        try:
            generated_ans = await llm_client.generate_text(user_prompt=prompt)
            token_used = (len(prompt) + len(generated_ans)) // 4
        except Exception as e:
            print(f"Error executing baseline prompt: {e}")
            generated_ans = ""
            token_used = len(prompt) // 4

        # Clean the generated answer for word counting and logging
        clean_ans = generated_ans.strip()
        
        # 1. Check if the model yapped (more than 1 word)
        answer_length = len(clean_ans.split())

        expected = problem['answer']
        is_correct = False
        
        if problem.get('task_type') == 'cluttr':
            mapped_response = CLUTTRManager.map_to_relation(clean_ans.lower())
            is_correct = (mapped_response == expected)
        else:
            is_correct = (clean_ans.lower() == expected.strip().lower())

        # 2. Print the tuple every 50 iterations
        if idx % 50 == 0:
            print(f"[Problem {idx:03d}] Expected: '{expected}' | Generated: '{clean_ans}' | Mapped: '{mapped_response if problem.get('task_type') == 'cluttr' else 'N/A'}'")

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
    
    print(f"Fetching {TOTAL_PROBLEMS} problems from dataset...")
    initial_problems_pool = dataset_manager.get_batch(batch_size=TOTAL_PROBLEMS)

    print(f"Starting Baseline evaluation with {MAX_CONCURRENT_REQUESTS} concurrent workers...\n")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Pass 'idx' using enumerate
    tasks = [
        _evaluate_single_problem(idx, problem, dataset_manager, llm_client, fitness, semaphore)
        for idx, problem in enumerate(initial_problems_pool)
    ]
    
    results = await asyncio.gather(*tasks)

    # 3. Aggregate all the results
    correct_count = sum(1 for res in results if res[0])
    total_score = sum(res[1] for res in results)
    average_answer_length = sum(res[2] for res in results) / len(initial_problems_pool) if initial_problems_pool else 0

    execution_time = time.time() - start_time

    print("\n--- Evaluation Finished ---")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Correct Answers: {correct_count}/{len(initial_problems_pool)}")
    print(f"Average Answer Length: {average_answer_length:.2f}")

    avg_score = total_score / len(initial_problems_pool)
    accuracy = (correct_count / len(initial_problems_pool)) * 100
    
    print(f"Baseline Average Fitness Score: {avg_score:.4f}")
    print(f"Baseline Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    try:
        asyncio.run(run_baseline())
    finally:
        force_cleanup()