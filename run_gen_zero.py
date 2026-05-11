"""
ERA: Evolving Reasoning Agents
Generation 0 Baseline Execution Script
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
TOTAL_PROBLEMS = 500 # Statistically significant sample size
MAX_CONCURRENT_REQUESTS = 50 # vLLM/Ollama batching limit

def force_cleanup():
    """Releases GPU memory."""
    print("\n🧹 Performing Memory Cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU Memory Released.")

async def _evaluate_single_problem(problem, dataset_manager, llm_client, fitness, semaphore):
    """Worker function to evaluate a single problem under semaphore constraints."""
    async with semaphore:

        prompt = f"""<start_of_turn>system\n{problem['system_instructions']}\n<end_of_turn>\n<start_of_turn>user\n{problem['question']}\n<end_of_turn>\n<start_of_turn>model\n{problem['primer']}"""
        
        try:
            # FIX: Added 'await' to properly resolve the async generation
            generated_ans = await llm_client.generate_text(user_prompt=prompt)
            # Safe token estimation
            token_used = (len(prompt) + len(generated_ans)) // 4
        except Exception as e:
            print(f"Error executing baseline prompt: {e}")
            generated_ans = ""
            token_used = len(prompt) // 4

        expected = problem['answer']
        is_correct = False
        
        if problem.get('task_type') == 'cluttr':
            mapped_response = CLUTTRManager.map_to_relation(generated_ans.strip().lower())
            is_correct = (mapped_response == expected)
        else:
            is_correct = (generated_ans.strip().lower() == expected.strip().lower())

        # Calculate Score
        score = fitness.calculator.compute_score(
            is_correct=is_correct,
            token_count=token_used
        )
        
        return is_correct, score

async def run_baseline():
    print("\n--- Initializing ERA Baseline ---")
    start_time = time.time()
    
    # 1. Initialize API and Core Tools
    llm_client = LLM(model_name=MODEL_NAME, base_url=BASE_URL)
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")
    fitness = Fitness(llm=llm_client)
    
    # 2. Problems Pool (Fetch 500 problems)
    print(f"Fetching {TOTAL_PROBLEMS} problems from dataset...")
    initial_problems_pool = dataset_manager.get_batch(batch_size=TOTAL_PROBLEMS)

    print(f"Starting Generation 0 baseline evaluation with {MAX_CONCURRENT_REQUESTS} concurrent workers...")
    
    # 3. Asynchronous Batching Setup
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    tasks = [
        _evaluate_single_problem(problem, dataset_manager, llm_client, fitness, semaphore)
        for problem in initial_problems_pool
    ]
    
    # Fire all tasks and wait for completion
    results = await asyncio.gather(*tasks)

    # 4. Aggregate Results
    correct_count = sum(1 for is_correct, _ in results if is_correct)
    total_score = sum(score for _, score in results)
    
    execution_time = time.time() - start_time

    # 5. Metrics Output
    print("\n--- Evaluation Finished ---")
    print(f"Execution Time: {execution_time:.2f} seconds")
    print(f"Correct Answers: {correct_count}/{len(initial_problems_pool)}")

    avg_score = total_score / len(initial_problems_pool)
    accuracy = (correct_count / len(initial_problems_pool)) * 100
    
    print(f"Generation 0 Baseline Average Fitness Score: {avg_score:.4f}")
    print(f"Generation 0 Baseline Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    try:
        asyncio.run(run_baseline())
    finally:
        force_cleanup()