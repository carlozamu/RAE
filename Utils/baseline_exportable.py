import asyncio
import time
from typing import List, Callable, Dict, Any
from Data.clutrr import CLUTTRManager

async def _evaluate_single_baseline_problem(
    problem: dict, 
    llm_client, 
    fitness, 
    semaphore: asyncio.Semaphore,
    prompt_builder_func: Callable
):
    """Worker function to evaluate a single problem for a baseline."""
    async with semaphore:
        # 1. Dynamically build the prompt (Zero-Shot or Few-Shot)
        # Assuming your problem dict has 'story' and 'query/question' keys
        metadata = problem.get('metadata', {})
        story = metadata.get('story', '')
        query = metadata.get('query', '')
        
        if prompt_builder_func:
            prompt_string = prompt_builder_func(story=story, query=query)
        else:
            prompt_string = query

        try:
            # 2. STRICT DECODING: Enforce the 1-word constraint at the hardware level
            generated_ans = await llm_client.generate_text(
                user_prompt=prompt_string, 
                temperature=0.0,
                max_tokens=12
            )
            token_used = (len(prompt_string) + len(generated_ans)) // 4
        except Exception as e:
            print(f"Error executing baseline prompt: {e}")
            generated_ans = ""
            token_used = len(prompt_string) // 4

        # 3. Clean and parse
        clean_ans = generated_ans.replace(".", "").replace("\n", "").strip().lower()
        answer_length = len(clean_ans.split()) if clean_ans else 0

        expected = problem['answer']
        is_correct = False
        
        if problem.get('task_type') == 'cluttr':
            mapped_response = CLUTTRManager.map_to_relation(clean_ans)
            is_correct = (mapped_response == expected.lower())
        else:
            is_correct = (clean_ans == expected.strip().lower())

        # 4. Compute fitness score
        score = fitness.calculator.compute_score(
            is_correct=is_correct,
            token_count=token_used
        )
        
        return is_correct, score, answer_length, problem.get('metadata', {}).get('reasoning_length', 0), token_used

async def evaluate_baseline_batch(
    baseline_name: str,
    problem_batch: List[dict],
    llm_client,
    fitness,
    prompt_builder_func: Callable,
    max_concurrent: int = 50
) -> Dict[str, Any]:
    """
    Evaluates a batch of problems against a specific baseline and prints a formatted report.
    Can be called directly from main.py every generation.
    """
    #print(f"\n--- Running {baseline_name} ---")
    start_time = time.time()
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = [
        _evaluate_single_baseline_problem(problem, llm_client, fitness, semaphore, prompt_builder_func)
        for problem in problem_batch
    ]
    
    results = await asyncio.gather(*tasks)

    # Aggregation Variables
    stratified_stats = {}
    total_correct = 0
    total_score = 0.0
    total_1_word = 0
    total_2_word = 0
    total_words_generated = 0
    total_tokens_used = 0

    for is_correct, score, answer_length, hop_length, tokens_used in results:
        # Update Word Count Metrics
        total_words_generated += answer_length
        total_tokens_used += tokens_used
        if answer_length == 1:
            total_1_word += 1
        elif answer_length == 2:
            total_2_word += 1
        
        # Update Stratified Metrics
        if hop_length not in stratified_stats:
            stratified_stats[hop_length] = {"correct": 0, "total": 0}
            
        stratified_stats[hop_length]["total"] += 1
        total_score += score
        
        if is_correct:
            stratified_stats[hop_length]["correct"] += 1
            total_correct += 1

    execution_time = time.time() - start_time
    total_problems = len(problem_batch)
    
    overall_accuracy = (total_correct / total_problems) * 100 if total_problems > 0 else 0
    overall_fitness = (total_score / total_problems) * 100 if total_problems > 0 else 0
    avg_tokens = (total_tokens_used / total_problems) if total_problems > 0 else 0

    # Return stats for main.py to log/plot
    return {
        "accuracy": overall_accuracy,
        "fitness": overall_fitness,
        "execution_time": execution_time,
        "avg_tokens": avg_tokens,
    }

