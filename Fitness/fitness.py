import asyncio
from typing import List, Dict, Tuple
from Utils.LLM import LLM
from Data.cluttr import CLUTTRManager
from Phenotype.phenotype import Phenotype
from Fitness.fitness_function import UnifiedFitnessCalculator

PERCENTAGE_FAILURE_THRESHOLD = 0.7 # If more than thi % of the batch is failed, trigger circuit breaker

class Fitness:
    def __init__(self, llm: LLM, use_reasoning: bool = True) -> None:
        self.use_reasoning = use_reasoning
        self.calculator = UnifiedFitnessCalculator(
            accuracy_score=1.0,
            max_penalty=0.9,
            llm=llm
        )
    
    async def _evaluate_single_problem(self, individual: Phenotype, problem: Dict) -> Tuple[float, int]:
        """Evaluates a single problem and returns (Score, Tokens_Used)."""        
        try:
            response = await individual.run(problem=problem['question'])
            generated_ans = response['answer']
            stats = response['stats']
            token_used = stats.get('total_tokens', 0)
        except Exception as e:
            print(f"Error executing phenotype: {e}")
            generated_ans = ""
            token_used = 0

        expected = problem['answer']
        is_correct = False
        
        if problem.get('task_type') == 'cluttr':
            mapped_response = CLUTTRManager.map_to_relation(generated_ans.strip().lower())
            is_correct = (mapped_response == expected)
        else:
            is_correct = (generated_ans.strip().lower() == expected.strip().lower())
            
        score = self.calculator.compute_score(
            is_correct=is_correct,
            token_count=token_used
        )
        return score, token_used

    async def _evaluate_individual_full(self, individual: Phenotype, problem_pool: List[Dict], batch_size: int) -> List[int]:
        """
        Helper method to evaluate a single individual across all problems sequentially,
        preserving the Circuit Breaker logic.
        """
        total_score = 0.0
        failed_count = 0
        token_usages = []
        
        for i, problem in enumerate(problem_pool):
            # This remains sequential per-individual to support the DAG and circuit breaker
            score, tokens = await self._evaluate_single_problem(individual, problem)
            total_score += score
            
            if tokens > 0:
                token_usages.append(tokens)
            
            if score < 0.1:
                failed_count += 1
            else:
                failed_count = 0 
                
            # --- CIRCUIT BREAKER ---
            if failed_count > len(problem_pool) * PERCENTAGE_FAILURE_THRESHOLD:
                # Aggressive Penalty
                remaining_questions = batch_size - (i + 1)
                total_score -= (remaining_questions * self.calculator.acc_score)
                break 
        
        # Calculate final fitness
        avg_score = (total_score * 100) / batch_size
        individual.genome.fitness = max(0.01, avg_score)
        
        # Return tokens so the parent gather() can collect them all
        return token_usages

    async def evaluate_population(self, population: List[Phenotype], problem_pool: List[Dict]):
        """
        Evaluates the population using a Semaphore to strictly control 
        GPU memory pressure (Continuous Batching limit).
        """
        if not problem_pool:
            raise ValueError("Problem pool cannot be empty.")
        
        print(f"Evaluating population of {len(population)} individuals on {len(problem_pool)} problems with circuit breaker threshold at {PERCENTAGE_FAILURE_THRESHOLD*100}% failures.")

        batch_size = len(problem_pool)

        # --- THE HARDWARE LIMITER ---
        MAX_CONCURRENT = 50 
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def _bounded_evaluate(individual):
            """Wrapper to enforce the semaphore limit per individual."""
            async with semaphore:
                return await self._evaluate_individual_full(individual, problem_pool, batch_size)

        # 1. Create the bounded tasks
        tasks = [
            _bounded_evaluate(individual)
            for individual in population
        ]

        # 2. Fire gather. It will attempt to run all 60, but the Semaphore 
        # will physically block it from sending more than MAX_CONCURRENT at once.
        results = await asyncio.gather(*tasks)

        # 3. Aggregate tokens and apply Red Queen shift
        all_token_usages = []
        for individual_tokens in results:
            all_token_usages.extend(individual_tokens)

        if all_token_usages:
            self.calculator.update_baselines(all_token_usages)
    