import asyncio
from typing import List, Dict, Tuple
from Gene.gene import PromptNode
from Utils.LLM import LLM
from Data.clutrr import CLUTTRManager
from Phenotype.phenotype import Phenotype
from Fitness.fitness_function import UnifiedFitnessCalculator

PERCENTAGE_FAILURE_THRESHOLD = 0.7 # If more than thi % of the batch is failed, trigger circuit breaker

class Fitness:
    def __init__(self, llm: LLM, use_reasoning: bool = False) -> None:
        self.use_reasoning = use_reasoning
        self.calculator = UnifiedFitnessCalculator(
            llm=llm
        )
        self.best_accuracy = 0.0
        self.avg_accuracy = 0.0
    
    async def _evaluate_single_problem(self, individual: Phenotype, problem: Dict, execution_order: list[tuple[PromptNode, list[int]]]) -> Tuple[float, int]:
        """Evaluates a single problem and returns (Score, Tokens_Used)."""        
        try:
            response = await individual.run(problem=problem['question'], execution_order=execution_order)
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
            token_count=token_used,
            answer_length=len(generated_ans.split()),
        )
        return score, token_used

    async def _evaluate_individual_full(self, individual: Phenotype, problem_pool: List[Dict]) -> List[tuple[int, float]]:
        """
        Helper method to evaluate a single individual across all problems sequentially,
        preserving the Circuit Breaker logic.
        """
        if not individual.genome.evaluated:
            total_score = 0.0
            failed_count = 0
            accuracy = 0
            problems_evaluated = 0
            token_usages = []

            execution_order = individual.genome.get_execution_order()
            
            for i, problem in enumerate(problem_pool):
                # This remains sequential per-individual to support the DAG and circuit breaker
                score, tokens = await self._evaluate_single_problem(individual, problem, execution_order)
                total_score += score
                problems_evaluated += 1
                
                if tokens > 0:
                    token_usages.append(tokens)
                
                if score < 0.1:
                    failed_count += 1
                else:
                    failed_count = 0 
                    accuracy += 1
                    
                # --- CIRCUIT BREAKER ---
                if failed_count > len(problem_pool) * PERCENTAGE_FAILURE_THRESHOLD:
                    break 
            
            # Calculate final fitness
            avg_score = ((total_score) / (problems_evaluated))*100 if problems_evaluated > 0 else 0.0
            accuracy = ((accuracy) / (problems_evaluated))*100 if problems_evaluated > 0 else 0.0
            avg_tokens = sum(token_usages)/len(token_usages) if token_usages else 0.0
            individual.genome.fitness = float(max(0.01, avg_score))
            individual.genome.accuracy = accuracy
            individual.genome.avg_tokens = avg_tokens
            individual.genome.evaluated = True
        
            # Return tokens so the parent gather() can collect them all
            return token_usages, accuracy
        else:
            return [individual.genome.avg_tokens], individual.genome.accuracy

    async def evaluate_population(self, population: List[Phenotype], problem_pool: List[Dict]):
        """
        Evaluates the population using a Semaphore to strictly control 
        GPU memory pressure (Continuous Batching limit).
        """
        if not problem_pool:
            raise ValueError("Problem pool cannot be empty.")
        
        print(f"Evaluating population of {len(population)} individuals on {len(problem_pool)} problems with circuit breaker threshold at {PERCENTAGE_FAILURE_THRESHOLD*100}% failures.")

        # --- THE HARDWARE LIMITER ---
        MAX_CONCURRENT = 50 
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        async def _bounded_evaluate(individual):
            """Wrapper to enforce the semaphore limit per individual."""
            async with semaphore:
                return await self._evaluate_individual_full(individual, problem_pool)

        # 1. Create the bounded tasks
        tasks = [
            _bounded_evaluate(individual)
            for individual in population
        ]

        # 2. Fire gather. It will attempt to run all, but the Semaphore 
        # will physically block it from sending more than MAX_CONCURRENT at once.
        results = await asyncio.gather(*tasks)

        # 3. Aggregate tokens and apply Red Queen shift
        all_token_usages: list[int] = []
        all_accuracies: list[float] = []
        for individual_tokens, accuracy in results:
            all_token_usages.extend(individual_tokens)
            all_accuracies.append(accuracy)

        if all_token_usages:
            self.calculator.update_baselines(all_token_usages)
        if all_accuracies:
            max_accuracy = max(all_accuracies)
            avg_accuracy = sum(all_accuracies) / len(all_accuracies)
            self.best_accuracy = max_accuracy
            self.avg_accuracy = avg_accuracy
    