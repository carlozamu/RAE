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

    async def evaluate_population(self, population: List[Phenotype], problem_pool: List[Dict]):
        """
        Evaluates the population, scales scores up, and calculates the Red Queen 
        token baselines for the next generation.
        """
        if not problem_pool:
            raise ValueError("Problem pool cannot be empty.")

        batch_size = len(problem_pool)
        all_token_usages = [] # Tracks API usage to shift baselines

        for individual in population:
            total_score = 0.0
            failed_count = 0
            
            for i, problem in enumerate(problem_pool):
                score, tokens = await self._evaluate_single_problem(individual, problem)
                total_score += score
                
                if tokens > 0:
                    all_token_usages.append(tokens)
                
                # If score < 0.1, it means they got the question wrong (due to missing ans_points)
                if score < 0.1:
                    failed_count += 1
                else:
                    failed_count = 0 
                    
                # --- CIRCUIT BREAKER ---
                if failed_count > len(problem_pool) * PERCENTAGE_FAILURE_THRESHOLD:
                    # Aggressive Penalty: Deduct points for the unseen questions
                    remaining_questions = batch_size - (i + 1)
                    total_score -= (remaining_questions * self.calculator.acc_score)
                    break 
            
            # 1. Average the score across the entire batch (scales cleanly up to 100)
            avg_score = (total_score * 100) / batch_size
            
            # 2. Convert to strictly positive fitness and assign
            # If the circuit breaker tanked the score into negatives, it safely floors at 0.01
            individual.genome.fitness = max(0.01, avg_score)

        # 3. Apply the Co-evolutionary Red Queen Shift for the NEXT generation
        if all_token_usages:
            self.calculator.update_baselines(all_token_usages)