from math import inf
from Utils import LLM
from Data.cluttr import CLUTTRManager
from Fitness.fitness_function import UnifiedFitnessCalculator

class Fitness:
    def __init__(self, llm: LLM, use_reasoning: bool = True) -> None:
        self.use_reasoning = use_reasoning
        self.calculator = UnifiedFitnessCalculator(
            w_accuracy=2.0,
            w_rationale=2.0 if use_reasoning else 0.0, # Disable rationale weight if disabled
            w_token_cost=0.001,
            w_complexity_cost=0.07,  #! DA SISTEMARE IN BASE A QUANTO GROSSO è IL GRAFO DI NORMA!
            llm=llm
        )
    
    from Phenotype.phenotype import Phenotype
    async def evaluate(self, individual: Phenotype, problem: dict) -> float:
        """
        Evaluate a single individual on a single problem.
        """        
        try:
            outputs = await individual.run(problem=problem['question'])
            generated_ans = outputs['answer']
        except Exception as e:
            print(f"Error executing phenotype: {e}")
            generated_ans = ""

        expected = problem['answer']
        
        # Check if this is a CLUTTR task
        custom_accuracy = None
        if problem.get('task_type') == 'cluttr':
            # Use specific CLUTTR evaluation
            mapped_response = CLUTTRManager.map_to_relation(generated_ans)
            # Binary accuracy: 1.0 if match, 0.0 otherwise
            custom_accuracy = 1.0 if mapped_response == expected else 0.0
            
            # For logging/debug, we might want to know what it mapped to
            # print(f"CLUTTR: '{generated_ans}' -> '{mapped_response}' vs '{expected}'")

        # Calculate fitness using UnifiedFitnessCalculator
        target_rationale = problem.get('rationale') if self.use_reasoning else None
        
        result = self.calculator.compute(
            generated_ans=generated_ans,
            target_ans=expected,
            generated_rat=None, # We don't have explicit rationale from agent yet
            target_rat=target_rationale,
            num_nodes=len(individual.genome.nodes),
            num_edges=len(individual.genome.connections),
            custom_accuracy=custom_accuracy
        )
        
        return result["loss"]

    def _update_fitness(self, problems_pool: list[dict], phenotype: Phenotype):
        fitness = 0.0
        for problem in problems_pool:
            fitness += self.evaluate(phenotype, problem)
        phenotype.genome.fitness = fitness / len(problems_pool) if problems_pool else inf
        #print(f"Updated fitness: {self.genome.fitness}") 
