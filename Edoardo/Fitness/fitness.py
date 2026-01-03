from Edoardo.Phenotype.phenotype import Phenotype
from Edoardo.Utils import LLM
from Edoardo.Data.cluttr import CLUTTRManager
from Edoardo.Fitness.fitness_function import UnifiedFitnessCalculator

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
    
    def evaluate(self, individual: Phenotype, problem: dict) -> float:
        """
        Evaluate a single individual on a single problem.
        """
        # Run individual
        # Note: individual.run is SYNC as per Edoardo/Phenotype/phenotype.py
        # It takes initial_input and answer_only=True
        
        try:
            outputs = individual.run(initial_input=problem['question'], answer_only=True)
            generated_ans = outputs[-1] if outputs else ""
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

    def evaluate_population(self, population: list[Phenotype], problem_pool: list[dict]):
        """
        Evaluates a list of phenotypes against a pool of problems.
        Updates the .fitness attribute of each phenotype with the average loss.
        """
        for individual in population:
            total_loss = 0.0
            
            for problem in problem_pool:
                loss = self.evaluate(individual, problem)
                total_loss += loss
            
            # Average loss
            if problem_pool:
                avg_loss = total_loss / len(problem_pool)
            else:
                avg_loss = 1.0 # Default high loss if no problems
            
            individual.fitness = avg_loss
