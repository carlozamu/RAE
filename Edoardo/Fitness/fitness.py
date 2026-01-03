from Edoardo.Phenotype.phenotype import Phenotype
from Edoardo.Utils import LLM
from fitness_function import UnifiedFitnessCalculator

class Fitness:
    def __init__(self, llm_client: LLM) -> None:
        self.calculator = UnifiedFitnessCalculator(
            w_accuracy=2.0,
            w_rationale=2.0,
            w_token_cost=0.001,
            w_complexity_cost=0.07,  #! DA SISTEMARE IN BASE A QUANTO GROSSO è IL GRAFO DI NORMA!
            llm_client=llm_client
        )
    
    def evaluate(self, individual: Phenotype, problem: dict) -> float:
        """
        Evaluate a single individual on a single problem.
        """
        # Run individual
        # Note: individual.run returns a list of outputs (one per node). We usually take the final valid answer.
        # The Phenotype.run method signature is: run(self, initial_input: str = "", answer_only: bool = True) -> list[str]
        outputs = individual.run(initial_input=problem['input'], answer_only=True)
        
        # Take the last output as the answer
        generated_ans = outputs[-1] if outputs else ""
        # TODO: Handle rationale extraction if the agent outputs it separately or if we need to parse it
        # For now, we assume the whole output is the answer
        
        # In a real scenario, we might want to ask the agent to output explicit "Answer: ... Rationale: ..." sections
        # But UnifiedFitnessCalculator handles generated_rat as optional.
        
        # Calculate fitness (Loss)
        result = self.calculator.compute(
            generated_ans=generated_ans,
            target_ans=problem['target'],
            generated_rat=None, # We don't have explicit rationale from agent yet
            target_rat=problem.get('rationale'),
            num_nodes=len(individual.genome.nodes),
            num_edges=len(individual.genome.connections)
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

        