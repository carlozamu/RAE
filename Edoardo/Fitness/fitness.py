from Edoardo.Phenotype.phenotype import Phenotype
from fitness_function import UnifiedFitnessCalculator

class Fitness:
    def __init__(self):
        self.calculator = UnifiedFitnessCalculator(
            w_accuracy=2.0,
            w_rationale=2.0,
            w_token_cost=0.001,
            w_complexity_cost=0.07  #! DA SISTEMARE IN BASE A QUANTO GROSSO è IL GRAFO DI NORMA!
        )
    
    def evaluate(self, individual: Phenotype) -> float:
        # TODO funzione per testare qui l'individio (tipo chiama API o altro) oppure appena dopo aver generato la risposta
       
        #? Calcolo di ESEMPIO (per ora)
        model_answer = "The capital of France is Paris"
        model_rationale = "Paris has been the capital of France since 987 AD."
        target_answer = "The capital of France is Paris"
        target_rationale = "Paris has been the capital of France since 987 AD."
        num_nodes = 2
        num_edges = 3

        return self.calculator.compute(
            generated_ans=model_answer,
            target_ans=target_answer,
            generated_rat=model_rationale, # opzionale (se si valuta il reasoning)
            target_rat=target_rationale, # opzionale (se si valuta il reasoning)
            num_nodes=num_nodes,
            num_edges=num_edges
        )["loss"]

        