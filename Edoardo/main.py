"""
1) Population initialization (simple same prompt, single node, 50 individuals, fitness computed on one individual and assigned to all)

2) Loop until termination condition (fitness under threshold (@DEV) or max time (X hours/ k-pressed) reached):

2) Loop until termination condition (fitness under threshold (@DEV) or max time (X hours/ k-pressed) reached):
    2.1) Loop for each species:
            - Selection (intra-species)
            - Crossover
            - Mutation
            - Fitness evaluation
    2.2) Average fitness calculation per species
    2.3) Replacement (keep 50 individuals top for each generation)

Variabili globali:
    next innovation number (incremental integer)
    generation index (incremental integer)
    species age (incremental integer per species)


Cose da loggare:
    generation number
    best-worse fitness per generation
    average fitness per species per generation
    average number of nodes and connections per individual per generation
    species count per generation
    paragone con prompt engineering

Cose da fare:
evolution manager todo
    mutazione genome max(0.1,0.9^(generation_number+1))
    mutazione gene 1/NUMERO DI NODI DEL GENOMA
    
    selection fix 

    genotype/phenotype
    new species handling


Report
    1) Abstract + Introduction (probem + why NEAT)
    2) Dataset + benchmarking
    2) Selection + Fitnes
    3) Crossover + Mutation
    3) Speciation + Evolution Management
    4) Experiments + Results
"""
from Edoardo.Fitness.fitness import Fitness
from Edoardo.Mutations.mutator import Mutator
from Edoardo.Data.cluttr import CLUTTRManager
from Edoardo.Data.cot import CoTManager
from Utils.utilities import _get_next_innovation_number
from Utils.LLM import LLM
from ERA.init_pop import initialize_population

# Initialize LLM client endpoint, exposes get_embeddings and generate_text methods
llm_client = LLM()
print("LLM client initialized.")

USE_REASONING = True # Toggle to enable/disable reasoning evaluation
fitness_evaluator = Fitness(llm=llm_client, use_reasoning=USE_REASONING)
mutator = Mutator(breeder_llm_client=llm_client)
# Choose Dataset based on Reasoning flag
if USE_REASONING:
    print("Reasoning Evaluation ENABLED. Using CoT-Collection.")
    dataset_manager = CoTManager(split="train")
else:
    print("Reasoning Evaluation DISABLED. Using CLUTTR Dataset.")
    dataset_manager = CLUTTRManager(split_config="gen_train234_test2to10")

print("Fitness, Mutator & Dataset Manager initialized.")

# Initialize population
starting_prompt = "You are an expert reasoning AI. Given the input, provide a detailed and accurate response following the instructions."
population = initialize_population(num_individuals=50, prompt=starting_prompt)
_get_next_innovation_number()  # Initialize global innovation number tracker, now it becomes 0, next time the function will be called it will return 1
print(f"Initialized population with {len(population)} individuals with fitness of {population[0].fitness}.")

# Main loop ...
from Edoardo.Evolution_Manager.evolution_manager import EvolutionManager
from Edoardo.Selection.selection import TournamentSelection
from Edoardo.Generation_Manager.generation_manager import CommaPlusStrategy

# Strategy Configuration
selection_strategy = TournamentSelection(tournament_size=3)
survivor_strategy = CommaPlusStrategy(elite_size=2) # Elitism

# Initialize Evolution Manager
evolution_manager = EvolutionManager(
    selection_strategy=selection_strategy,
    survivor_strategy=survivor_strategy,
    mutator=mutator,
    fitness_evaluator=fitness_evaluator,
    dataset_manager=dataset_manager,
    num_parents=2,
    per_species_hof_size=5,
    hof_parent_ratio=0.2
)

# TODO: Add the actual loop calling evolution_manager.create_new_generation()
print("Evolution Manager initialized.")

