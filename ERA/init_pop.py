"""
Population Initialization Module.
Creates the minimal "Generation 0" starting point.
"""
from Fitness.fitness import Fitness
from Phenotype.phenotype import Phenotype
from Utils.LLM import LLM
from Gene.gene import PromptNode
from Genome.agent_genome import AgentGenome

async def initialize_population(num_individuals: int, prompt: str, problems_pool: list[dict], llm_client: LLM, fitness_evaluator: Fitness) -> list[Phenotype]:
    """
    Initializes a population by evaluating a single minimal graph and cloning it.
    """
    population: list[Phenotype] = []
    
    # 1. Create a minimal structure (Start and End are the same node)
    node = PromptNode(
        name=f"Start+End",
        instruction=prompt,
        innovation_number=0
    )
    
    genome = AgentGenome(
        nodes_dict={node.innovation_number: node},
        connections_dict={},
        start_node_innovation_number=node.innovation_number,
        end_node_innovation_number=node.innovation_number
    )

    phenotype = Phenotype(genome=genome, llm_client=llm_client)
    
    # 2. Evaluate the single pioneer organism using the NEW fitness logic
    # We pass it as a list of 1 to utilize the evaluate_population method
    await fitness_evaluator.evaluate_population([phenotype], problems_pool)

    # 3. Clone the evaluated organism to fill the population
    for _ in range(num_individuals):
        # deep_copy() must ensure a new unique Genome ID but preserve the innovation numbers!
        new_phenotype = phenotype.deep_copy()
        population.append(new_phenotype)

    return population