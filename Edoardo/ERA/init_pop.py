"""Initialize population module."""
import numpy as np
from Edoardo.Phenotype.phenotype import Phenotype
from Edoardo.Utils.LLM import LLM
from Gene.gene import PromptNode
from Genome.agent_genome import AgentGenome

def initialize_population(num_individuals: int, prompt: str, llm_client: LLM, problems_pool: list[dict]) -> list[Phenotype]:
    """Initialize a population with a given number of individuals."""
    population: list[Phenotype] = []
    
    # Create a new PromptNode for first individual
    node = PromptNode(
        name=f"Start+End",
        instruction=prompt,
        innovation_number=0
    )
    # Create a new AgentGenome for first individual
    genome = AgentGenome(
        nodes_dict={node.innovation_number: node},
        connections_dict={},
        start_node_innovation_number=node.innovation_number,
        end_node_innovation_number=node.innovation_number
    )

    # Compute fitness for one individual
    phenotype = Phenotype(genome=genome, llm_client=llm_client)
    phenotype._update_fitness(problems_pool)  # Empty problem pool for initial fitness

    for _ in range(num_individuals):
        # Clone the genome for each individual
        new_phenotype = phenotype.deep_copy()
        population.append(new_phenotype)

    return population