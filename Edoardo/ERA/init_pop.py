"""Initialize population module."""
import numpy as np
from Gene.gene import PromptNode
from Genome.agent_genome import AgentGenome

def initialize_population(num_individuals: int, prompt: str) -> list[AgentGenome]:
    """Initialize a population with a given number of individuals."""
    population: list[AgentGenome] = []
    for i in range(num_individuals):
        # Create a new PromptNode for each individual
        node = PromptNode(
            name=f"Start+End",
            instruction=prompt,
            innovation_number=0
        )
        # Create a new AgentGenome for each individual
        genome = AgentGenome(
            nodes_dict={node.innovation_number: node},
            connections_dict={},
            start_node_innovation_number=node.innovation_number,
            end_node_innovation_number=node.innovation_number
        )
        population.append(genome)
    # Compute fitness for one individual
    fitness = np.inf # Placeholder fitness value
    for individual in population:
        individual.fitness = fitness
    return population