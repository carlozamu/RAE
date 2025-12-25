from copy import deepcopy

import numpy as np
from Edoardo.Genome.AgentGenome import AgentGenome


class Crossover:
    @staticmethod
    def create_offspring(parent1_genome: AgentGenome, parent2_genome: AgentGenome) -> AgentGenome:
        offspring_genome = AgentGenome()
        node_list = list(parent1_genome.nodes.keys()) + list(parent2_genome.nodes.keys())
        node_list = list(set(node_list))
        for node_id in node_list:
            if node_id in parent1_genome.nodes and node_id in parent2_genome.nodes:
                if np.random.rand() < 0.5:
                    offspring_genome.add_node(deepcopy(parent1_genome.nodes[node_id]))
                else:
                    offspring_genome.add_node(deepcopy(parent2_genome.nodes[node_id]))
            else:
                try:
                    offspring_genome.add_node(deepcopy(parent1_genome.nodes[node_id]))
                except KeyError:
                    offspring_genome.add_node(deepcopy(parent2_genome.nodes[node_id]))
        edge_list = list(c.innovation_number for c in parent1_genome.connections) + list(c.innovation_number for c in parent2_genome.connections)
        edge_list = list(set(edge_list))  # Remove duplicates
        #TODO: fix this part
        for edge_id in edge_list:
            if edge_id in parent1_genome.connections and edge_id in parent2_genome.connections:
                if np.random.rand() < 0.5:
                    offspring_genome.add_connection(deepcopy(parent1_genome.connections[edge_id]))
                else:
                    offspring_genome.add_connection(deepcopy(parent2_genome.connections[edge_id]))
            else:
                try:
                    offspring_genome.add_connection(deepcopy(parent1_genome.connections[edge_id]))
                except KeyError:
                    offspring_genome.add_connection(deepcopy(parent2_genome.connections[edge_id]))
        return offspring_genome