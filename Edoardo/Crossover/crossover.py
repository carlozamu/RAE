from copy import deepcopy

import numpy as np
from Edoardo.Genome.AgentGenome import AgentGenome


class Crossover:
    @staticmethod
    def create_offspring(parent1_genome: AgentGenome, parent2_genome: AgentGenome) -> AgentGenome:
        """
        Given two parent genomes, creates an offspring genome by combining their nodes and connections.
        For each node and connection present in either parent, the offspring randomly inherits it from one of the parents.
        If a node or connection is only present in one parent, it is inherited from that parent.
        
        :param parent1_genome: AgentGenome
        :param parent2_genome: AgentGenome
        :return: AgentGenome
        """
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
        edge_list = list(c for c in parent1_genome.connections.keys()) + list(c for c in parent2_genome.connections.keys())
        edge_list = list(set(edge_list))  # Remove duplicates
        for edge_id in edge_list:
            if edge_id in parent1_genome.connections.keys() and edge_id in parent2_genome.connections.keys():
                if np.random.rand() < 0.5:
                    inherited_connection = deepcopy(parent1_genome.connections[edge_id])
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
                else:
                    inherited_connection = deepcopy(parent2_genome.connections[edge_id])
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
            else:
                try:
                    inherited_connection = deepcopy(parent1_genome.connections[edge_id])
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
                except KeyError:
                    inherited_connection = deepcopy(parent2_genome.connections[edge_id])
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
        return offspring_genome