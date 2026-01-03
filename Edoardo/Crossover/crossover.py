import numpy as np
from Genome.agent_genome import AgentGenome


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
        offspring_genome.start_node_innovation_number = parent1_genome.start_node_innovation_number
        offspring_genome.end_node_innovation_number = parent1_genome.end_node_innovation_number
        node_list = list(parent1_genome.nodes.keys()) + list(parent2_genome.nodes.keys())
        node_list = list(set(node_list))
        for node_innovation_number in node_list:
            if node_innovation_number in parent1_genome.nodes and node_innovation_number in parent2_genome.nodes:
                if np.random.rand() < 0.5:
                    offspring_genome.add_node(parent1_genome.nodes[node_innovation_number].copy())
                else:
                    offspring_genome.add_node(parent2_genome.nodes[node_innovation_number].copy())
            else:
                try:
                    offspring_genome.add_node(parent1_genome.nodes[node_innovation_number].copy())
                except KeyError:
                    offspring_genome.add_node(parent2_genome.nodes[node_innovation_number].copy())
        edge_list = list(c for c in parent1_genome.connections.keys()) + list(c for c in parent2_genome.connections.keys())
        edge_list = list(set(edge_list))  # Remove duplicates
        for edge_id in edge_list:
            if edge_id in parent1_genome.connections.keys() and edge_id in parent2_genome.connections.keys():
                if np.random.rand() < 0.5:
                    inherited_connection = parent1_genome.connections[edge_id].copy()
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
                else:
                    inherited_connection = parent2_genome.connections[edge_id].copy()
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
            else:
                try:
                    inherited_connection = parent1_genome.connections[edge_id].copy()
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
                except KeyError:
                    inherited_connection = parent2_genome.connections[edge_id].copy()
                    offspring_genome.add_connection(inherited_connection.in_node, inherited_connection.out_node)
        
        # Remove any cycles that may have been created during crossover
        offspring_genome.remove_cycles()
        
        return offspring_genome