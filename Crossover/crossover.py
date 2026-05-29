import numpy as np
from Genome.agent_genome import AgentGenome

class Crossover:
    @staticmethod
    def create_offspring(parent1: AgentGenome, parent2: AgentGenome) -> AgentGenome:
        offspring = AgentGenome()
        # 1. Determine Fitness First
        if parent1.fitness > parent2.fitness:
            better_parent, worse_parent = parent1, parent2
        elif parent2.fitness > parent1.fitness:
            better_parent, worse_parent = parent2, parent1
        else: #choose the smallest as better partent if fitness is equal, otherwise choose randomly
            size_p1 = len(parent1.nodes) + sum(1 for conn in parent1.connections.values() if conn.enabled)
            size_p2 = len(parent2.nodes) + sum(1 for conn in parent2.connections.values() if conn.enabled)
            if size_p1 < size_p2:
                better_parent, worse_parent = parent1, parent2
            elif size_p2 < size_p1:
                better_parent, worse_parent = parent2, parent1
            else:
                if np.random.rand() < 0.5:
                    better_parent, worse_parent = parent1, parent2
                else:
                    better_parent, worse_parent = parent2, parent1

        # 2. Inherit Pointers from the Fitter Parent
        offspring.start_node_innovation_number = better_parent.start_node_innovation_number
        offspring.end_node_innovation_number = better_parent.end_node_innovation_number

        # 2. Inherit Nodes (FIXED: 50/50 Semantic Inheritance)
        worst_parent_node_ids = set(worse_parent.nodes.keys())
        for node_id, better_node in better_parent.nodes.items():
            if node_id in worst_parent_node_ids:
                # Node exists in both: 50/50 chance to get exact prompt semantics from either
                random_number = np.random.rand()
                if random_number < 0.5:
                    offspring.nodes[node_id] = better_node.copy()
                else:
                    offspring.nodes[node_id] = worse_parent.nodes[node_id].copy()
            else:
                # Disjoint/Excess in better parent
                offspring.nodes[node_id] = better_node.copy()
                
        # if equal_fitness:
        #     # Union the remaining disjoint/excess nodes from worse parent
        #     for _id in worst_parent_node_ids:
        #         if _id not in list(offspring.nodes.keys()):
        #             offspring.nodes[_id] = worse_parent.nodes[_id].copy()

        # 3. Inherit Connections
        for conn_id, better_conn in better_parent.connections.items():
            offspring.connections[conn_id] = better_conn.copy()

        # if equal_fitness:
        #     for conn_id, worse_conn in worse_parent.connections.items():
        #         if conn_id not in list(offspring.connections.keys()):
        #             offspring_nodes_list = list(offspring.nodes.keys())
        #             if worse_conn.in_node in offspring_nodes_list and worse_conn.out_node in offspring_nodes_list:
        #                 offspring.connections[conn_id] = worse_conn.copy()

        # 4. Fix Graph Integrity
        #offspring.remove_cycles()
        
        return offspring
