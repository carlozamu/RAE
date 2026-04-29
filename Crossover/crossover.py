import numpy as np
from Genome.agent_genome import AgentGenome

class Crossover:
    @staticmethod
    def create_offspring(parent1: AgentGenome, parent2: AgentGenome) -> AgentGenome:
        offspring = AgentGenome()
        offspring.start_node_innovation_number = parent1.start_node_innovation_number
        offspring.end_node_innovation_number = parent1.end_node_innovation_number
        
        # 1. Determine Fitness (Assuming higher is better)
        if parent1.fitness > parent2.fitness:
            better_parent, worse_parent = parent1, parent2
            equal_fitness = False
        elif parent2.fitness > parent1.fitness:
            better_parent, worse_parent = parent2, parent1
            equal_fitness = False
        else:
            better_parent, worse_parent = parent1, parent2
            equal_fitness = True

        # 2. Inherit Nodes (FIXED: 50/50 Semantic Inheritance)
        for node_id, better_node in better_parent.nodes.items():
            if node_id in worse_parent.nodes:
                # Node exists in both: 50/50 chance to get exact prompt semantics from either
                worse_node = worse_parent.nodes[node_id]
                offspring.nodes[node_id] = better_node.copy() if np.random.rand() < 0.5 else worse_node.copy()
            else:
                # Disjoint/Excess in better parent
                offspring.nodes[node_id] = better_node.copy()
                
        if equal_fitness:
            # Union the remaining disjoint/excess nodes from worse parent
            for node_id, worse_node in worse_parent.nodes.items():
                if node_id not in offspring.nodes:
                    offspring.nodes[node_id] = worse_node.copy()

        # 3. Inherit Connections
        for conn_id, better_conn in better_parent.connections.items():
            if conn_id in worse_parent.connections:
                worse_conn = worse_parent.connections[conn_id]
                inherited_conn = better_conn.copy() if np.random.rand() < 0.5 else worse_conn.copy()
                
                # 75% Disable Rule
                if not better_conn.enabled or not worse_conn.enabled:
                    inherited_conn.enabled = False if np.random.rand() < 0.75 else True
                else:
                    inherited_conn.enabled = True
                    
                offspring.connections[conn_id] = inherited_conn
            else:
                offspring.connections[conn_id] = better_conn.copy()

        if equal_fitness:
            for conn_id, worse_conn in worse_parent.connections.items():
                if conn_id not in better_parent.connections:
                    offspring.connections[conn_id] = worse_conn.copy()

        # 4. Fix Graph Integrity
        # Step A: Disable any cycles created by merging the two graphs
        offspring.remove_cycles()
        
        # Step B: Prune faulty/orphaned topology (NEW)
        Crossover._prune_invalid_topology(offspring)
        
        return offspring

    @staticmethod
    def _prune_invalid_topology(genome: AgentGenome):
        """
        Removes dead ends and unreached nodes. 
        A valid node MUST be reachable from Start AND be able to reach End.
        """
        if not genome.nodes or genome.start_node_innovation_number not in genome.nodes or genome.end_node_innovation_number not in genome.nodes:
            return

        adj = {node_id: [] for node_id in genome.nodes}
        rev_adj = {node_id: [] for node_id in genome.nodes}
        
        # Only map enabled connections
        for conn in genome.connections.values():
            if conn.enabled:
                adj[conn.in_node].append(conn.out_node)
                rev_adj[conn.out_node].append(conn.in_node)

        # Forward BFS: What can start reach?
        reachable_from_start = set()
        queue = [genome.start_node_innovation_number]
        reachable_from_start.add(genome.start_node_innovation_number)
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, []):
                if neighbor not in reachable_from_start:
                    reachable_from_start.add(neighbor)
                    queue.append(neighbor)

        # Backward BFS: What can reach end?
        can_reach_end = set()
        queue = [genome.end_node_innovation_number]
        can_reach_end.add(genome.end_node_innovation_number)
        while queue:
            curr = queue.pop(0)
            for neighbor in rev_adj.get(curr, []):
                if neighbor not in can_reach_end:
                    can_reach_end.add(neighbor)
                    queue.append(neighbor)

        # Valid nodes are the intersection
        valid_nodes = reachable_from_start.intersection(can_reach_end)

        # 1. Delete invalid nodes
        invalid_nodes = set(genome.nodes.keys()) - valid_nodes
        for node_id in invalid_nodes:
            del genome.nodes[node_id]

        # 2. Delete invalid connections (any connection touching a deleted node)
        invalid_conns = []
        for conn_id, conn in genome.connections.items():
            if conn.in_node not in valid_nodes or conn.out_node not in valid_nodes:
                invalid_conns.append(conn_id)
                
        for conn_id in invalid_conns:
            del genome.connections[conn_id]