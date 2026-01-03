import uuid

import numpy as np
from Edoardo.Gene.gene import PromptNode
from Edoardo.Gene.connection import Connection

class AgentGenome:
    def __init__(self, nodes_dict: dict[str, PromptNode] = None, connections_dict: dict[str, Connection] = None, start_node_innovation_number:int=None, end_node_innovation_number:int=None, fitness:float=np.inf):
        self.id = str(uuid.uuid4())
        self.nodes: dict[str, PromptNode] = nodes_dict if nodes_dict is not None else {} # dict of innovation_number: PromptNode
        self.connections: dict[str, Connection] = connections_dict if connections_dict is not None else {} # dict of innovation_number: Connection
        self.start_node_innovation_number = start_node_innovation_number if start_node_innovation_number is not None else None
        self.end_node_innovation_number = end_node_innovation_number if end_node_innovation_number is not None else None
        self.fitness = fitness if fitness is not None else 0.0

    def add_node(self, node: PromptNode):
        self.nodes[node.innovation_number] = node

    def add_connection(self, in_node_innovation_number, out_node_innovation_number):
        # Check for duplicates to prevent multi-edges between same nodes
        for conn in self.connections.values():
            if conn.in_node == in_node_innovation_number and conn.out_node == out_node_innovation_number:
                return # Connection already exists, do nothing

        # Create new connection (Innovation number generated automatically by Connection class)
        new_conn = Connection(in_node_innovation_number, out_node_innovation_number)
        self.connections[new_conn.innovation_number] = new_conn

    def get_execution_order(self) -> list[PromptNode]:
            """
            Returns a topologically sorted list of nodes (Kahn's Algorithm).
            Ensures that a node is only executed after ALL its dependencies (incoming connections) are processed.
            
            Example: If Start->A, Start->B, B->C, C->A.
            Order: Start -> B -> C -> A.
            """
            if self.start_node_innovation_number is None:
                return []

            # 1. Build Adjacency List and In-Degree Count
            # adj[u] = [v1, v2] means u points to v1 and v2
            adj = {node_id: [] for node_id in self.nodes}
            in_degree = {node_id: 0 for node_id in self.nodes}

            # Only consider ENABLED connections
            for conn in self.connections.values():
                if conn.enabled:
                    u, v = conn.in_node, conn.out_node
                    # Safety check to ensure nodes still exist in the genome
                    if u in adj and v in adj:
                        adj[u].append(v)
                        in_degree[v] += 1

            # 2. Initialize Queue with the Start Node
            # We start specifically with the defined start_node to ensure the chain begins correctly.
            # (In a valid DAG, start_node should have in_degree 0, but we force it just in case)
            queue = [self.start_node_innovation_number]
            
            sorted_nodes = []
            visited_count = 0

            # 3. Kahn's Algorithm Loop
            while queue:
                # Pop the first ready node
                curr_id = queue.pop(0)
                
                # Fetch the actual object and add to result
                if curr_id in self.nodes:
                    sorted_nodes.append(self.nodes[curr_id])
                    visited_count += 1

                # Iterate through neighbors
                for neighbor in adj[curr_id]:
                    in_degree[neighbor] -= 1
                    
                    # If neighbor has no more unsatisfied dependencies, it's ready to run
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # 4. Cycle / Reachability Detection
            # If the number of sorted nodes < total nodes reachable from start, 
            # it means there was a cycle or disconnected dead-ends.
            
            # Check if END node is missing (critical failure)
            has_end = any(n.id == self.end_node_innovation_number for n in sorted_nodes)
            if not has_end and self.end_node_innovation_number in self.nodes:
                print("Warning: End node not reachable or caught in a cycle.")
            
            return sorted_nodes
    
    def copy(self):
        """
        Deep copies the entire genome.
        This is used when selecting a parent to produce a child.
        """
        new_genome = AgentGenome()
        new_genome.fitness = self.fitness
        
        # Copy all nodes (preserving IDs)
        for node in self.nodes.values():
            new_genome.nodes[node.innovation_number] = node.copy()
            
        # Copy all connections
        for conn in self.connections.values():
            new_genome.connections[conn.innovation_number] = conn.copy()
            
        return new_genome
    