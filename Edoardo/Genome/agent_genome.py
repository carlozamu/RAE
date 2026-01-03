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

    def get_execution_order(self) -> list[tuple[PromptNode, list[str]]]:
            """
            Returns a topologically sorted execution plan.
            Each element is a tuple: (The Node to Run, List of Parent Node IDs to get input from).
            
            Example Output:
            [
            (StartNode, []),
            (NodeB, ['start_id']),
            (NodeC, ['B_id']),
            (NodeA, ['start_id', 'C_id'])  <-- Needs inputs from BOTH Start and C
            ]
            """
            if self.start_node_innovation_number is None:
                return []

            # 1. Build Adjacency List (Forward) and Parent Map (Backward)
            adj = {node_id: [] for node_id in self.nodes}
            
            # This will hold the "context sources" for each node
            # parent_map[target_id] = [source_id1, source_id2...]
            parent_map = {node_id: [] for node_id in self.nodes} 
            
            in_degree = {node_id: 0 for node_id in self.nodes}

            # Only consider ENABLED connections
            for conn in self.connections.values():
                if conn.enabled:
                    u, v = conn.in_node, conn.out_node
                    
                    if u in adj and v in adj:
                        adj[u].append(v)
                        in_degree[v] += 1
                        
                        # Track that 'v' receives input from 'u'
                        parent_map[v].append(u)

            # 2. Initialize Queue with Start Node
            queue = [self.start_node_innovation_number]
            
            execution_plan = [] # List of (Node, [Inputs])
            
            # 3. Kahn's Algorithm Loop
            while queue:
                curr_id = queue.pop(0)
                
                if curr_id in self.nodes:
                    node_obj = self.nodes[curr_id]
                    # Retrieve the list of parents for this specific node
                    input_sources = parent_map.get(curr_id, [])
                    
                    # Append the tuple (Node, Inputs)
                    execution_plan.append((node_obj, input_sources))

                for neighbor in adj[curr_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # 4. Cycle Detection Check
            has_end = any(item[0].id == self.end_node_innovation_number for item in execution_plan)
            if not has_end and self.end_node_innovation_number in self.nodes:
                print("Warning: End node not reachable or caught in a cycle.")
            
            return execution_plan
    
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
    