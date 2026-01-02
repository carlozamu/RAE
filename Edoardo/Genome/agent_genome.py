import uuid

import numpy as np
from Gene.gene import PromptNode
from Gene.connection import Connection

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

    def get_linear_chain(self) -> list[PromptNode]:
        """
        Retrieves the linear chain of nodes starting from start_node_id and following enabled connections.
        It's needed to practically execute the agent's prompt sequence.
        """
        chain = []
        if self.start_node_innovation_number is None:
            return chain # return empty if no start node defined
            
        current_node_innovation_number = self.start_node_innovation_number # begin from start node
        steps = 0
        max_steps = 50 # arbitrary limit to prevent infinite loops
        
        while current_node_innovation_number is not None and steps < max_steps:
            curr_node = self.nodes.get(current_node_innovation_number)
            if not curr_node: 
                break
            chain.append(curr_node)
            
            next_node = None
            for conn in self.connections.values():
                if conn.in_node == current_node_innovation_number and conn.enabled:
                    next_node = conn.out_node
                    break
            
            current_node_innovation_number = next_node
            steps += 1

        if steps == max_steps:
            print("Warning: Max steps reached in get_linear_chain, possible cycle detected.")
            if self.end_node_innovation_number and self.end_node_innovation_number in self.nodes and self.nodes[self.end_node_innovation_number] != chain[-1]:
                chain.append(self.nodes[self.end_node_innovation_number]) # append the end node at the end to ensure it's the last node that will be executed

        return chain

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
    