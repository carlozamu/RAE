from Edoardo.Gene.gene import PromptNode
from Edoardo.Gene.connection import Connection

class AgentGenome:
    def __init__(self):
        self.family_id = None  # Optional: to track lineage
        self.nodes: dict[str, PromptNode] = {} # dict of node_id: PromptNode
        self.connections: dict[str, Connection] = {} # dict of innovation_number: Connection
        self.start_node_id = None 
        self.end_node_id = None
        self.fitness = None #TODO: do we need fitness here?
        self.adjusted_fitness = None 

    def add_node(self, node: PromptNode):
        #TODO: if two similar individuals add a node in the same position, are they going to get the same id and innovation number on the same connection?
        self.nodes[node.id] = node

    def add_connection(self, in_node_id, out_node_id):
        # Check for duplicates to prevent multi-edges between same nodes
        for conn in self.connections.values():
            if conn.in_node == in_node_id and conn.out_node == out_node_id:
                return # Connection already exists, do nothing

        #TODO: if i create the same connection in two different istances will the innovation number be the same?

        # Create new connection (Innovation number generated automatically by Connection class)
        new_conn = Connection(in_node_id, out_node_id)
        self.connections[new_conn.innovation_number] = new_conn

    def get_linear_chain(self) -> list[PromptNode]:
        """
        Retrieves the linear chain of nodes starting from start_node_id and following enabled connections.
        It's needed to practically execute the agent's prompt sequence.
        """
        chain = []
        if self.start_node_id is None:
            return chain # return empty if no start node defined
            
        current_node_id = self.start_node_id # begin from start node
        steps = 0
        max_steps = 50 # arbitrary limit to prevent infinite loops
        
        while current_node_id is not None and steps < max_steps:
            curr_node = self.nodes.get(current_node_id)
            if not curr_node: 
                break
            chain.append(curr_node)
            
            next_id = None
            for conn in self.connections.values():
                if conn.in_node == current_node_id and conn.enabled:
                    next_id = conn.out_node
                    break
            
            current_node_id = next_id
            steps += 1

        if steps == max_steps:
            print("Warning: Max steps reached in get_linear_chain, possible cycle detected.")
            if self.end_node_id and self.end_node_id in self.nodes and self.nodes[self.end_node_id] != chain[-1]:
                chain.append(self.nodes[self.end_node_id]) # append the end node at the end to ensure it's the last node that will be executed

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
            new_node = node.copy()
            new_genome.nodes[new_node.id] = new_node
            if node.id == self.start_node_id:
                new_genome.start_node_id = new_node.id
            if node.id == self.end_node_id:
                new_genome.end_node_id = new_node.id
            
        # Copy all connections
        for conn in self.connections.values():
            new_genome.connections[conn.innovation_number] = conn.copy()
            
        return new_genome
    