from Filippo.Node import PromptNode
from Filippo.Connection import Connection

class AgentGenome:
    def __init__(self):
        self.nodes = {} # dict of node_id: PromptNode    
        self.connections = [] # list of Connection objects
        self.start_node_id = None 
        self.end_node_id = None
        self.fitness = None  
        self.adjusted_fitness = None 


    #TODO: continue from here
    def add_node(self, node: PromptNode):
        self.nodes[node.id] = node

    def add_connection(self, in_node_id, out_node_id):
        for conn in self.connections:
            if conn.in_node == in_node_id and conn.out_node == out_node_id:
                return 
        new_conn = Connection(in_node_id, out_node_id)
        self.connections.append(new_conn)

    def get_linear_chain(self):
        chain = []
        if self.start_node_id is None:
            return chain
            
        current_node_id = self.start_node_id
        steps = 0
        max_steps = 50 
        
        while current_node_id is not None and steps < max_steps:
            curr_node = self.nodes.get(current_node_id)
            if not curr_node: 
                break
            chain.append(curr_node)
            
            next_id = None
            for conn in self.connections:
                if conn.in_node == current_node_id and conn.enabled:
                    next_id = conn.out_node
                    break
            
            current_node_id = next_id
            steps += 1
        return chain

    def copy(self):
        """
        Deep copies the entire genome.
        This is used when selecting a parent to produce a child.
        """
        new_genome = AgentGenome()
        new_genome.start_node_id = self.start_node_id
        new_genome.fitness = self.fitness
        
        # Copy all nodes (preserving IDs)
        for node_id, node in self.nodes.items():
            new_genome.nodes[node_id] = node.copy()
            
        # Copy all connections
        for conn in self.connections:
            new_genome.connections.append(conn.copy())
            
        return new_genome
    