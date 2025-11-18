import uuid
import hashlib
import copy # Needed for deep copying

class NodeType:
    START = "START"
    MIDDLE = "MIDDLE"
    END = "END"
    ROUTER = "ROUTER"

class EmbeddingEngine:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from sentence_transformers import SentenceTransformer
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

    @staticmethod
    def get_embedding(text):
        model = EmbeddingEngine.get_instance()
        # FIX: Convert numpy array to list for easier serialization/storage
        return model.encode(text).tolist() 

def generate_innovation_hash(input_id, output_id):
    unique_str = f"{input_id}->{output_id}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:8]

class PromptNode:
    # FIX: Added node_id=None to allow preserving identity during cloning
    def __init__(self, type: str, name: str, instruction: str, node_id=None):
        # 1. Identity
        self.id = node_id if node_id is not None else str(uuid.uuid4())
        
        # 2. Structural Type
        self.type = type
        
        # 3. The "DNA"
        self.name = name
        self._instruction = instruction 
        
        # 4. Speciation Data
        self.embedding = []
        self._update_embedding() 

    @property
    def instruction(self):
        return self._instruction

    @instruction.setter
    def instruction(self, new_instruction):
        if new_instruction != self._instruction:
            self._instruction = new_instruction
            self._update_embedding()

    def _update_embedding(self):
        try:
            self.embedding = EmbeddingEngine.get_embedding(self._instruction)
        except Exception as e:
            print(f"Warning: Could not compute embedding. {e}")
            self.embedding = []

    def copy(self):
        """
        Creates a clone of the node WITH THE SAME ID.
        Essential for crossover/reproduction.
        """
        # FIX: Pass self.id to the constructor
        new_node = PromptNode(self.type, self.name, self.instruction, node_id=self.id)
        new_node.embedding = self.embedding.copy() # Optimization: don't re-compute
        return new_node

class ConnectionGene:
    def __init__(self, input_node_id, output_node_id, innovation_number=None, enabled=True):
        self.in_node = input_node_id
        self.out_node = output_node_id
        self.enabled = enabled # FIX: Allow setting enabled state in init
        
        if innovation_number is None:
            self.innovation_number = generate_innovation_hash(input_node_id, output_node_id)
        else:
            self.innovation_number = innovation_number
            
    def copy(self):
        """Clone the connection gene"""
        return ConnectionGene(
            self.in_node, 
            self.out_node, 
            self.innovation_number, 
            self.enabled
        )

class AgentGenome:
    def __init__(self):
        self.nodes = {}       
        self.connections = [] 
        self.start_node_id = None 
        self.fitness = None  
        self.adjusted_fitness = None 

    def add_node(self, node: PromptNode):
        self.nodes[node.id] = node

    def add_connection(self, in_node_id, out_node_id):
        for conn in self.connections:
            if conn.in_node == in_node_id and conn.out_node == out_node_id:
                return 
        new_conn = ConnectionGene(in_node_id, out_node_id)
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
    