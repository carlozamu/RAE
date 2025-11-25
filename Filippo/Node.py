import uuid

# EmbeddingEngine is a singleton class to manage text embeddings
class EmbeddingEngine:
    _instance = None
    
    # this method ensures only one instance of the embedding model is created
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            from sentence_transformers import SentenceTransformer
            cls._instance = SentenceTransformer('all-MiniLM-L6-v2')
        return cls._instance

    # this method computes the embedding for a given text
    @staticmethod
    def get_embedding(text):
        model = EmbeddingEngine.get_instance()
        # FIX: Convert numpy array to list for easier serialization/storage
        return model.encode(text).tolist() 

# PromptNode represents a node in the architecture
class PromptNode:
    """
    This class represents a node in the architecture.
    """
    def __init__(self, type: str, name: str, instruction: str, node_id=None, innovation_number:int=None):
        # 1. Identity
        self.id = node_id if node_id is not None else str(uuid.uuid4())

        # 2. Innovation Number
        self.innovation_number = innovation_number if innovation_number is not None else self.id
        
        # 3. The "DNA"
        # the name of the node, is intended to briefly represent the prompt main task 
        #for easy identification for debugging and for the LLm that will need to perform action on them
        self.name = name 
        self._instruction = instruction 
        
        # 4. Speciation Data
        self.embedding = [] # embedding vector for the prompt instruction
        self._update_embedding() 

    @property
    def instruction(self):
        return self._instruction

    # handles change in instruction and updates embedding accordingly
    @instruction.setter
    def instruction(self, new_instruction):
        if new_instruction != self._instruction:
            self._instruction = new_instruction
            self._update_embedding()

    # function to update the embedding of a prompt node when the instruction changes
    def _update_embedding(self):
        try:
            self.embedding = EmbeddingEngine.get_embedding(self._instruction)
        except Exception as e:
            print(f"Warning: Could not compute embedding. {e}")
            self.embedding = []

    # function to create a clone of the node
    def copy(self):
        """
        Creates a clone of the node WITH THE SAME ID.
        Essential for crossover/reproduction.
        """
        new_node = PromptNode(self.type, self.name, self.instruction, node_id=self.id)
        new_node.embedding = self.embedding.copy()
        return new_node

  