import uuid

class PromptNode:
    """
    Represents a node in the reasoning chain.
    """
    def __init__(self, name: str, instruction: str, embedding=None, node_id=None, innovation_number=None):
        # 1. Identity
        self.id = node_id if node_id is not None else str(uuid.uuid4())
        self.innovation_number = innovation_number if innovation_number is not None else self.id
        
        # 2. Structure & DNA
        # We removed 'type' as requested, relying on Genome start/end pointers
        self.name = name 
        self.instruction = instruction # Just a plain variable now
        
        # 3. Speciation Data
        self.embedding = embedding if embedding is not None else []

    def copy(self):
        """
        Creates a deep clone with a NEW ID (default behavior for mutation).
        """
        return PromptNode(
            #TODO: Use deepcopy maybe(copy only does a shallow copy, it might not be enough), as of now the two objects point at the same attributes
            name=self.name, 
            instruction=self.instruction, 
            # We copy the list to ensure the new node has its own memory space
            embedding=self.embedding.copy() if isinstance(self.embedding, list) else [],
            node_id=str(uuid.uuid4()), 
            innovation_number=self.innovation_number
        )