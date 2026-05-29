class Connection:
    
    @staticmethod
    def generate_connection_id(in_node: int, out_node: int) -> str:
        """
        Generates a semantic string ID (e.g., '3.12').
        Uses string formatting intead of float to mathematically prevent the 3.1 == 3.10 collision bug.
        """
        return f"{in_node}.{out_node}"
    
    def __init__(self, input_node_in: int, output_node_in: int, enabled: bool = True):
        self.in_node: int = input_node_in
        self.out_node: int = output_node_in
        self.enabled: bool = enabled
        
        # Automatically generates the ID upon creation based on the provided nodes
        self.innovation_number: str = self.generate_connection_id(self.in_node, self.out_node)
            
    def copy(self):
        """
        Clone the connection gene. 
        Notice we don't need to pass the ID anymore, as __init__ builds it automatically.
        """
        return Connection(
            input_node_in=self.in_node, 
            output_node_in=self.out_node, 
            enabled=self.enabled
        )