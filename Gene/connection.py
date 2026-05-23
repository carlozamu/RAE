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
            
    def update_id(self, new_in_node: int = None, new_out_node: int = None):
        """
        Updates the connection ID. Can be called with new nodes to re-route the connection,
        or called with no arguments to just refresh the ID based on current internal state.
        """
        if new_in_node is not None:
            self.in_node = new_in_node
        if new_out_node is not None:
            self.out_node = new_out_node
            
        self.innovation_number = self.generate_connection_id(self.in_node, self.out_node)

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