import hashlib

# Generates a unique innovation hash for a connection between two nodes (used as id for connections)
def generate_connection_hash(start_innovation_number, end_innovation_number):
    unique_str = f"{start_innovation_number}->{end_innovation_number}"
    return hashlib.md5(unique_str.encode()).hexdigest()[:8]

class Connection:
    def __init__(self, input_node_in, output_node_in, innovation_number=None, enabled=True):
        self.in_node = input_node_in
        self.out_node = output_node_in
        self.enabled = enabled
        
        if innovation_number is None:
            self.innovation_number = generate_connection_hash(input_node_in, output_node_in)
        else:
            self.innovation_number = innovation_number
            
    def copy(self):
        """Clone the connection gene"""
        return Connection(
            self.in_node, 
            self.out_node, 
            self.innovation_number, 
            self.enabled
        )
