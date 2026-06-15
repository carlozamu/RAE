import uuid
import math
from collections import deque
from Gene.gene import PromptNode
from Gene.connection import Connection

class AgentGenome:
    def __init__(self, nodes_dict: dict[int, PromptNode] = None, connections_dict: dict[str, Connection] = None, start_node_innovation_number: int = None, end_node_innovation_number: int = None, fitness: float = math.inf):
        self.id = str(uuid.uuid4())
        self.nodes: dict[int, PromptNode] = nodes_dict if nodes_dict is not None else {} 
        self.connections: dict[str, Connection] = connections_dict if connections_dict is not None else {} 
        self.start_node_innovation_number = start_node_innovation_number
        self.end_node_innovation_number = end_node_innovation_number
        self.fitness = fitness if fitness is not None else 0.0
        self.accuracy = 0.0
        self.avg_tokens = 0.0
        self.evaluated = False

    def _would_create_cycle(self, in_node: int, out_node: int) -> bool:
        """
        Performs a BFS to determine if adding a path from in_node to out_node creates a cycle.
        If out_node can already reach in_node, adding the connection forms a loop.
        """
        if in_node == out_node:
            return True
            
        visited = set()
        queue = deque([out_node])
        
        while queue:
            curr = queue.popleft()
            if curr == in_node:
                return True
            if curr not in visited:
                visited.add(curr)
                # Find all nodes the current node connects TO
                for conn in self.connections.values():
                    if conn.enabled and conn.in_node == curr:
                        queue.append(conn.out_node)
        return False

    def add_node_safely(self, node: PromptNode, target_connection_id: str) -> bool:
        """
        Atomically adds a node by splitting an existing connection. 
        This guarantees the new node is instantly integrated into a valid path, preventing orphans.
        """
        if target_connection_id not in self.connections:
            return False
            
        target_conn = self.connections[target_connection_id]
        
        # 1. Register the new node
        self.nodes[node.innovation_number] = node
        
        # 2. Purge the old connection
        self.connections.pop(target_connection_id)
        
        # 3. Wire the new node in the middle (A -> New -> B)
        conn_in_id = Connection.generate_connection_id(target_conn.in_node, node.innovation_number)
        conn_out_id = Connection.generate_connection_id(node.innovation_number, target_conn.out_node)
        
        new_conn_in = Connection(target_conn.in_node, node.innovation_number)
        new_conn_out = Connection(node.innovation_number, target_conn.out_node)
        
        self.connections[conn_in_id] = new_conn_in
        self.connections[conn_out_id] = new_conn_out
        
        return True
    
    def add_connection_safely(self, in_node_innovation_number: int, out_node_innovation_number: int) -> Connection:
        """Adds a connection only if it preserves DAG integrity and boundary rules."""
        # 1. Boundary Enforcement
        if in_node_innovation_number == self.end_node_innovation_number:
            return None # End node cannot have outgoing connections
        if out_node_innovation_number == self.start_node_innovation_number:
            return None # Start node cannot have incoming connections
            
        # 2. Cycle Enforcement
        if self._would_create_cycle(in_node_innovation_number, out_node_innovation_number):
            return None

        # 3. Duplicate Enforcement
        conn_id = Connection.generate_connection_id(in_node_innovation_number, out_node_innovation_number)
        if conn_id in self.connections:
            return self.connections[conn_id]

        new_conn = Connection(in_node_innovation_number, out_node_innovation_number)
        self.connections[conn_id] = new_conn
        return new_conn

    def remove_connection_safely(self, innovation_number: str) -> bool:
        """
        Transactional removal. Reverts if the removal breaks reachability.
        Returns True if successful, False if the connection was a critical load-bearing path.
        """
        if innovation_number not in self.connections:
            return False
            
        # Temporarily remove
        removed_conn = self.connections.pop(innovation_number)
        
        # Verify Graph state
        if not self.verify_all_paths_lead_to_end():
            # Revert transaction
            self.connections[innovation_number] = removed_conn
            return False
            
        return True

    def remove_node_safely(self, node_innovation_number: int) -> dict:
        """
        Removes a node and reroutes connections to prevent topological fragmentation.
        Guards against boundary node deletion.
        """
        transaction = {
            "removed_connections": {},
            "added_connections": {},
            "removed_node": None
        }

        if node_innovation_number not in self.nodes:
            return transaction
            
        # Strict boundary guard
        if node_innovation_number in (self.start_node_innovation_number, self.end_node_innovation_number):
            return transaction

        incoming_conns = {}
        outgoing_conns = {}
        for c_id, conn in self.connections.items():
            if conn.out_node == node_innovation_number: 
                incoming_conns[c_id] = conn
            elif conn.in_node == node_innovation_number: 
                outgoing_conns[c_id] = conn

        source_nodes = set(conn.in_node for conn in incoming_conns.values())
        target_nodes = set(conn.out_node for conn in outgoing_conns.values())

        for c_id in list(incoming_conns.keys()) + list(outgoing_conns.keys()):
            transaction["removed_connections"][c_id] = self.connections.pop(c_id)

        transaction["removed_node"] = self.nodes.pop(node_innovation_number)

        # Reroute: Bridge every source directly to every target
        for s_id in source_nodes:
            for t_id in target_nodes:
                if s_id == t_id:
                    continue

                connection_exists = any(
                    c.in_node == s_id and c.out_node == t_id 
                    for c in self.connections.values()
                )

                if not connection_exists:
                    new_c_id = Connection.generate_connection_id(s_id, t_id) 
                    new_conn = Connection(s_id, t_id) 
                    
                    self.connections[new_c_id] = new_conn
                    transaction["added_connections"][new_c_id] = new_conn

        return transaction

    def get_execution_order(self) -> list[tuple[PromptNode, list[int]]]:
        if not self.nodes or self.start_node_innovation_number is None:
            return []

        if self.start_node_innovation_number == self.end_node_innovation_number:
            start_node = self.nodes.get(self.start_node_innovation_number)
            return [(start_node, [])] if start_node else []

        adj = {node_inn: [] for node_inn in self.nodes}
        parent_map = {node_inn: [] for node_inn in self.nodes} 
        in_degree = {node_inn: 0 for node_inn in self.nodes}

        for conn in self.connections.values():
            if conn.enabled:
                u, v = conn.in_node, conn.out_node
                if u in adj and v in adj:
                    adj[u].append(v)
                    in_degree[v] += 1
                    parent_map[v].append(u)

        queue = [node_inn for node_inn, degree in in_degree.items() if degree == 0]
        
        if self.start_node_innovation_number in queue:
            queue.remove(self.start_node_innovation_number)
            queue.insert(0, self.start_node_innovation_number)
            
        execution_plan: list[tuple[PromptNode, list[int]]] = []
        
        while queue:
            curr_id = queue.pop(0)
            if curr_id not in self.nodes:
                continue

            execution_plan.append((self.nodes[curr_id], parent_map.get(curr_id, [])))

            for neighbor in adj[curr_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return execution_plan

    def verify_all_paths_lead_to_end(self) -> bool:
        if self.start_node_innovation_number is None or self.end_node_innovation_number is None:
            return False
            
        start_id = self.start_node_innovation_number
        end_id = self.end_node_innovation_number
        all_node_ids = set(self.nodes.keys())
        expected_count = len(all_node_ids)
        
        forward_adj = {node_id: [] for node_id in all_node_ids}
        backward_adj = {node_id: [] for node_id in all_node_ids}
        
        for conn in self.connections.values():
            if conn.enabled and conn.in_node in all_node_ids and conn.out_node in all_node_ids:
                forward_adj[conn.in_node].append(conn.out_node)
                backward_adj[conn.out_node].append(conn.in_node)
                
        # Forward BFS: Check for orphans
        forward_visited = {start_id}
        queue = deque([start_id])
        while queue:
            curr = queue.popleft()
            for neighbor in forward_adj[curr]:
                if neighbor not in forward_visited:
                    forward_visited.add(neighbor)
                    queue.append(neighbor)
                    
        if len(forward_visited) != expected_count:
            return False
            
        # Backward BFS: Check for dead ends
        backward_visited = {end_id}
        queue = deque([end_id])
        while queue:
            curr = queue.popleft()
            for neighbor in backward_adj[curr]:
                if neighbor not in backward_visited:
                    backward_visited.add(neighbor)
                    queue.append(neighbor)
                    
        return len(backward_visited) == expected_count

    def copy(self):
        new_genome = AgentGenome()
        new_genome.fitness = self.fitness
        new_genome.accuracy = self.accuracy
        new_genome.avg_tokens = self.avg_tokens
        new_genome.start_node_innovation_number = self.start_node_innovation_number
        new_genome.end_node_innovation_number = self.end_node_innovation_number
        new_genome.evaluated = self.evaluated
        
        for node in self.nodes.values():
            new_genome.nodes[node.innovation_number] = node.copy()
            
        for conn in self.connections.values():
            new_genome.connections[conn.innovation_number] = conn.copy()
            
        return new_genome
    