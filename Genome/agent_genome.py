import uuid
import math
import random
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

    def add_node(self, node: PromptNode):
        self.nodes[node.innovation_number] = node
    
    def add_connection(self, in_node_innovation_number: int, out_node_innovation_number: int) -> Connection:
        # Check for duplicates to prevent multi-edges between same nodes
        conn_id = Connection.generate_connection_id(in_node_innovation_number, out_node_innovation_number)
        if conn_id in self.connections:
            return self.connections[conn_id]

        new_conn = Connection(in_node_innovation_number, out_node_innovation_number)
        self.connections[new_conn.innovation_number] = new_conn
        return new_conn

    def remove_connection(self, innovation_number: str):
        """Safely removes a connection by its string ID key."""
        if innovation_number in self.connections:
            del self.connections[innovation_number]

    def remove_node_safely(self, node_innovation_number: int) -> dict[str, Connection]:
        """
        Removes a node and purges all related connections out of the master dictionary.
        Returns a dictionary of the removed connections to support rollback transactions.
        """
        removed_connections = {}
        if node_innovation_number not in self.nodes:
            return removed_connections

        # Find all keys bound structurally to this node ID
        conns_to_remove = [
            c_id for c_id, conn in self.connections.items()
            if conn.in_node == node_innovation_number or conn.out_node == node_innovation_number
        ]

        for c_id in conns_to_remove:
            removed_connections[c_id] = self.connections.pop(c_id)
            
        self.nodes.pop(node_innovation_number)
        return removed_connections

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