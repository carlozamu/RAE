import uuid

import numpy as np
from Edoardo.Gene.gene import PromptNode
from Edoardo.Gene.connection import Connection

class AgentGenome:
    def __init__(self, nodes_dict: dict[str, PromptNode] = None, connections_dict: dict[str, Connection] = None, start_node_innovation_number:int=None, end_node_innovation_number:int=None, fitness:float=np.inf):
        self.id = str(uuid.uuid4())
        self.nodes: dict[str, PromptNode] = nodes_dict if nodes_dict is not None else {} # dict of innovation_number: PromptNode
        self.connections: dict[str, Connection] = connections_dict if connections_dict is not None else {} # dict of innovation_number: Connection
        self.start_node_innovation_number = start_node_innovation_number if start_node_innovation_number is not None else None
        self.end_node_innovation_number = end_node_innovation_number if end_node_innovation_number is not None else None
        self.fitness = fitness if fitness is not None else 0.0

    def add_node(self, node: PromptNode):
        self.nodes[node.innovation_number] = node

    def add_connection(self, in_node_innovation_number, out_node_innovation_number):
        # Check for duplicates to prevent multi-edges between same nodes
        for conn in self.connections.values():
            if conn.in_node == in_node_innovation_number and conn.out_node == out_node_innovation_number:
                return # Connection already exists, do nothing

        # Create new connection (Innovation number generated automatically by Connection class)
        new_conn = Connection(in_node_innovation_number, out_node_innovation_number)
        self.connections[new_conn.innovation_number] = new_conn

    def get_execution_order(self) -> list[tuple[PromptNode, list[str]]]:
            """
            Returns a topologically sorted execution plan.
            Each element is a tuple: (The Node to Run, List of Parent Node IDs to get input from).
            
            Example Output:
            [
            (StartNode, []),
            (NodeB, ['start_id']),
            (NodeC, ['B_id']),
            (NodeA, ['start_id', 'C_id'])  <-- Needs inputs from BOTH Start and C
            ]
            """
            if self.start_node_innovation_number is None:
                return []

            # 1. Build Adjacency List (Forward) and Parent Map (Backward)
            adj = {node_id: [] for node_id in self.nodes}
            
            # This will hold the "context sources" for each node
            # parent_map[target_id] = [source_id1, source_id2...]
            parent_map = {node_id: [] for node_id in self.nodes} 
            
            in_degree = {node_id: 0 for node_id in self.nodes}

            # Only consider ENABLED connections
            for conn in self.connections.values():
                if conn.enabled:
                    u, v = conn.in_node, conn.out_node
                    
                    if u in adj and v in adj:
                        adj[u].append(v)
                        in_degree[v] += 1
                        
                        # Track that 'v' receives input from 'u'
                        parent_map[v].append(u)

            # 2. Initialize Queue with Start Node
            queue = [self.start_node_innovation_number]
            
            execution_plan = [] # List of (Node, [Inputs])
            
            # 3. Kahn's Algorithm Loop
            while queue:
                curr_id = queue.pop(0)
                
                if curr_id in self.nodes:
                    node_obj = self.nodes[curr_id]
                    # Retrieve the list of parents for this specific node
                    input_sources = parent_map.get(curr_id, [])
                    
                    # Append the tuple (Node, Inputs)
                    execution_plan.append((node_obj, input_sources))

                for neighbor in adj[curr_id]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)

            # 4. Cycle Detection Check
            has_end = any(item[0].id == self.end_node_innovation_number for item in execution_plan)
            if not has_end and self.end_node_innovation_number in self.nodes:
                print("Warning: End node not reachable or caught in a cycle.")
            
            return execution_plan
    
    def copy(self):
        """
        Deep copies the entire genome.
        This is used when selecting a parent to produce a child.
        """
        new_genome = AgentGenome()
        new_genome.fitness = self.fitness
        
        # Copy all nodes (preserving IDs)
        for node in self.nodes.values():
            new_genome.nodes[node.innovation_number] = node.copy()
            
        # Copy all connections
        for conn in self.connections.values():
            new_genome.connections[conn.innovation_number] = conn.copy()
            
        return new_genome
    
    def detect_cycle_edges(self) -> list[str]:
        """
        Uses Kahn's algorithm (topological sort) to detect edges involved in cycles.
        Returns list of connection IDs that are part of cycles.
        If the graph is a DAG, returns empty list.
        """
        # Build adjacency list and calculate in-degrees (only enabled connections)
        adjacency_list = {node_id: [] for node_id in self.nodes.keys()}
        in_degree = {node_id: 0 for node_id in self.nodes.keys()}
        edge_to_target = {}  # Maps edge_id -> target node for tracking
        
        for edge_id, conn in self.connections.items():
            if conn.enabled:
                adjacency_list[conn.in_node].append((conn.out_node, edge_id))
                in_degree[conn.out_node] += 1
                edge_to_target[edge_id] = conn.out_node
        
        # Kahn's algorithm: Start with nodes that have in-degree 0
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        processed_count = 0
        
        while queue:
            current = queue.pop(0)
            processed_count += 1
            
            # Process all outgoing edges
            for neighbor, edge_id in adjacency_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If not all nodes processed, there's a cycle
        if processed_count < len(self.nodes):
            # Collect edges that are part of the cycle
            # Only edges where BOTH endpoints are in the cycle are actually cycle edges
            # (not edges coming from outside the cycle into it)
            cycle_edges = []
            nodes_in_cycle = {node_id for node_id, degree in in_degree.items() if degree > 0}
            
            for edge_id, conn in self.connections.items():
                if conn.enabled and conn.in_node in nodes_in_cycle and conn.out_node in nodes_in_cycle:
                    cycle_edges.append(edge_id)
            
            return cycle_edges
        
        return []  # No cycles detected
    
    def score_edge_for_removal(self, edge_id: str) -> float:
        """
        Scores an edge for removal priority. Lower scores = safer to remove.
        
        Scoring criteria:
        - High penalty for edges connected to start/end nodes
        - High penalty for edges that would isolate nodes (only in/out connection)
        - Lower penalty for edges from nodes with multiple outputs
        """
        conn = self.connections[edge_id]
        score = 0.0
        
        # PENALTY: Connected to start node (critical for execution)
        if conn.in_node == self.start_node_innovation_number:
            score += 1000.0
        
        # PENALTY: Connected to end node (critical for execution)
        if conn.out_node == self.end_node_innovation_number:
            score += 1000.0
        
        # Calculate out-degree of source node (enabled connections only)
        out_degree = sum(1 for c in self.connections.values() 
                        if c.in_node == conn.in_node and c.enabled)
        
        # PENALTY: Only output from this node (would isolate it)
        if out_degree == 1:
            score += 500.0
        else:
            # BONUS: More alternative outputs = safer to remove this one
            score -= (out_degree - 1) * 10.0
        
        # Calculate in-degree of target node (enabled connections only)
        in_degree = sum(1 for c in self.connections.values() 
                       if c.out_node == conn.out_node and c.enabled)
        
        # PENALTY: Only input to target node (would isolate it)
        if in_degree == 1:
            score += 500.0
        else:
            # BONUS: More alternative inputs = safer to remove this one
            score -= (in_degree - 1) * 10.0
        
        return score
    
    def verify_all_paths_lead_to_end(self) -> bool:
        """
        Returns True only if EVERY branch/path starting from the start_node
        eventually reaches the end_node. 
        Returns False if there are any dead ends.
        """
        if self.start_node_innovation_number is None or self.end_node_innovation_number is None:
            print("Warning: Start or End node not defined in genome.")
            return True

        # 1. Build Adjacency Lists (Forward and Backward)
        adj = {node_id: [] for node_id in self.nodes.keys()}
        rev_adj = {node_id: [] for node_id in self.nodes.keys()}
        
        for conn in self.connections.values():
            if conn.enabled:
                adj[conn.in_node].append(conn.out_node)
                rev_adj[conn.out_node].append(conn.in_node)

        # 2. Find all nodes reachable from START (Forward BFS)
        reachable_from_start = set()
        queue = [self.start_node_innovation_number]
        reachable_from_start.add(self.start_node_innovation_number)
        while queue:
            curr = queue.pop(0)
            for neighbor in adj.get(curr, []):
                if neighbor not in reachable_from_start:
                    reachable_from_start.add(neighbor)
                    queue.append(neighbor)

        # 3. Find all nodes that can reach END (Backward BFS)
        can_reach_end = set()
        queue = [self.end_node_innovation_number]
        can_reach_end.add(self.end_node_innovation_number)
        while queue:
            curr = queue.pop(0)
            for neighbor in rev_adj.get(curr, []):
                if neighbor not in can_reach_end:
                    can_reach_end.add(neighbor)
                    queue.append(neighbor)

        # 4. Final Validation: 
        # Every node that the start node can reach MUST be able to reach the end node.
        # If a node is in 'reachable_from_start' but NOT in 'can_reach_end', it's a dead end.
        for node in reachable_from_start:
            if node not in can_reach_end:
                # Exception: The end node itself doesn't need to reach 'the end'
                if node != self.end_node_innovation_number:
                    return False 

        return True
    
    def remove_cycles(self) -> int:
        """
        Greedy algorithm to remove cycles by disabling edges one at a time.
        Uses Kahn's algorithm to detect cycles and scores edges for safe removal.
        Ensures end node remains reachable from start after each removal.
        
        Returns the number of edges disabled.
        """
        edges_disabled = 0
        max_iterations = len(self.connections)  # Safety limit
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            
            # Detect cycle edges using Kahn's algorithm
            cycle_edges = self.detect_cycle_edges()
            
            if not cycle_edges:
                # No more cycles, we're done
                break
            
            # Score all edges involved in cycles
            scored_edges = [(self.score_edge_for_removal(edge_id), edge_id) 
                          for edge_id in cycle_edges]
            
            # Sort by score (lowest = safest to remove)
            scored_edges.sort()
            
            # Try to remove edges in order until we find one that preserves connectivity
            edge_removed = False
            for score, edge_id in scored_edges:
                # Temporarily disable the edge
                self.connections[edge_id].enabled = False
                
                # Check if end is still reachable
                if self.verify_all_paths_lead_to_end():
                    # Good! Keep it disabled and continue
                    edges_disabled += 1
                    edge_removed = True
                    break
                else:
                    # Bad! Re-enable this edge and try next one
                    self.connections[edge_id].enabled = True
            
            if not edge_removed:
                # Couldn't find any safe edge to remove
                print(f"Warning: Could not remove cycle without breaking connectivity. "
                      f"{len(cycle_edges)} edges remain in cycles.")
                break
        
        if iteration >= max_iterations:
            print(f"Warning: Max iterations reached in remove_cycles.")
        
        return edges_disabled
    