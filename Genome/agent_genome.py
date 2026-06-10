import uuid
import math
from collections import deque
from Gene.gene import PromptNode
from Gene.connection import Connection

class AgentGenome:
    def __init__(self, nodes_dict: dict[int, PromptNode] = None, connections_dict: dict[str, Connection] = None, start_node_innovation_number:int=None, end_node_innovation_number:int=None, fitness:float=math.inf):
        self.id = str(uuid.uuid4())
        self.nodes: dict[int, PromptNode] = nodes_dict if nodes_dict is not None else {} # dict of innovation_number: PromptNode
        self.connections: dict[str, Connection] = connections_dict if connections_dict is not None else {} # dict of innovation_number: Connection
        self.start_node_innovation_number = start_node_innovation_number if start_node_innovation_number is not None else None
        self.end_node_innovation_number = end_node_innovation_number if end_node_innovation_number is not None else None
        self.fitness = fitness if fitness is not None else 0.0
        self.accuracy = 0.0
        self.avg_tokens = 0.0
        self.evaluated = False

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

    def get_execution_order(self) -> list[tuple[PromptNode, list[int]]]:
        """
        Returns a topologically sorted execution plan.
        """
        if not self.nodes or self.start_node_innovation_number is None:
            return []

        # Optimization: Instant return for Single Node Genomes (Start == End)
        if self.start_node_innovation_number == self.end_node_innovation_number:
            start_node = self.nodes.get(self.start_node_innovation_number)
            if start_node:
                # Single node has no parents
                return [(start_node, [])]
            else:
                print(f"Critical Error: Start Node {self.start_node_innovation_number} missing.")
                return []

        # 1. Build Adjacency List and Parent Map
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

        # 2. Initialize Queue (The Fix)
        # Kahn's algorithm MUST start with ALL nodes that have zero incoming dependencies,
        # otherwise deadlocks occur when orphan nodes exist.
        queue = [node_inn for node_inn, degree in in_degree.items() if degree == 0]
        
        # Optional but recommended: Force the designated Start Node to the front of the queue 
        # so it processes first.
        if self.start_node_innovation_number in queue:
            queue.remove(self.start_node_innovation_number)
            queue.insert(0, self.start_node_innovation_number)
            
        execution_plan: list[tuple[PromptNode, list[int]]] = []
        
        # 3. Kahn's Algorithm Loop
        while queue:
            curr_id = queue.pop(0)
            
            if curr_id not in self.nodes:
                continue

            node_obj = self.nodes[curr_id]
            input_sources: list[int] = parent_map.get(curr_id, [])
            execution_plan.append((node_obj, input_sources))

            for neighbor in adj[curr_id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        has_end = any(item[0].innovation_number == self.end_node_innovation_number for item in execution_plan)
        
        if not has_end and self.end_node_innovation_number in self.nodes:
             print(f"Warning: End node {self.end_node_innovation_number} is unreachable from Start.")
             print("")
        
        return execution_plan

    def copy(self):
        """
        Deep copies the entire genome.
        This is used when selecting a parent to produce a child.
        """
        new_genome = AgentGenome()
        new_genome.fitness = self.fitness
        new_genome.accuracy = self.accuracy
        new_genome.avg_tokens = self.avg_tokens
        new_genome.start_node_innovation_number = self.start_node_innovation_number
        new_genome.end_node_innovation_number = self.end_node_innovation_number
        new_genome.evaluated = self.evaluated
        
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
        node_keys = list(self.nodes.keys())
        # Build adjacency list and calculate in-degrees (only enabled connections)
        adjacency_list = {node_id: [] for node_id in node_keys}
        in_degree = {node_id: 0 for node_id in node_keys}
        edge_to_target = {}  # Maps edge_id -> target node for tracking
        
        for edge_id, conn in list(self.connections.items()):
            if conn.enabled:
                if conn.in_node in node_keys and conn.out_node in node_keys:
                    adjacency_list[conn.in_node].append((conn.out_node, edge_id))
                    in_degree[conn.out_node] += 1
                    edge_to_target[edge_id] = conn.out_node
        
        # Kahn's algorithm: Start with nodes that have in-degree 0
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        if len(queue) > 1:
            print(f"Warning: Multiple start nodes detected in cycle detection. Queue: {queue}")
            # Add a connection from the start node to the extra start node
            for extra_start in queue[1:]:
                new_conn = Connection(self.start_node_innovation_number, extra_start)
                self.connections[new_conn.innovation_number] = new_conn
                adjacency_list[self.start_node_innovation_number].append((extra_start, new_conn.innovation_number))
                in_degree[extra_start] += 1
                edge_to_target[new_conn.innovation_number] = extra_start
            queue = [self.start_node_innovation_number]  # Reset queue to start node only
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
        Strict structural integrity check: O(V + E) Time, O(V) Space.
        Ensures 100% of nodes are reachable from Start (No Orphans) 
        AND 100% of nodes can reach End (No Dead Ends).
        """
        if self.start_node_innovation_number is None or self.end_node_innovation_number is None:
            return False
            
        start_id = self.start_node_innovation_number
        end_id = self.end_node_innovation_number
        
        all_node_ids = set(self.nodes.keys())
        expected_count = len(all_node_ids)
        
        # 1. Single-Pass Adjacency Building
        # We pre-allocate arrays to avoid dictionary key lookups during the BFS
        forward_adj = {node_id: [] for node_id in all_node_ids}
        backward_adj = {node_id: [] for node_id in all_node_ids}
        
        for conn in self.connections.values():
            if conn.enabled and conn.in_node in all_node_ids and conn.out_node in all_node_ids:
                forward_adj[conn.in_node].append(conn.out_node)
                backward_adj[conn.out_node].append(conn.in_node)
                
        # 2. Forward BFS (Detects Orphans)
        forward_visited = {start_id}
        queue = deque([start_id]) # deque guarantees O(1) popleft
        
        while queue:
            curr = queue.popleft()
            for neighbor in forward_adj[curr]:
                if neighbor not in forward_visited:
                    forward_visited.add(neighbor)
                    queue.append(neighbor)
                    
        # EARLY EXIT: If Start cannot reach every node, the graph is already invalid.
        # We skip the second search entirely, saving 50% of the compute time.
        if len(forward_visited) != expected_count:
            return False
            
        # 3. Backward BFS (Detects Dead Ends)
        backward_visited = {end_id}
        queue = deque([end_id])
        
        while queue:
            curr = queue.popleft()
            for neighbor in backward_adj[curr]:
                if neighbor not in backward_visited:
                    backward_visited.add(neighbor)
                    queue.append(neighbor)
                    
        # 4. Final Validation
        # If the backward search also reached exactly every node, the graph is perfect.
        return len(backward_visited) == expected_count
    
    def remove_cycles(self) -> int:
        """
        Aggressively removes ALL cycles from the genome.
        If removing cycles breaks connectivity, fixes connectivity afterward
        by adding edges that don't create cycles.
        
        Returns the number of edges disabled.
        """
        edges_disabled = 0
        
        # Phase 1: Aggressively remove all cycles
        # We keep removing the safest cycle edge until the graph is a DAG.
        # We do NOT check connectivity here — breaking connectivity is allowed.
        for _ in range(10000):  # Absolute safety limit
            cycle_edges = self.detect_cycle_edges()
            if not cycle_edges:
                break
                
            # Score all edges involved in cycles
            scored_edges = [(self.score_edge_for_removal(edge_id), edge_id) for edge_id in cycle_edges]
            scored_edges.sort()
            
            # Remove the safest edge (lowest score = safest to remove)
            score, edge_id = scored_edges[0]
            self.connections[edge_id].enabled = False
            edges_disabled += 1
        else:
            # If we hit the limit, verify whether cycles actually remain
            remaining = self.detect_cycle_edges()
            if remaining:
                print(f"Warning: Could not remove all cycles after max iterations. "
                    f"{len(remaining)} cycle edges may remain.")
        
        # Phase 2: Fix connectivity if broken
        # After the graph is guaranteed cycle-free, we restore connectivity
        # by adding edges that respect the DAG order.
        if not self.verify_all_paths_lead_to_end():
            self._fix_connectivity_without_cycles()
        
        return edges_disabled

    def _fix_connectivity_without_cycles(self):
        """
        Fixes connectivity in a DAG by adding edges.
        Ensures all nodes are reachable from start and can reach end.
        Only adds edges that don't create cycles.
        """
        if self.start_node_innovation_number is None or self.end_node_innovation_number is None:
            return
        
        if self.start_node_innovation_number == self.end_node_innovation_number:
            return
        
        execution_plan = self.get_execution_order()
        if not execution_plan:
            return
        
        topo_order = [node.innovation_number for node, _ in execution_plan]
        
        # Build adjacency lists
        forward_adj = {node_id: [] for node_id in self.nodes}
        backward_adj = {node_id: [] for node_id in self.nodes}
        for conn in self.connections.values():
            if conn.enabled and conn.in_node in self.nodes and conn.out_node in self.nodes:
                forward_adj[conn.in_node].append(conn.out_node)
                backward_adj[conn.out_node].append(conn.in_node)
        
        # --- Forward pass: Ensure all nodes reachable from start ---
        if self.start_node_innovation_number in self.nodes:
            reachable = {self.start_node_innovation_number}
            queue = deque([self.start_node_innovation_number])
            while queue:
                curr = queue.popleft()
                for neighbor in forward_adj[curr]:
                    if neighbor not in reachable:
                        reachable.add(neighbor)
                        queue.append(neighbor)
            
            for node_id in topo_order:
                if node_id not in reachable:
                    # Try to connect from the earliest reachable node that comes before it
                    for candidate in topo_order:
                        if candidate in reachable and self._safe_add_edge(candidate, node_id):
                            if node_id not in forward_adj[candidate]:
                                forward_adj[candidate].append(node_id)
                            if candidate not in backward_adj[node_id]:
                                backward_adj[node_id].append(candidate)
                            
                            # BFS to update reachable set
                            subqueue = deque([node_id])
                            while subqueue:
                                curr = subqueue.popleft()
                                if curr not in reachable:
                                    reachable.add(curr)
                                for neighbor in forward_adj[curr]:
                                    if neighbor not in reachable:
                                        reachable.add(neighbor)
                                        subqueue.append(neighbor)
                            break
                    else:
                        print(f"Warning: Could not connect node {node_id} to start without creating cycles.")
        
        # --- Backward pass: Ensure all nodes can reach end ---
        if self.end_node_innovation_number in self.nodes:
            can_reach_end = {self.end_node_innovation_number}
            queue = deque([self.end_node_innovation_number])
            while queue:
                curr = queue.popleft()
                for neighbor in backward_adj[curr]:
                    if neighbor not in can_reach_end:
                        can_reach_end.add(neighbor)
                        queue.append(neighbor)
            
            for node_id in reversed(topo_order):
                if node_id not in can_reach_end:
                    # Try to connect to the latest node that can reach end and comes after it
                    for candidate in reversed(topo_order):
                        if candidate in can_reach_end and self._safe_add_edge(node_id, candidate):
                            if candidate not in forward_adj[node_id]:
                                forward_adj[node_id].append(candidate)
                            if node_id not in backward_adj[candidate]:
                                backward_adj[candidate].append(node_id)
                            
                            # Backward BFS to update can_reach_end
                            subqueue = deque([node_id])
                            while subqueue:
                                curr = subqueue.popleft()
                                if curr not in can_reach_end:
                                    can_reach_end.add(curr)
                                for neighbor in backward_adj[curr]:
                                    if neighbor not in can_reach_end:
                                        can_reach_end.add(neighbor)
                                        subqueue.append(neighbor)
                            break
                    else:
                        print(f"Warning: Could not connect node {node_id} to end without creating cycles.")

    def _safe_add_edge(self, from_node, to_node) -> bool:
        """
        Safely adds or enables an edge from from_node to to_node.
        Returns True if the edge exists and is enabled without creating cycles,
        False otherwise.
        """
        # Check if edge already exists
        existing_edge_id = None
        for edge_id, conn in self.connections.items():
            if conn.in_node == from_node and conn.out_node == to_node:
                existing_edge_id = edge_id
                break
        
        if existing_edge_id is not None:
            conn = self.connections[existing_edge_id]
            if conn.enabled:
                return True
            
            # Try enabling existing edge
            conn.enabled = True
            if not self._has_cycle():
                return True
            
            # Cycle created, disable again
            conn.enabled = False
            return False
        
        # Add new edge
        self.add_connection(from_node, to_node)
        
        # Find the newly added edge
        new_edge_id = None
        for edge_id, conn in self.connections.items():
            if conn.in_node == from_node and conn.out_node == to_node:
                new_edge_id = edge_id
                break
        
        if new_edge_id is None:
            return False
        
        if not self._has_cycle():
            return True
        
        # Cycle created, remove the edge we just added
        del self.connections[new_edge_id]
        return False

    def _has_cycle(self) -> bool:
        """
        Returns True if the enabled subgraph contains a cycle.
        Uses Kahn's algorithm without modifying the graph.
        """
        if not self.nodes:
            return False
        
        node_keys = list(self.nodes.keys())
        in_degree = {node_id: 0 for node_id in node_keys}
        adj = {node_id: [] for node_id in node_keys}
        
        for conn in self.connections.values():
            if conn.enabled and conn.in_node in node_keys and conn.out_node in node_keys:
                adj[conn.in_node].append(conn.out_node)
                in_degree[conn.out_node] += 1
        
        queue = deque([node_id for node_id, degree in in_degree.items() if degree == 0])
        processed = 0
        
        while queue:
            curr = queue.popleft()
            processed += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return processed < len(node_keys)
    