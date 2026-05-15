import numpy as np
import random
import math
from typing import List
from Genome.agent_genome import AgentGenome

class Species:
    # Asymptotic Distance Coefficients
    c_nodes = 0.15   # Penalty per unmatched node
    c_edges = 0.15   # Penalty per unmatched connection
    c_weight = 1.0   # Penalty for semantic drift (Cosine distance)

    def __init__(self, representative: AgentGenome, species_id: int):
        self.id = species_id
        self.representative = representative
        self.members: List[AgentGenome] = [representative]
        
        # Stagnation tracking
        self.age = 0
        self.generations_without_improvement = 0
        self.max_fitness_ever = representative.fitness

    def add_member(self, genome: AgentGenome):
        self.members.append(genome)

    def get_average_fitness(self) -> float:
        """Calculates the shared fitness of the niche."""
        if not self.members:
            return 0.0
        return sum(m.fitness for m in self.members) / len(self.members)

    def update_representative(self):
        """Picks a random member from this generation to represent the next."""
        if self.members:
            self.representative = random.choice(self.members)
        self.members = [] 
        self.age += 1

    def update_stagnation(self):
        """Tracks if the species has stopped improving."""
        if not self.members:
            return
            
        current_max = max(m.fitness for m in self.members)
        if current_max > self.max_fitness_ever:
            self.max_fitness_ever = current_max
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1

    def compatibility_distance(self, candidate: AgentGenome) -> float:
        """
        Calculates the distance using a Bounded Asymptotic Curve (tanh).
        Returns a strict value between 0.0 (Identical) and 10.0 (Alien).
        """
        # 1. Absolute Topological Node Difference (Symmetric Difference)
        rep_nodes = set(self.representative.nodes.keys())
        cand_nodes = set(candidate.nodes.keys())
        unmatched_nodes_count = len(rep_nodes.symmetric_difference(cand_nodes))
        
        # 2. Absolute Topological Edge Difference (Symmetric Difference)
        rep_edges = set(e for e, conn in self.representative.connections.items() if conn.enabled)
        cand_edges = set(e for e, conn in candidate.connections.items() if conn.enabled)
        unmatched_edges_count = len(rep_edges.symmetric_difference(cand_edges))
        
        # 3. Average Semantic Difference (for nodes that share the same innovation ID)
        avg_weight_diff = self._compute_average_weight_difference(self.representative, candidate)

        # 4. Compute Linear Raw Distance
        raw_distance = (
            (self.c_nodes * unmatched_nodes_count) +
            (self.c_edges * unmatched_edges_count) +
            (self.c_weight * avg_weight_diff)
        )
        
        # 5. Apply Asymptotic Squashing (Tanh) mapped to 0.0 -> 10.0
        bounded_distance = 10.0 * math.tanh(raw_distance)
        
        return bounded_distance
    
    def _compute_average_weight_difference(self, ind1: AgentGenome, ind2: AgentGenome) -> float:
        """
        Calculates the average semantic distance (Cosine Distance) between matching nodes.
        Value ranges from 0.0 (identical thoughts) to 1.0 (opposite thoughts).
        """
        common_ids = set(ind1.nodes.keys()) & set(ind2.nodes.keys())
        
        if not common_ids:
            return 0.0 # Topology penalty will handle entirely disjoint graphs

        total_distance = 0.0
        valid_comparisons = 0
        
        for node_id in common_ids:
            emb1 = ind1.nodes[node_id].embedding
            emb2 = ind2.nodes[node_id].embedding
            
            if not emb1 or not emb2:
                continue

            v1 = np.array(emb1)
            v2 = np.array(emb2)

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                dist = 0.5 
            else:
                similarity = np.dot(v1, v2) / (norm1 * norm2)
                # Ensure float inaccuracies don't push similarity outside [-1, 1]
                dist = (1.0 - max(-1.0, min(1.0, similarity))) / 2.0
            
            total_distance += dist
            valid_comparisons += 1

        return total_distance / valid_comparisons if valid_comparisons > 0 else 0.0