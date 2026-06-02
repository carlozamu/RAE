import numpy as np
import random
import math
from typing import List
from Genome.agent_genome import AgentGenome

class Species:
    def __init__(self, representative: AgentGenome, species_id: int):
        self.id = species_id
        self.representative = representative
        self.members: List[AgentGenome] = [representative]
        
        # Stagnation tracking
        self.age = 0
        self.generations_without_improvement = 0
        self.max_fitness_ever = representative.fitness
        self.alive = True

    def add_member(self, genome: AgentGenome):
        self.members.append(genome)

    def get_average_fitness(self) -> float:
        """Calculates the shared fitness of the niche."""
        if not self.members:
            return 0.0
        return sum(m.fitness for m in self.members) / len(self.members)
    
    def get_avg_accuracy(self) -> float:
        """Calculates the average accuracy of the species."""
        if not self.members:
            return 0.0
        return sum(m.accuracy for m in self.members) / len(self.members)

    def update_representative(self):
        """Picks a random member from this generation to represent the next."""
        if self.members and self.alive:
            self.representative = random.choice(self.members)
            self.age += 1
        self.members = []  # Reset the member list

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

    def compatibility_distance(self, other_genome: AgentGenome) -> float:
        """
        Calculates distance using Jaccard Similarity of Innovation Numbers.
        Returns a strict float between 1.0 (Identical) and 0.0 (Completely Alien).
        """
        # Get sets of innovation numbers for both nodes and connections
        self_nodes = set(self.representative.nodes.keys())
        other_nodes = set(other_genome.nodes.keys())
        
        # Jaccard Math
        shared_genes = self_nodes.intersection(other_nodes)
        max_nodes = max(len(self_nodes), len(other_nodes))

        if max_nodes == 0:
            return 0.0 # Both are totally empty graphs
            
        similarity = len(shared_genes) / max_nodes
        
        return similarity
    
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