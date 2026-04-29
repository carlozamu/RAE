import numpy as np
import random
from typing import List
from Genome.agent_genome import AgentGenome

class Species:
    # NEAT Distance Coefficients
    c1 = 1.0  # Excess nodes weight
    c2 = 1.0  # Disjoint nodes weight
    c3 = 1.0  # Edge count difference weight
    c4 = 1.0  # Semantic Prompt difference weight

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
        self.members = [] # Clear the bucket for the new generation
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
        Calculates the topological and semantic distance between a candidate 
        and this species' representative using the NEAT formula.
        """
        excess_genes = self._count_excess_genes(self.representative, candidate)
        disjoint_genes = self._count_disjoint_genes(self.representative, candidate)
        
        rep_active_edges = [e for e in self.representative.connections.values() if e.enabled]
        cand_active_edges = [e for e in candidate.connections.values() if e.enabled]
        
        max_nodes = max(len(self.representative.nodes), len(candidate.nodes), 1)
        max_edges = max(len(rep_active_edges), len(cand_active_edges), 1)
        
        different_edges = self._count_different_edges(self.representative, candidate)
        avg_weight_diff = self._compute_average_weight_difference(self.representative, candidate)

        distance = (
            (self.c1 * excess_genes / max_nodes) +
            (self.c2 * disjoint_genes / max_nodes) +
            (self.c3 * different_edges / max_edges) +
            (self.c4 * avg_weight_diff)
        )
        return distance
    
    def _count_excess_genes(self, ind1: AgentGenome, ind2: AgentGenome) -> int:
        member_genes = list(ind1.nodes.keys())
        candidate_genes = list(ind2.nodes.keys())
        member_newest_node_gene = max(member_genes, default=0)
        candidate_newest_node_gene = max(candidate_genes, default=0)
        min_newest_gene = min(int(member_newest_node_gene), int(candidate_newest_node_gene))
        excess_genes = 0
        total_genes = sorted(set(member_genes + candidate_genes))
        for gene in total_genes:
            if int(gene) > min_newest_gene:
                if not (gene in member_genes and gene in candidate_genes):
                    excess_genes += 1
        return excess_genes

    def _count_disjoint_genes(self, ind1: AgentGenome, ind2: AgentGenome) -> int:
        member_genes = list(ind1.nodes.keys())
        candidate_genes = list(ind2.nodes.keys())
        member_newest_node_gene = max(member_genes, default=0)
        candidate_newest_node_gene = max(candidate_genes, default=0)
        min_newest_gene = min(int(member_newest_node_gene), int(candidate_newest_node_gene))
        disjoint_genes = 0
        different_genes = list(set(member_genes).difference(set(candidate_genes)).union(set(candidate_genes).difference(set(member_genes))))
        for gene in different_genes:
            if int(gene) < min_newest_gene:
                disjoint_genes += 1
        return disjoint_genes
    
    def _count_different_edges(self, ind1: AgentGenome, ind2: AgentGenome) -> int:
        ind1_edges = set([edge for edge in ind1.connections.keys() if ind1.connections[edge].enabled])
        ind2_edges = set([edge for edge in ind2.connections.keys() if ind2.connections[edge].enabled])
        unique_to_1 = ind1_edges.difference(ind2_edges)
        unique_to_2 = ind2_edges.difference(ind1_edges)
        total_diff = len(unique_to_1) + len(unique_to_2)
        
        return total_diff
    
    def _compute_average_weight_difference(self, ind1: AgentGenome, ind2: AgentGenome) -> float:
        """
        Calculates the average semantic distance (Cosine Distance) between matching nodes.
        Used for the 'Weight Difference' component of the NEAT compatibility formula.
        """
        # 1. Efficiently find common nodes (Intersection)
        # Set operations are O(min(len(s1), len(s2))) vs O(N) loop
        common_ids = set(ind1.nodes.keys()) & set(ind2.nodes.keys())
        
        if not common_ids:
            return 0.0

        total_distance = 0.0
        valid_comparisons = 0
        
        for node_id in common_ids:
            emb1 = ind1.nodes[node_id].embedding
            emb2 = ind2.nodes[node_id].embedding
            if not emb1 or not emb2:
                continue

            # 2. Convert to Numpy
            v1 = np.array(emb1)
            v2 = np.array(emb2)

            # 3. Compute Cosine Distance (1 - Cosine Similarity), normalized to [0, 1]
            # Formula: (1 - (A . B) / (||A|| * ||B||)) / 2
            # This returns 0.0 for identical vectors, and 1.0 for opposite vectors.
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            
            if norm1 == 0 or norm2 == 0:
                dist = 0.5 # Neutral penalty if a vector is null/zero
            else:
                similarity = np.dot(v1, v2) / (norm1 * norm2)
                dist = (1.0 - max(-1.0, min(1.0, similarity))) / 2.0
            
            total_distance += dist
            valid_comparisons += 1

        return total_distance / valid_comparisons if valid_comparisons > 0 else 0.0
