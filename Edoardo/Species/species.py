from typing import List, Dict, Optional, TYPE_CHECKING, Any, Union
import hashlib

import numpy as np

from Fitness.fitness import Fitness
from Genome.agent_genome import AgentGenome
from Phenotype.phenotype import Phenotype

from Selection.selection import SelectionStrategy

class Species:

    top_r = 10
    c1 = 1.0  # Coefficient for excess genes
    c2 = 1.0  # Coefficient for disjoint genes
    c3 = 1.0  # Coefficient for edges count difference
    c4 = 1.0  # Coefficient for average node differences
    protection_base = 3  # Base number of generations for protection
    adjust_rate_protected_species = 1.5  # Adjustment rate for protected species
    compatibility_threshold = 2.0  # Threshold for species compatibility
    species_id_counter = 0

    def __init__(self, initial_members: List[Phenotype], generation: int = 0, selection_strategy: Optional['SelectionStrategy'] = None, max_hof_size: int = 10):
        self.id = Species.species_id_counter
        Species.species_id_counter += 1
        self.generations: List[List[Dict[str, Union[Phenotype, float]]]] = []
        self.generation_offset = generation # It's the index of the first global generation at which this species appears
        self.selection_strategy = selection_strategy
        self.max_hof_size = max_hof_size
        self.hall_of_fame: List[Dict[str, Any]] = []
        self.representative = initial_members[0] # Keep a persistent representative

        
        # We assume members already have their fitness evaluated before being added to species
        members: List[Dict[str, Union[Phenotype, float]]] = [{"member": member, "fitness": member.genome.fitness} for member in initial_members]
        self.generations.append(members)

    def update_hall_of_fame(self) -> None:
        """
        Updates the Hall of Fame for this species with the best individuals from all generations.
        Maintains unique individuals based on genome signature.
        """
        # Collect all members from history
        all_members: List[Dict[str, Any]] = []
        for gen_members in self.generations:
            all_members.extend(gen_members)
            
        # Process candidates
        # We want to merge existing HoF with historical population
        # But efficiently: usually we just need to check the NEWEST generation against current HoF
        # For robustness, we'll check everything but filter duplicates
        
        current_pool = self.hall_of_fame + all_members
        
        unique_best = {} # Signature -> MemberDict
        
        for member_dict in current_pool:
            sig = self._get_individual_signature(member_dict)
            
            # If we haven't seen this signature, or this instance has better fitness (unlikely if immutable, but good practice)
            if sig not in unique_best:
                unique_best[sig] = member_dict
            elif member_dict["fitness"] > unique_best[sig]["fitness"]:
                unique_best[sig] = member_dict
                
        # Convert back to list and sort
        sorted_best = sorted(unique_best.values(), key=lambda x: x["fitness"], reverse=False)
        
        # Keep top N
        self.hall_of_fame = sorted_best[:self.max_hof_size]

    @staticmethod
    def _get_individual_signature(individual: Dict[str, Any]) -> str:
        """
        Generate a unique signature for an individual based on its genome structure.
        """
        genome = individual['member'].genome
        # Signature based on nodes and connections to identify topological duplicates
        nodes_sig = ",".join(sorted(str(k) for k in genome.nodes.keys()))
        conns_sig = ",".join(sorted([f"{c.in_node}->{c.out_node}" for c in genome.connections.values() if c.enabled]))
        
        raw_sig = f"N:[{nodes_sig}]|C:[{conns_sig}]"
        return hashlib.md5(raw_sig.encode()).hexdigest()
    
    def adjusted_offspring_count(self, average_fitness: float, generation: int) -> float:
        """
        Calculate adjusted offspring count for the species based on average fitness.
        
        :param average_fitness: Average fitness of the entire population
        :param generation: Current generation index
        :return: Adjusted number of offspring for this species
        """
        print(f"Calculating adjusted offspring for Species {self.id} at Gen {generation}, with average fitness {average_fitness:.4f}")
        species_age = generation - self.generation_offset
        
        # Note: cumulative_fitness and member_count expect global generation index
        cumulative_fit = self.cumulative_fitness(generation)
        member_count = self.member_count(generation)
        
        if average_fitness > 0:
            adjusted_count = (1.0 / cumulative_fit) / (1.0 / average_fitness)
        else:
            adjusted_count = float(member_count)
            
        # Calculate average complexity (nodes + connections)
        if 0 <= species_age < len(self.generations):
            current_members = self.generations[species_age]
            if current_members:
                total_complexity = sum(
                    len(m["member"].genome.nodes) + len(list(e for e in m["member"].genome.connections.keys() if m["member"].genome.connections[e].enabled))
                    for m in current_members
                )
                avg_complexity = total_complexity / max(1, len(current_members))
            else:
                avg_complexity = 0.0
        else:
            avg_complexity = 0.0

        # Protection mechanism for young and complex species
        # Base protection of 5 generations, plus complexity factor
        protection_limit = Species.protection_base + (avg_complexity * 0.5)
        
        if species_age < protection_limit:
            # Boost adjusted count to prevent premature extinction
            adjusted_count *= Species.adjust_rate_protected_species
            # Ensure at least one offspring
            if adjusted_count < 1.0:
                adjusted_count = 1.0
        
        return adjusted_count
        
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


    def get_compatibility_distance(self, genome1: AgentGenome, genome2: AgentGenome) -> float:
        """
        Calculates the compatibility distance between two genomes.
        Formula: δ = c1*(E/N) + c2*(D/N) + c3*(ΔEdges/M) + c4*ΔWeight
        
        :param genome1: First genome to compare
        :param genome2: Second genome to compare
        :return: The calculated distance (float)
        """
        # 1. Gene Computations (Nodes)
        excess_genes = self._count_excess_genes(genome1, genome2)
        disjoint_genes = self._count_disjoint_genes(genome1, genome2)
        
        # Normalizer for genes (N) -> Standard NEAT uses 1 if genome size < 20, but max() is safer for large networks
        max_number_of_genes = max(len(genome1.nodes), len(genome2.nodes), 1)

        # 2. Weight/Embedding Difference (Semantic distance)
        average_weight_diff = self._compute_average_weight_difference(genome1, genome2)

        # 3. Structural Edge Differences (Topology distance)
        # Note: We filter for ENABLED edges usually, but checking all keys is also valid depending on implementation.
        # Your previous code checked enabled edges for the normalizer but keys for difference. 
        # Here we align them to be consistent (using enabled edges for both is standard).
        
        g1_enabled = [k for k, v in genome1.connections.items() if v.enabled]
        g2_enabled = [k for k, v in genome2.connections.items() if v.enabled]
        
        # Recalculating edge difference here locally to ensure consistency with the enabled list
        unique_to_1 = set(g1_enabled) - set(g2_enabled)
        unique_to_2 = set(g2_enabled) - set(g1_enabled)
        different_edges = len(unique_to_1) + len(unique_to_2)
        
        # Normalizer for edges
        max_number_of_edges = max(len(g1_enabled), len(g2_enabled), 1)

        # 4. Final Calculation
        distance = (
            Species.c1 * (excess_genes / max_number_of_genes) +
            Species.c2 * (disjoint_genes / max_number_of_genes) +
            Species.c3 * (different_edges / max_number_of_edges) +
            Species.c4 * average_weight_diff
        )
        
        return distance

    def belongs_to_species(self, candidate: Phenotype, generation: Optional[int] = None) -> bool:
        """
        Checks if a candidate belongs to this species by comparing it 
        against the species' representative.
        """
        # We ignore the 'generation' param here because strict NEAT compares against 
        # a fixed representative for the specific generation cycle.
        
        if self.representative is None:
            # Fallback for safety, though representative should always exist
            return True
            
        dist = self.get_compatibility_distance(self.representative.genome, candidate.genome)
        return dist < self.compatibility_threshold

    def add_members(self, members: List[Phenotype], generation: Optional[int]) -> None:
        """
        Adds new members to the species storage for the specific generation.
        IMPORTANT: Does NOT update the representative. That happens only once per generation via elect_new_representative.
        """
        gen_idx = generation - self.generation_offset if generation is not None else -1
        
        # Handle case where we need to extend the generations list
        if gen_idx >= 0:
            while len(self.generations) <= gen_idx:
                self.generations.append([])
            target_list = self.generations[gen_idx]
        else:
            # -1 case (append to last)
            if not self.generations:
                self.generations.append([])
            target_list = self.generations[-1]

        for member in members:
            target_list.append({"member": member, "fitness": member.genome.fitness})

        # NOTE: No self.representative update here!

    def member_count(self, generation: Optional[int]) -> int:
        """Safe member count that returns 0 if generation doesn't exist yet."""
        if generation is None:
            return len(self.generations[-1]) if self.generations else 0
            
        gen_idx = generation - self.generation_offset
        if 0 <= gen_idx < len(self.generations):
            return len(self.generations[gen_idx])
        return 0

    def cumulative_fitness(self, generation: Optional[int]) -> float:
        """Safe fitness sum."""
        if generation is None:
            target_list = self.generations[-1] if self.generations else []
        else:
            gen_idx = generation - self.generation_offset
            if 0 <= gen_idx < len(self.generations):
                target_list = self.generations[gen_idx]
            else:
                target_list = []
        return sum(float(member["fitness"]) for member in target_list)

    def get_all_members_from_generation(self, generation: Optional[int]) -> List[Dict[str, Union[Phenotype, float]]]:
        """Safe member retrieval."""
        if generation is None:
            return self.generations[-1] if self.generations else []

        gen_idx = generation - self.generation_offset
        if 0 <= gen_idx < len(self.generations):
            return self.generations[gen_idx]
        return []

    def elect_new_representative(self) -> None:
        """Picks a random survivor to represent the species for the NEXT generation."""
        import random
        if self.generations and self.generations[-1]:
            new_rep = random.choice(self.generations[-1])
            self.representative = new_rep['member']

        # Pick a random survivor from the current generation to represent the species next time
        new_rep_dict = random.choice(self.generations[-1])
        self.representative = new_rep_dict['member']
    
    def get_top_members(self, generation: Optional[int] = None) -> List[Dict[str, Phenotype | float]]:
        """
        Get top members from the species using selection strategy if available, otherwise by fitness sorting.
        
        :param generation: Generation index to get members from (None for last generation)
        :return: List of top members as dicts with 'member' and 'fitness' keys
        """
        generation = generation-self.generation_offset if generation is not None else -1
        population = self.generations[generation]
        
        if not population:
            return []
        
        # Use selection strategy if provided, otherwise fall back to sorting
        if self.selection_strategy is not None:
            # Use selection strategy to select top_r members
            selected = self.selection_strategy.select(population, num_parents=min(self.top_r, len(population)))
            return selected
        else:
            # Fallback to original behavior: sort by fitness
            sorted_members = sorted(population, key=lambda x: x["fitness"], reverse=False)
            return sorted_members[:self.top_r]

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

    def first_generation_index(self) -> int:
        return self.generation_offset

    def last_generation_index(self) -> int:
        return self.generation_offset + len(self.generations) - 1
