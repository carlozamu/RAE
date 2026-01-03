from typing import List, Dict, Optional, TYPE_CHECKING, Any, Union
import hashlib

import numpy as np

from Fitness.fitness import Fitness
from Genome.agent_genome import AgentGenome
from Phenotype.phenotype import Phenotype

if TYPE_CHECKING:
    from Selection.selection import SelectionStrategy

class Species:

    top_r = 10
    c1 = 1.0  # Coefficient for excess genes
    c2 = 1.0  # Coefficient for disjoint genes
    c3 = 0.4  # Coefficient for edges count difference
    c4 = 0.2  # Coefficient for average node differences
    protection_base = 3  # Base number of generations for protection
    adjust_rate_protected_species = 1.5  # Adjustment rate for protected species
    compatibility_threshold = 3.0  # Threshold for species compatibility
    species_id_counter = 0

    def __init__(self, initial_members: List[Phenotype], generation: int = 0, selection_strategy: Optional['SelectionStrategy'] = None, max_hof_size: int = 10):
        self.id = Species.species_id_counter
        Species.species_id_counter += 1
        self.generations = []
        self.generation_offset = generation # It's the index of the first global generation at which this species appears
        self.selection_strategy = selection_strategy
        self.max_hof_size = max_hof_size
        self.hall_of_fame: List[Dict[str, Any]] = []
        
        # We assume members already have their fitness evaluated before being added to species
        members: List[Dict[str, Union[Phenotype, float]]] = [{"member": member, "fitness": member.genome.fitness} for member in initial_members]
        self.generations.append(members)

    def update_hall_of_fame(self) -> None:
        """
        Updates the Hall of Fame for this species with the best individuals from all generations.
        Maintains unique individuals based on genome signature.
        """
        # Collect all members from history
        all_members = []
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
        sorted_best = sorted(unique_best.values(), key=lambda x: x["fitness"], reverse=True)
        
        # Keep top N
        self.hall_of_fame = sorted_best[:self.max_hof_size]

    @staticmethod
    def _get_individual_signature(individual: Dict[str, Any]) -> str:
        """
        Generate a unique signature for an individual based on its genome structure.
        """
        genome = individual['member'].genome
        # Signature based on nodes and connections to identify topological duplicates
        nodes_sig = ",".join(sorted(genome.nodes.keys()))
        conns_sig = ",".join(sorted([f"{c.in_node}->{c.out_node}" for c in genome.connections.values() if c.enabled]))
        
        raw_sig = f"N:[{nodes_sig}]|C:[{conns_sig}]"
        return hashlib.md5(raw_sig.encode()).hexdigest()
    
    def adjusted_offspring_count(self, average_fitness: float, generation: int) -> int:
        """
        Calculate adjusted offspring count for the species based on average fitness.
        
        :param average_fitness: Average fitness of the entire population
        :param generation: Current generation index
        :return: Adjusted number of offspring for this species
        """
        species_age = generation - self.generation_offset
        
        # Note: cumulative_fitness and member_count expect global generation index
        cumulative_fit = self.cumulative_fitness(generation)
        member_count = self.member_count(generation)
        
        if average_fitness > 0:
            adjusted_count = cumulative_fit / average_fitness
        else:
            adjusted_count = float(member_count)
            
        # Calculate average complexity (nodes + connections)
        if 0 <= species_age < len(self.generations):
            current_members = self.generations[species_age]
            if current_members:
                total_complexity = sum(
                    len(m["member"].genome.nodes) + len(m["member"].genome.connections)
                    for m in current_members
                )
                avg_complexity = total_complexity / len(current_members)
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
        
        return int(adjusted_count)
    
    def _compute_average_weight_difference(self, ind1: AgentGenome, ind2: AgentGenome) -> float:
        node_list = list(ind1.nodes.keys()) + list(ind2.nodes.keys())
        node_list = list(set(node_list))
        common_nodes = 0
        total_diff = 0.0
        for node_innovation_number in node_list:
            if node_innovation_number in ind1.nodes and node_innovation_number in ind2.nodes:
                common_nodes += 1
                total_diff += np.sum((np.array(ind1.nodes[node_innovation_number].embedding) - 
                       np.array(ind2.nodes[node_innovation_number].embedding))**2)
        return total_diff / common_nodes if common_nodes > 0 else 0.0


    def belongs_to_species(self, candidate: Phenotype, generation: Optional[int]) -> bool:
        """
        Docstring for belongs_to_species

        :param candidate: Phenotype(phenotype needed since fitness will be saveed along with it)
        :param generation: number of generation to compare with, if None the last generation is used
        :return: bool
        """
        generation = generation-self.generation_offset if generation is not None else -1
        population_member: Phenotype = self.generations[generation][0]["member"]
        candidate_genome = candidate.genome
        excess_genes = self._count_excess_genes(population_member.genome, candidate_genome)
        disjoint_genes = self._count_disjoint_genes(population_member.genome, candidate_genome)
        max_number_of_genes = max(len(population_member.genome.nodes), len(candidate_genome.nodes))
        average_weight_diff = self._compute_average_weight_difference(population_member.genome, candidate.genome)
        different_edges = self._count_different_edges(population_member.genome, candidate_genome)
        compatibility_distance = (Species.c1 * (excess_genes/max_number_of_genes) +
                                  Species.c2 * (disjoint_genes/max_number_of_genes) +
                                  Species.c3 * (different_edges/max_number_of_genes) +
                                  Species.c4 * average_weight_diff)
        return compatibility_distance < self.compatibility_threshold

    def add_members(self, members: List[Phenotype], generation: Optional[int]) -> None:
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
            sorted_members = sorted(population, key=lambda x: x["fitness"], reverse=True)
            return sorted_members[:self.top_r]

    def _count_excess_genes(self, ind1: AgentGenome, ind2: AgentGenome) -> int:
        member_genes = list(ind1.nodes.keys())
        candidate_genes = list(ind2.nodes.keys())
        member_newest_node_gene = max(member_genes, default=0)
        candidate_newest_node_gene = max(candidate_genes, default=0)
        min_newest_gene = str(min(member_newest_node_gene, candidate_newest_node_gene))
        excess_genes = 0
        total_genes = sorted(set(member_genes + candidate_genes))
        for gene in total_genes:
            if gene > min_newest_gene:
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
        ind1_edges = list(ind1.connections.keys())
        ind2_edges = list(ind2.connections.keys())
        different_edges = list(set(ind1_edges).difference(set(ind2_edges)).union(set(ind2_edges).difference(set(ind1_edges))))
        return len(different_edges)

    def cumulative_fitness(self, generation: Optional[int]) -> float:
        generation = generation-self.generation_offset if generation is not None else -1
        return sum(member["fitness"] for member in self.generations[generation])

    def member_count(self, generation: Optional[int]) -> int:
        generation = generation-self.generation_offset if generation is not None else -1
        return len(self.generations[generation])

    def first_generation_index(self) -> int:
        return self.generation_offset

    def last_generation_index(self) -> int:
        return self.generation_offset + len(self.generations) - 1

    def get_all_members_from_generation(self, generation: Optional[int]) -> List[Dict[str, Union[Phenotype, float]]]:
        generation = generation-self.generation_offset if generation is not None else -1
        if 0 <= generation < len(self.generations):
            return self.generations[generation]
        else:
            return []