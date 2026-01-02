from typing import List, Dict, Optional, TYPE_CHECKING

from Edoardo.Fitness.fitness import Fitness
from Edoardo.Genome.agent_genome import AgentGenome
from Edoardo.Phenotype.phenotype import Phenotype

if TYPE_CHECKING:
    from Edoardo.Selection.selection import SelectionStrategy

class Species:

    top_r = 10
    c1 = 1.0  # Coefficient for excess genes
    c2 = 1.0  # Coefficient for disjoint genes
    compatibility_threshold = 3.0  # Threshold for species compatibility

    def __init__(self, members: List[Phenotype], generation: int = 0, selection_strategy: Optional['SelectionStrategy'] = None):
        self.id = None # TODO: decide how to assign species id
        self.generations = []
        self.generation_offset = generation # It's the index of the first global generation at which this species appears
        self.selection_strategy = selection_strategy
        members = [{"member": member, "fitness": Fitness.evaluate(member)} for member in members]
        self.generations.append(members)

    def belongs_to_species(self, candidate: Phenotype, generation: Optional[int]) -> bool:
        """
        Docstring for belongs_to_species

        :param candidate: Phenotype(phenotype needed since fitness will be saveed along with it)
        :param generation: number of generation to compare with, if None the last generation is used
        :return: bool
        """
        generation = generation-self.generation_offset if generation is not None else -1
        population_member = self.generations[generation][0]["member"].genome
        candidate_genome = candidate.genome
        excess_genes = self._count_excess_genes(population_member, candidate_genome)
        disjoint_genes = self._count_disjoint_genes(population_member, candidate_genome)
        max_number_of_genes = max(len(population_member.nodes), len(candidate_genome.nodes))
        compatibility_distance = (self.c1 * (excess_genes/max_number_of_genes) +
                                  self.c2 * (disjoint_genes/max_number_of_genes))
        return compatibility_distance < self.compatibility_threshold

    def add_members(self, members: List[Phenotype], generation: Optional[int]) -> None:
        generation = generation-self.generation_offset if generation is not None else -1
        if not self.generations[generation]:
            self.generations[generation] = []
        for member in members:
            self.generations[generation].append({"member": member, "fitness": Fitness.evaluate(member)})
    
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
        #TODO: can this work with current implementation of nodes with string ids?
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
        #TODO: can this work with current implementation of nodes with string ids?
        member_genes = list(ind1.nodes.keys())
        candidate_genes = list(ind2.nodes.keys())
        member_newest_node_gene = max(member_genes, default=0)
        candidate_newest_node_gene = max(candidate_genes, default=0)
        min_newest_gene = str(min(member_newest_node_gene, candidate_newest_node_gene))
        disjoint_genes = 0
        total_genes = sorted(set(member_genes + candidate_genes))
        for gene in total_genes:
            if gene < min_newest_gene:
                if not (gene in member_genes and gene in candidate_genes):
                    disjoint_genes += 1
        return disjoint_genes

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

    def get_all_members_from_generation(self, generation: Optional[int]) -> List[Phenotype]:
        generation = generation-self.generation_offset if generation is not None else -1
        if 0 <= generation < len(self.generations):
            return [member["member"] for member in self.generations[generation]]
        else:
            return []