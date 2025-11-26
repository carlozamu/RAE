from typing import List, Any, Dict

from Edoardo.Fitness.fitness import Fitness
from Edoardo.Phenotype.phenotype import Phenotype


class Species:
    top_r = 10
    c1 = 1.0  # Coefficient for excess genes
    c2 = 1.0  # Coefficient for disjoint genes
    c3 = 0.4  # Coefficient for average weight differences
    compatibility_threshold = 3.0  # Threshold for species compatibility
    def __init__(self, members: List[Phenotype]):
        self.id = None # Todo: decide how to assign species id
        #TODO: check if I'm creating circular dependencies
        self.members = [{"members": member, "fitness": Fitness.evaluate(member)} for member in members]

    def belongs_to_species(self, candidate: Phenotype) -> bool:
        population_member = self.members[0]["members"]
        excess_genes = self._count_excess_genes(population_member, candidate)
        disjoiont_genes = self._count_disjoint_genes(population_member, candidate)
        avg_weight_diff = self._average_weight_difference(population_member, candidate)
        max_number_of_genes = max(len(population_member.nodes), len(candidate.nodes))
        compatibility_distance = (self.c1 * (excess_genes/max_number_of_genes) +
                                  self.c2 * (disjoiont_genes/max_number_of_genes) +
                                  self.c3 * avg_weight_diff)
        return compatibility_distance < self.compatibility_threshold

    def add_member(self, member: Phenotype) -> None:
        self.members.append(member)

    def get_top_members(self) -> List[Dict[float, Phenotype]]:
        sorted_members = sorted(self.members, key=lambda x: x["fitness"], reverse=True)
        return sorted_members[:self.top_r]

    def _count_excess_genes(self, ind1: Phenotype, ind2: Phenotype) -> int:
        pass

    def _count_disjoint_genes(self, ind1: Phenotype, ind2: Phenotype) -> int:
        pass

    def _average_weight_difference(self, ind1: Phenotype, ind2: Phenotype) -> float:
        pass