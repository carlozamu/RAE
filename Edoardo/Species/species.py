from typing import List, Dict, Optional

from Edoardo.Fitness.fitness import Fitness
from Edoardo.Genome.agent_genome import AgentGenome
from Edoardo.Phenotype.phenotype import Phenotype

class Species:

    top_r = 10
    c1 = 1.0  # Coefficient for excess genes
    c2 = 1.0  # Coefficient for disjoint genes
    c3 = 0.4  # Coefficient for average weight differences
    compatibility_threshold = 3.0  # Threshold for species compatibility

    def __init__(self, members: List[Phenotype], generation: int = 0):
        self.id = None # TODO: decide how to assign species id
        #TODO: check if I'm creating circular dependencies
        self.generations = []
        self.generation_offset = generation
        members = [{"member": member, "fitness": Fitness.evaluate(member)} for member in members]
        self.generations.append(members)

    def belongs_to_species(self, candidate: Phenotype, generation: Optional[int]) -> bool:
        """
        Docstring for belongs_to_species
        
        :param self: Description
        :param candidate: Description
        :type candidate: Phenotype(phenotype needed since fitness will be saveed along with it)
        :param generation: Description
        :type generation: Optional[int]
        :return: Description
        :rtype: bool
        """
        generation = generation-self.generation_offset if generation is not None else -1
        population_member = self.generations[generation][0]["member"]
        excess_genes = self._count_excess_genes(population_member, candidate)
        disjoint_genes = self._count_disjoint_genes(population_member, candidate)
        avg_weight_diff = self._average_weight_difference(population_member, candidate)
        #TODO: change depending on phenotype implementation
        max_number_of_genes = max(len(population_member.nodes), len(candidate.nodes))
        compatibility_distance = (self.c1 * (excess_genes/max_number_of_genes) +
                                  self.c2 * (disjoint_genes/max_number_of_genes) +
                                  self.c3 * avg_weight_diff)
        return compatibility_distance < self.compatibility_threshold

    def add_member(self, member: Phenotype, generation: Optional[int]) -> None:
        generation = generation-self.generation_offset if generation is not None else -1
        if not self.generations[generation]:
            self.generations[generation] = []
        self.generations[generation].append({"member": member, "fitness": Fitness.evaluate(member)})

    def add_new_generation(self, members: List[Phenotype]) -> None:
        self.generations.append([{"member": member, "fitness": Fitness.evaluate(member)} for member in members])

    def get_top_members(self, generation: Optional[int]) -> List[Dict[str, Phenotype | float]]:
        generation = generation-self.generation_offset if generation is not None else -1
        sorted_members = sorted(self.generations[generation], key=lambda x: x["fitness"], reverse=True)
        return sorted_members[:self.top_r]

    def _count_excess_genes(self, ind1: Phenotype, ind2: Phenotype) -> int:
        pass

    def _count_disjoint_genes(self, ind1: Phenotype, ind2: Phenotype) -> int:
        pass

    def _average_weight_difference(self, ind1: Phenotype, ind2: Phenotype) -> float:
        pass

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

    def get_all_members(self, generation: Optional[int]) -> List[Phenotype]:
        generation = generation-self.generation_offset if generation is not None else -1
        return [member["member"] for member in self.generations[generation]]