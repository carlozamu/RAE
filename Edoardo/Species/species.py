from typing import List, Any, Dict

from Edoardo.Fitness.fitness import Fitness
from Edoardo.Phenotype.phenotype import Phenotype


class Species:
    top_r = 10
    def __init__(self, members: List[Phenotype]):
        self.id = None # Todo: decide how to assign species id
        #TODO: check if I'm creating circular dependencies
        self.members = [{"members": member, "fitness": Fitness.evaluate(member)} for member in members]

    def belongs_to_species(self, candidate: Phenotype) -> bool:
        pass

    def add_member(self, member: Phenotype) -> None:
        self.members.append(member)

    def get_top_members(self) -> List[Dict[float, Phenotype]]:
        sorted_members = sorted(self.members, key=lambda x: x["fitness"], reverse=True)
        return sorted_members[:self.top_r]