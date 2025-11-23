from typing import List, Any

from Edoardo.Fitness.fitness import Fitness


class Species:
    top_r = 10
    def __init__(self, members: List[Any]):
        self.id = None # Todo: decide how to assign species id
        #TODO: check if I'm creating circular dependencies
        self.members = [{"members": member, "fitness": Fitness.evaluate(member)} for member in members]

    def belongs_to_species(self, candidate: Any) -> bool:
        pass

    def add_member(self, member: Any) -> None:
        self.members.append(member)

    def get_top_members(self):
        sorted_members = sorted(self.members, key=lambda x: x["fitness"], reverse=True)
        return sorted_members[:self.top_r]