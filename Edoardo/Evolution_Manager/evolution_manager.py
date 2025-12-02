from Edoardo.Species.species import Species
import random


class EvolutionManager:
    def __init__(self, num_parents: int = 2):
        self.current_generation_index = 0
        self.species = []
        self.num_parents = num_parents

    def create_new_generation(self):
        #TODO: Is there a better way to calculate total average fitness?
        total_fitness = 0
        total_individuals = 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                total_fitness += species.cumulative_fitness()
                total_individuals += species.member_count()
        average_fitness = total_fitness / total_individuals if total_individuals > 0 else 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                new_species_count = int(species.cumulative_fitness() / average_fitness if average_fitness > 0 else species.member_count())
                for _ in range(new_species_count):
                    self.create_offspring(species)
        pass

    def get_latest_generation(self):
        members = []
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                members.extend(species.get_all_members())
        return None

    def create_offspring(self, species):
        parent_pool = species.get_top_members()
        healthy_child = False
        child = None
        while not healthy_child:
            parents = self._select_parents(parent_pool, self.num_parents)
            #TODO: get individual
            new_species = True
            for s in self.species:
                # TODO: this loop means that a new species can be populated by only one individual, is that ok?
                if s.belongs_to_species(child) and s != species:
                    healthy_child = False
                    new_species = False
                    break
                elif s.belongs_to_species(child) and s == species:
                    healthy_child = True
                    new_species = False
                    break
            if new_species:
                healthy_child = True
                self.species.append(Species([child], generation=self.current_generation_index+1))
        return child

    @staticmethod
    def _select_parents(parent_pool, num_parents):
        """randomly select num_parents elements from the parent_pool list"""
        return random.sample(parent_pool, num_parents)