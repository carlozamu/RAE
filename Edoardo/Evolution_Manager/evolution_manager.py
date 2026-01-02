from Edoardo.Crossover.crossover import Crossover
from Edoardo.Species.species import Species
import random


class EvolutionManager:
    def __init__(self, num_parents: int = 2):
        """Manages the evolution process across generations and species.
        :param num_parents: Number of parents to use for creating offspring.
        """
        self.current_generation_index = 0 # Index of the current generation
        self.species = []
        self.num_parents = num_parents

    def create_new_generation(self):
        """
        Creates a new generation of individuals based on the fitness of the current generation.
        First computes average fitness for each species and across whole population.
        Then allocates number of offspring for each species based on their average fitness compared to population average fitness.
        Finally creates offspring for each species.
        
        :param self: Description
        """
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

    def get_active_species_count(self):
        """
        Counts the number of species that have members in the current generation.
        """
        count = 0
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                count += 1
        return count
    
    def get_latest_generation(self):
        """
        Retrieves all members from the latest generation across all species.
        """
        members = []
        for species in self.species:
            if species.last_generation_index() == self.current_generation_index:
                members.extend(species.get_all_members())
        return None

    def create_offspring(self, species):
        """
        Creates the new generation offspring for a given species.
        """
        parent_pool = species.get_top_members()
        healthy_child = False
        child = None
        while not healthy_child:
            parents = self._select_parents(parent_pool, self.num_parents)
            child = Crossover.create_offspring(parents[0].genome, parents[1].genome)
            #TODO: add mutations
            new_species = True
            for s in self.species:
                if s.belongs_to_species(child) and s != species and s.generation_offset == self.current_generation_index+1:
                    # new child belongs to newly generates species together with another individual
                    healthy_child = True
                    new_species = False
                    break
                if s.belongs_to_species(child) and s != species:
                    # new child would belong to an already existing species. Abort creation
                    healthy_child = False
                    new_species = False
                    break
                elif s.belongs_to_species(child) and s == species:
                    # new child belongs to same species as its parents
                    healthy_child = True
                    new_species = False
                    break
            if new_species:
                healthy_child = True
                #TODO: decide where to use genotype and where phenotype
                self.species.append(Species([child], generation=self.current_generation_index+1))
        return child

    @staticmethod
    def _select_parents(parent_pool, num_parents: int = 2):
        """
        Randomly select num_parents elements from the parent_pool list
        """
        return random.sample(parent_pool, num_parents)