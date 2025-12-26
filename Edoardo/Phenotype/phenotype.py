"""Working individual composed by many traits, created from a genotype"""
from Edoardo.Genome.agent_genome import AgentGenome


class Phenotype:
    def __init__(self, genome: AgentGenome):
        self.genome = genome
        pass