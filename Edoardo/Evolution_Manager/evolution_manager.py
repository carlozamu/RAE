class EvolutionManager:
    def __init__(self):
        self.generations = []

    def create_new_generation(self):
        pass

    def get_latest_generation(self):
        if self.generations:
            return self.generations[-1]
        return None