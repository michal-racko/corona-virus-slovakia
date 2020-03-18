import json


class SimulationResult:
    """
    Encapsulates results of a simulation
    """

    def __init__(self,
                 days: list,
                 infected: list,
                 unaffected: list,
                 immune: list,
                 dead: list):
        assert len(infected) == len(days)
        assert len(unaffected) == len(days)
        assert len(immune) == len(days)
        assert len(dead) == len(days)

        self.days = days
        self.infected = infected
        self.unaffected = unaffected
        self.immune = immune
        self.dead = dead

    def to_json(self, filepath: str):
        """
        Saves results as a .json file to the given path
        """
        with open(filepath, 'w') as f:
            json.dump({
                'days': self.days,
                'infected': self.infected,
                'unaffected': self.unaffected,
                'immune': self.immune,
                'dead': self.dead
            }, f)
