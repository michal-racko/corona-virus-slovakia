import json

try:
    import cupy as cp

    cuda = True

except ImportError:
    import numpy as cp

    cuda = False


class SocialNetwork:
    """
    Provides an iterator for meeting patterns within the given social network.

    Allows for contact tracing within the social network.
    """

    def __init__(self, meeting_dict: dict, daily_fraction=0.3):
        self._interactions = {}

        for interaction_i, data in meeting_dict.items():
            self._interactions[int(interaction_i)] = {
                'vertex_start': cp.array(data['vertex_start']),
                'vertex_end': cp.array(data['vertex_end']),
            }

        self._daily_fraction = daily_fraction

    def __iter__(self):
        self._iter_i = 0

        return self

    def __next__(self):
        try:
            res = self.get_vertices(self._iter_i)

        except KeyError:
            raise StopIteration

        self._iter_i += 1

        return res

    def get_vertices(self, interaction_i: int) -> tuple:
        """
        :param interaction_i:       Desired interaction i

        :returns:                   A tuple of start and end vertices respectively
        """
        vertex_start = self._interactions[interaction_i]['vertex_start']
        vertex_end = self._interactions[interaction_i]['vertex_end']

        random_mask = cp.random.random(len(vertex_start)) <= self._daily_fraction

        return vertex_start[random_mask], vertex_end[random_mask]

    @classmethod
    def read_json(cls, filepath: str, daily_fraction=0.3):
        with open(filepath) as f:
            data = json.load(f)

        return cls(data, daily_fraction)
