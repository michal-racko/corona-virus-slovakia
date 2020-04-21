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
                'vertex_start': cp.array(data['vertex_start']).astype(int),
                'vertex_end': cp.array(data['vertex_end']).astype(int),
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

    def multiply_daily_fraction(self, by: float):
        """
        Multiplies the daily fraction by the given number
        """
        self._daily_fraction *= by

    def get_vertices(self, interaction_i: int) -> tuple:
        """
        :param interaction_i:       Desired interaction i

        :returns:                   A tuple of start and end vertices respectively
        """
        vertex_start = self._interactions[interaction_i]['vertex_start']
        vertex_end = self._interactions[interaction_i]['vertex_end']

        random_mask = cp.random.random(len(vertex_start)) <= self._daily_fraction

        return vertex_start[random_mask], vertex_end[random_mask]

    def get_contacts(self, indices):
        """
        Returns all contacts of the given indices
        """
        indices = cp.sort(indices)

        res = []

        for start_vertices, end_vertices in self:
            _indices = cp.searchsorted(start_vertices, indices)
            _indices = _indices[_indices != 0]

            if len(_indices) == 0:
                continue

            res.append(end_vertices[_indices])

        if len(res) == 0:
            return cp.array([])

        else:
            return cp.hstack(res)

    def trace(self, indices: cp.ndarray, recursion_depth: int, efficiency: float) -> cp.ndarray:
        """
        Searches contacts of the given indices (start vertices) up to the given recursion depth.
        Only (1 - efficiency) contacts will be returned

        :param indices:                 desired indices

        :param recursion_depth:         desired recursion depth

        :param efficiency:              efficiency of contact tracing

        :return:                        indices of corresponding contacts (end vertices)
        """
        res = []

        searched = indices

        for i in range(recursion_depth):
            contacts = self._search_contacts(searched)

            contacts = contacts[cp.random.random(len(contacts)) <= efficiency]

            res.append(contacts)

            searched = cp.hstack(res)

        return cp.hstack(searched)

    @classmethod
    def read_json(cls, filepath: str, daily_fraction=0.3):
        with open(filepath) as f:
            data = json.load(f)

        return cls(data, daily_fraction)
