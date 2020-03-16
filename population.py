import numpy as np

from abc import ABCMeta, abstractmethod


class Population:
    """
    Abstract a population of people e.g. in a city.

    attributes:
        - size          <int>               population size

        - ill           <np.ndarray<int>>   defines whether ill or healthly
                                            (1 or 0 respectively)

        - illness_days  <np.ndarray<int>>   number of days for which each individual
                                            has been ill (-1 means healthly)

        - is_alive      <np.ndarray<int>>   defines whether live or dead (1 or 0 respectively)

        - is_immune     <np.ndarray<int>>   defines whether has build immunity (1 or 0 respectively)
    """

    __metaclass__ = ABCMeta

    def __init__(self,
                 size: int):
        self.size = size

        self.ill = np.zeros(size)
        self.illness_days = np.ones(size) * -1

        self.is_imune = np.zeros(size)
        self.is_alive = np.ones(size)

        self._day_i = 0

    def __len__(self):
        return self.size

    @abstractmethod
    def get_members(self, n_sample: int, random_seed=None) -> np.ndarray:
        """
        :returns:       randomly selected members of the population
        """
        pass

    @abstractmethod
    def infect(self, ):
        """
        Makes members of the population meet and spreads the disease among each other
        """
        pass

    @abstractmethod
    def heal(self):
        """
        Heals members of the population if they are infected
        """
        pass

    @abstractmethod
    def kill(self):
        """
        Kills a portion of the infected members
        """
        pass
