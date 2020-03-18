import numpy as np


class Population:
    """
    Abstracts a population of people e.g. in a city.

    attributes:
        - size          <int>               population size

        - ill           <np.ndarray<bool>>  defines whether ill or healthly
                                            (True or False respectively)

        - illness_days  <np.ndarray<int>>   day_i when individuals contracted the illness
                                            the illness (-1 means healthly)

        - is_alive      <np.ndarray<bool>>  defines whether live or dead
                                            (True or False respectively)

        - is_immune     <np.ndarray<bool>>  defines whether has build immunity
                                            (True or False respectively)
    """

    def __init__(self,
                 size: int):
        self.size = size
        self.indexes = np.arange(size)

        self.ill = np.zeros(size).astype(bool)
        self.illness_days = np.ones(size) * -1

        self.is_immune = np.zeros(size).astype(bool)
        self.is_alive = np.ones(size).astype(bool)

        self._day_i = 0

    def __len__(self):
        return self.size

    def get_members(self, n: int, random_seed=None) -> np.ndarray:
        """
        :returns:       health state of randomly selected members of the population
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        currently_ill = self.ill[self.is_alive]

        if n < len(currently_ill):
            return np.random.choice(currently_ill, n, replace=False)

        else:
            return currently_ill

    def get_n_infected(self) -> int:
        """
        :returns:       number of members who are currently infected
        """
        return self.ill.astype(int).sum()

    def get_n_immune(self) -> int:
        """
        :returns:       number of members who are currently immune
        """
        return self.ill.astype(int).sum()

    def get_n_dead(self) -> int:
        """
        :returns:       number of members who are currently immune
        """
        return (~self.is_alive).astype(int).sum()

    def infect(self, n: int, random_seed=None):
        """
        Infects n randomly selected people
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        infectable = self.indexes.copy()
        infectable[self.is_immune * ~self.is_alive] = -1

        if n < self.size:
            indexes = np.random.choice(
                infectable,
                n
            )

        else:
            indexes = infectable

        indexes = indexes[indexes >= 0]

        self.illness_days[indexes][~self.ill[indexes]] = self._day_i

        if n < self.size:
            self.ill[indexes] = True

        else:
            self.ill[:] = True

    def heal(self,
             healing_days_mean: int,
             healing_days_std: int):
        """
        Heals members of the population if they are infected

        :param healing_days_mean:       mean number of days it takes for
                                        a diseased person to recover

        :param healing_days_std:        stev of days it takes
                                        a diseased person to recover
        """
        healed = (self._day_i - self.illness_days) >= abs(
            np.random.normal(healing_days_mean, healing_days_std, len(self.illness_days))
        )

        self.ill[healed] = False
        self.is_immune[healed] = True

    def kill(self, death_probability: float, healing_days: int):
        """
        Kills a portion of the infected population
        """
        daily_prob = death_probability / healing_days

        ill_alive = (self.ill * self.is_alive).copy()

        self.is_alive[ill_alive] = np.random.random(
            ill_alive.astype(int).sum()
        ) > daily_prob

    def next_day(self):
        self._day_i += 1


class City(Population):
    """
    Abstracts a population living in an urban area
    """

    def __init__(self,
                 size: int,
                 latitude: float,
                 longitude: float):
        self.latitude = latitude
        self.longitude = longitude

        super().__init__(size)
