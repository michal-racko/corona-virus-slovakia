import time

import numpy as np

from tools.config import Config
from tools.simulation.virus import Virus


class PopulationBase:
    """
    Abstracts a population of people e.g. in a city.

    attributes:
        - size                  <int>               population size

        - ill                   <np.ndarray<bool>>  defines whether ill or healthly
                                                    (True or False respectively)

        - illness_days_start    <np.ndarray<int>>   day_i when individuals contracted the illness
                                                    the illness (-1 means healthly)

        - illness_days          <np.ndarray<int>>   total number of days it will have taken
                                                    to recover (-1 means healthly)

        - is_alive              <np.ndarray<bool>>  defines whether live or dead
                                                    (True or False respectively)

        - is_immune             <np.ndarray<bool>>  defines whether has build immunity
                                                    (True or False respectively)
    """

    def __init__(self,
                 size: int,
                 virus: Virus):
        config = Config()

        self._size = size
        self._indexes = np.arange(size)

        self._ill = np.zeros(size).astype(bool)
        self._illness_days = np.ones(size) * -1
        self._illness_days_start = np.ones(size) * -1

        self._health_condition = np.random.random(size)
        self._need_hospitalization = np.zeros(size).astype(bool)

        self._hospitalization_start = np.random.poisson(
            config.get('hospitalization_start'),
            size
        )
        self._hospitalization_percentage = config.get('hospitalization_percentage')

        self._is_new_case = np.zeros(size).astype(bool)

        self._is_immune = np.zeros(size).astype(bool)
        self._is_alive = np.ones(size).astype(bool)

        self._infectious_start = np.random.poisson(
            config.get('infectious_start'),
            size
        )

        self._mean_stochastic_interactions = config.get('population', 'mean_stochastic_interactions')
        self._mean_periodic_interactions = config.get('population', 'mean_periodic_interactions')

        self._virus = virus

        self._day_i = 0

    def __len__(self):
        return self._size

    @staticmethod
    def _reset_random_seed():
        """
        Resets numpy's random seed based on unix timestamp
        with 10 nanosecond precision
        """
        current_time = time.time() * 1e8

        np.random.seed(
            int(current_time % (2 ** 32 - 1))
        )

    def get_health_states(self, n=None, random_seed=None) -> np.ndarray:
        """
        :param n:               subsample size (returns all members by default)

        :param random_seed:     random seed to be used for the random selection

        :returns:               health state of randomly selected members of the population
        """
        if random_seed is None:
            self._reset_random_seed()

        else:
            np.random.seed(random_seed)

        current_infectious = self._ill[self._is_alive]
        current_infectious[
            self._day_i - self._illness_days_start[self._is_alive] < self._infectious_start[self._is_alive]
            ] = False  # not yet infectious

        if n is None:
            return current_infectious

        elif n < len(current_infectious):
            return np.random.choice(current_infectious, n, replace=False)

        else:
            return current_infectious

    def get_stochastic_interaction_multiplicities(self, n=None) -> np.ndarray:
        """
        :returns:               numbers of random interactions for each person
        """
        self._reset_random_seed()

        n_alive = self._is_alive.astype(int).sum()

        if n is None:
            return np.random.poisson(
                self._mean_stochastic_interactions,
                n_alive
            ).astype(int)

        elif n < n_alive:
            return np.random.poisson(
                self._mean_stochastic_interactions,
                n
            ).astype(int)

        else:
            return np.random.poisson(
                self._mean_stochastic_interactions,
                n_alive
            ).astype(int)

    def get_periodic_interaction_multiplicities(self,
                                                n=None,
                                                random_seed=42) -> np.ndarray:
        """
        :returns:               numbers of periodic interactions for each person
        """
        np.random.seed(random_seed)

        n_alive = self._is_alive.astype(int).sum()

        if n is None:
            return np.random.poisson(
                self._mean_periodic_interactions,
                n_alive
            ).astype(int)

        elif n < n_alive:
            return np.random.poisson(
                self._mean_periodic_interactions,
                n
            ).astype(int)

        else:
            return np.random.poisson(
                self._mean_periodic_interactions,
                n_alive
            ).astype(int)

    def get_n_unaffected(self) -> int:
        """
        :returns:       number of members who are currently unaffected by the virus
        """
        return int((self._illness_days_start == -1).astype(int).sum())

    def get_n_infected(self) -> int:
        """
        :returns:       number of members who are currently infected
        """
        return int(self._ill.astype(int).sum())

    def get_n_new_cases(self) -> int:
        """
        :returns:       number of members who were infected in the current day
        """
        return int(self._is_new_case.astype(int).sum())

    def get_n_immune(self) -> int:
        """
        :returns:       number of members who are currently immune
        """
        return int(self._is_immune.astype(int).sum())

    def get_n_dead(self) -> int:
        """
        :returns:       number of members who have passed away due to the illness
        """
        return int((~self._is_alive).astype(int).sum())

    def get_n_hospitalized(self) -> int:
        """
        :returns:       number of members who are currently need hospitalization
        """
        return int(self._need_hospitalization.astype(int).sum())

    def infect(self, n: float, random_seed=None):
        """
        Infects n randomly selected people
        """
        if random_seed is None:
            self._reset_random_seed()

        else:
            np.random.seed(random_seed)

        infectable = self._indexes.copy()
        infectable[self._is_immune * ~self._is_alive] = -1

        true_n = int(n)

        if n % 1 >= np.random.random():
            true_n += 1

        if true_n < self._size:
            indexes = np.random.choice(
                infectable,
                true_n
            )

        else:
            indexes = infectable

        indexes = indexes[indexes >= 0]

        current_days = self._illness_days_start[indexes]

        current_days[self._illness_days_start[indexes] == -1] = self._day_i

        self._illness_days_start[indexes] = current_days

        if n < self._size:
            self._ill[indexes] = True

        else:
            self._ill[:] = True

        self._is_new_case = self._illness_days_start == self._day_i

        self._need_hospitalization = self._ill * \
                                     (self._health_condition <= self._hospitalization_percentage) * \
                                     (self._hospitalization_start >= (self._day_i - self._illness_days_start))

    def heal(self):
        """
        Heals members of the population if they are infected
        """
        healed = (self._day_i - self._illness_days_start - self._infectious_start) >= abs(
            np.random.normal(
                self._virus.illness_days_mean,
                self._virus.illness_days_std,
                len(self._illness_days_start)
            )
        )

        immune = self._ill * healed

        self._ill[healed] = False
        self._illness_days[healed] = self._day_i - self._illness_days_start[healed]

        self._need_hospitalization[immune] = False

        self._is_immune[immune] = True

    def kill(self):
        """
        Kills a portion of the infected population
        """
        daily_prob = self._virus.get_mortality() / self._virus.illness_days_mean

        ill_alive = (self._ill * self._is_alive).copy()

        passed_away = np.random.random(
            ill_alive.astype(int).sum()
        ) > daily_prob

        self._is_alive[ill_alive] = passed_away

        self._need_hospitalization[ill_alive] *= passed_away

    def next_day(self):
        """
        Next day of the simulation
        """
        self._day_i += 1

        # if self._day_i == 90:
        #     self._mean_stochastic_interactions = 18
