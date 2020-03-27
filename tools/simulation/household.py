import numpy as np
from typing import List
from tools.simulation.population import PopulationBase
#from tools.simulation.population_centre import PopulationCentreBase


class Household:
    """
    Defines a household where inhabitants of a city live. It contains indexes of people living in the household.
    People living together have a higher chance of infecting each other.

    attributes:
        - n_inhabitants         <int>               Number of people living in the household

        - population_index      <np.ndarray<int>>   Array of the people living in a household. Index of a population to which the inhabitant belongs to.

        - citizen_index         <np.ndarray<int>>   Array of the people living in a household. Index of the citizen inside his population.
    """
    def __init__(self):
        """
        Creates empty household
        """
        self.n_inhabitants = 0
        self.population_index = np.empty((0), dtype=int)
        self.citizen_index    = np.empty((0), dtype=int)


    def add_inhabitant(self,
                      population_index: int,
                      citizen_index: int):
        self.population_index = np.append(self.population_index, np.array([population_index]), axis = 0)
        self.citizen_index    = np.append(self.citizen_index   , np.array([citizen_index])   , axis = 0)
        self.n_inhabitants += 1

    def simulate_daily_spread_in_household( self,
                                            transmition_prob : float,
                                            populations: List[PopulationBase]
                                            ):
        """
        Loop over inhabitants. If they are infected, but not dead or hospitalized, simulate the spread of the virus inside the house

        :param transmition_prob: probability for an infected person to infect other household inhabitants during the day (night)

        :param populations: list of popolations in the household's town
        """

        # spread the virus in direction i->j
        for i in range(self.n_inhabitants):
            # continue if the inhabitant is not infrected:
            if (not populations[self.population_index[i]]._ill[self.citizen_index[i]]):
                continue

            # continue if he is already hospitalized
            if (populations[self.population_index[i]]._need_hospitalization[self.citizen_index[i]]):
                continue

            infection_mask = np.random.random(self.n_inhabitants) < transmition_prob
            for j in range(self.n_inhabitants):
                if i == j:
                    continue
                if infection_mask[j]:
                    populations[self.population_index[j]].infect_particular_citizen(self.citizen_index[j])
