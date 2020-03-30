import json
import pickle
import logging

import numpy as np
import pandas as pd

from tools.config import Config
from tools.general import singleton


@singleton
class InputData:
    """
    Reads data from files as specified in the config and stores it.
    """

    def __init__(self):
        self._config = Config()

        self._municipal_df = self._prepare_municipal_df()

        with open(self._config.get('migration_matrix'), 'rb') as f:
            self.migration_matrix = pickle.load(f)

        min_inhabitants = self._config.get('min_inhabitants')

        inhabitants = self._municipal_df.popul.values

        population_size_mask = inhabitants > min_inhabitants

        self._municipal_df = self._municipal_df.loc[population_size_mask]
        self.migration_matrix = self.migration_matrix[population_size_mask].T[population_size_mask].T

        self._municipal_df['city_id'] = np.arange(len(self._municipal_df))

        logging.info(f'Municipal data preview:\n {self._municipal_df}')

        self.mean_travel_ratio = self._get_mean_travel_ratio()

        self.age_distribution = None

        self._prepare_age_distribution()

        self.symptoms = None

        self._prepare_symptoms()

        self.household_data = None
        self.mean_household_daily_meetings = None

        self._prepare_households()

    def _prepare_symptoms(self):
        with open(self._config.get('age_symptoms')) as f:
            symptom_data = json.load(f)

        self.symptoms = {}

        for symptom_type, data in symptom_data.items():
            self.symptoms[symptom_type] = {
                int(age): prob for age, prob in data.items()
            }

    def _prepare_age_distribution(self):
        with open(self._config.get('age_distribution')) as f:
            age_data = json.load(f)

        n_all = sum(age_data.values())

        self.age_distribution = {
            int(age): n / n_all for age, n in age_data.items()
        }

    def _prepare_households(self):
        with open(self._config.get('household_distribution')) as f:
            data = json.load(f)

            self.household_data = {
                'elderly': {
                    int(size): ratio for size, ratio in data['elderly'].items()
                },
                'young': {
                    int(size): ratio for size, ratio in data['young'].items()
                }
            }

        elderly_ratio = 0
        young_ratio = 0

        for age, ratio in self.age_distribution.items():
            if age >= 60:
                elderly_ratio += ratio

            else:
                young_ratio += ratio

        self.mean_household_daily_meetings = 0

        self.mean_household_daily_meetings += self.household_data['elderly'][2] * elderly_ratio

        for size, ratio in self.household_data['young'].items():
            if size > 1:
                self.mean_household_daily_meetings += size * ratio * young_ratio

    def _get_mean_travel_ratio(self) -> float:
        """
        :returns:       Ratio of mean number of daily travelling people
                        to the full population size
        """
        total_meetings = 0

        for i in range(len(self.migration_matrix)):
            for j in range(len(self.migration_matrix)):
                if i == j:
                    continue

                total_meetings += self.migration_matrix[i][j]

        return total_meetings / self._municipal_df.popul.sum()

    def get_population_sizes(self) -> np.ndarray:
        return self._municipal_df.popul.values

    def get_city_ids(self) -> np.ndarray:
        return self._municipal_df.city_id.values

    def get_longitudes(self) -> np.ndarray:
        return self._municipal_df.long.values

    def get_latitudes(self) -> np.ndarray:
        return self._municipal_df.lat.values

    def get_city_names(self) -> list:
        return self._municipal_df.NM4.tolist()

    def get_infected(self) -> np.ndarray:
        return self._municipal_df.infected.values

    def get_migration(self, i: int, j: int) -> int:
        """
        :param i:       index of the first city
        :param j:       index of the second city

        :returns:       mean number of people who daily travel between city i and j
        """
        return int(self.migration_matrix[i][j])

    def get_migration_row(self, i) -> np.ndarray:
        return self.migration_matrix[i]

    def get_migration_by_names(self, city_name_a: str, city_name_b: str) -> int:
        city_names = self.get_city_names()

        i = city_names.index(city_name_a)
        j = city_names.index(city_name_b)

        return self.get_migration(i, j)

    def _prepare_municipal_df(self):
        population_df = pd.read_excel(
            self._config.get('populations_file')
        )

        town_location_df = pd.read_excel(
            self._config.get('town_locations_file')
        )

        return population_df.merge(
            town_location_df,
            left_on='munic',
            right_on='IDN4'
        )
