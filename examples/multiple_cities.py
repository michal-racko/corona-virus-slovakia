import gc
import logging

import numpy as np

from multiprocessing import Pool

from tools.config import Config
from tools.input_data import InputData
from tools.factories import city_factory
from tools.simulation.virus import Virus
from tools.data_structure import GeographicalResult


class ProcessData:
    """
    Encapsulates data which will be passed to each process
    """

    def __init__(self,
                 cities_subset,
                 cities_all,
                 process_i,
                 virus,
                 migrations_smeared):
        self.cities_subset = cities_subset
        self.cities_all = cities_all
        self.process_i = process_i
        self.virus = virus
        self.migrations_smeared = migrations_smeared


def travel(data: ProcessData) -> dict:
    """
    Simalates migrations among cities in
    """
    cities_subset = data.cities_subset
    virus = data.virus
    migrations_smeared = data.migrations_smeared

    for i, city_i in enumerate(data.cities_all):
        for j, city_j in enumerate(cities_subset):
            if i == j:
                continue

            health_states, interaction_multiplicities = city_i.get_travelling(migrations_smeared[i][j])

            if len(interaction_multiplicities) == 0:
                continue

            n_infected = 0

            for interaction_i in range(interaction_multiplicities.max()):
                transmission_mask = (interaction_multiplicities > interaction_i) * (
                        np.random.random(len(interaction_multiplicities)) < virus.transmission_probability
                )

                n_infected += health_states[transmission_mask].astype(int).sum()

            n_infected = health_states.astype(int).sum()

            city_j.infect(n_infected)

    return {
        data.process_i: cities_subset
    }


def run_simulation() -> GeographicalResult:
    config = Config()

    virus = Virus.from_string(
        config.get('virus', 'name'),
        illness_days_mean=config.get('virus', 'infectious_days_mean'),
        illness_days_std=config.get('virus', 'infectious_days_std'),
        transmission_probability=config.get('virus', 'transmission_probability'),
        mean_periodic_interactions=config.get('virus', 'infectious_days_mean'),
        mean_stochastic_interactions=config.get('virus', 'infectious_days_mean')
    )
    n_days = config.get('simulation_days')

    cities = city_factory(virus)

    results = GeographicalResult()

    input_data = InputData()

    n_processes = config.get('n_processes')

    pool = Pool(processes=n_processes)

    for day_i in range(n_days):
        if True:
            logging.info(f'day: {day_i}')

        migrations_smeared = np.random.poisson(input_data.migration_matrix)

        city_indexes = np.array_split(
            np.arange(len(cities)),
            n_processes
        )

        processes_data = [
            ProcessData(
                cities_subset=[cities[i] for i in subset_indexes],
                cities_all=cities,
                process_i=i,
                virus=virus,
                migrations_smeared=migrations_smeared
            ) for i, subset_indexes in enumerate(city_indexes)
        ]

        result_dict = {}

        for r in pool.map(travel, processes_data):
            result_dict.update(r)

        cities = []
        gc.collect()

        for i in range(len(result_dict)):
            cities += result_dict[i]

        for city in cities:
            city.next_day()

        for city in cities:
            results.add_result(city.to_dict())

    return results
