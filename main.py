import time
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)

import numpy as np
import matplotlib.pyplot as pl

from tools.config import Config
from tools.input_data import InputData
from tools.simulation.virus import Virus
from tools.simulation.population import Population
from tools.data_structure import GeographicalResult

DEFAULT_CONFIG_PATH = 'config/business_as_usual.yml'

parser = argparse.ArgumentParser(
    description='A Monte Carlo simulation of virus spreading'
)

parser.add_argument(
    '-c',
    '--config-path',
    help='Path to the config file (.yml)',
    required=False
)

args = parser.parse_args()

config = Config()

try:
    config.read(args.config_path)

    logging.info(f'Using config: {args.config_path}')

except TypeError:
    config.read(DEFAULT_CONFIG_PATH)

    logging.info(f'Using config: {DEFAULT_CONFIG_PATH}')

if __name__ == '__main__':
    virus = Virus.from_string(config.get('virus', 'name'))
    results = GeographicalResult()

    population = Population(virus=virus)

    prior = time.time()

    input_data = InputData()

    mig_matrix = input_data.migration_matrix

    city_ids = input_data.get_city_ids()
    city_names = input_data.get_city_names()
    city_longitudes = input_data.get_longitudes()
    city_latitudes = input_data.get_latitudes()
    city_sizes = input_data.get_population_sizes()

    results.set_city_ids(city_ids, city_names)
    results.set_city_coords(city_longitudes, city_latitudes)
    results.set_city_sizes(city_sizes)

    infected = input_data.get_infected() * 10

    population.infect_by_cities(city_ids, infected)

    results_file = config.get('result_file')

    for day_i in range(config.get('simulation_days')):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        if day_i < 20:
            _mig_matrix = np.random.poisson(mig_matrix)

        else:
            _mig_matrix = np.random.poisson(mig_matrix) * 0.7

        _mig_matrix = np.diag(_mig_matrix) - np.identity(len(_mig_matrix))

        population.travel(_mig_matrix)
        population.next_day()

        _, susceptible = population.get_susceptible_by_city()
        _, infected = population.get_infected_by_city()
        _, immune = population.get_immune_by_city()
        _, dead = population.get_dead_by_city()
        _, hospitalized = population.get_hospitalized_by_city()
        _, critical_care = population.get_critical_care_by_city()
        _, new_cases = population.get_new_cases_by_city()

        results.add_susceptible(susceptible)
        results.add_infected(infected)
        results.add_immune(immune)
        results.add_dead(dead)
        results.add_hospitalized(hospitalized)
        results.add_critical_care(critical_care)
        results.add_new_cases(new_cases)

        results.to_json(results_file)

    pl.hist(population.get_dead_ages())
    pl.show()

    logging.info(f'Simulation done in {time.time() - prior:.2e} sec')
