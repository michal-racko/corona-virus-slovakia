import logging

import numpy as np

from tools.config import Config
from tools.factories import city_factory
from tools.simulation.virus import Virus
from tools.data_structure import GeographicalResult


def wander_aroud(city_indexes, cities):
    """
    Simulates people travelling among cities. Uses a single CPU core.
    """
    for i in city_indexes:
        for j, city_j in enumerate(cities):
            city_i = cities[i]

            distance = np.sqrt(
                (city_i.longitude - city_j.longitude) ** 2 + (city_i.latitude - city_j.latitude) ** 2
            )

            if distance == 0:
                continue

            travelling = min(len(city_i), len(city_j)) / (distance * 3) ** 2

            health_states = city_i.get_health_states(travelling)

            n_infected = health_states.astype(int).sum()

            city_j.infect(n_infected)


def run_simulation() -> GeographicalResult:
    config = Config()

    virus = Virus.from_string(config.get('virus'))
    city_csv = config.get('city_csv')
    n_days = config.get('simulation_days')

    cities = city_factory(city_csv, virus)

    results = GeographicalResult()

    cities[5].infect(50)

    for day_i in range(n_days):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        for city_i in cities:
            for city_j in cities:
                distance = np.sqrt(
                    (city_i.longitude - city_j.longitude) ** 2 + (city_i.latitude - city_j.latitude) ** 2
                )

                if distance == 0:
                    continue

                travelling = min(len(city_i), len(city_j)) / (distance * 3) ** 2

                health_states = city_i.get_health_states(travelling)

                n_infected = health_states.astype(int).sum()

                city_j.infect(n_infected)

        for city in cities:
            city.next_day()

        for city in cities:
            results.add_result(city.to_dict())

    return results
