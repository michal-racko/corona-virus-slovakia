import logging

import numpy as np

from tools.config import Config
from tools.input_data import InputData
from tools.factories import city_factory
from tools.simulation.virus import Virus
from tools.data_structure import GeographicalResult


def run_simulation() -> GeographicalResult:
    config = Config()

    virus = Virus.from_string(config.get('virus', 'name'))
    n_days = config.get('simulation_days')

    cities = city_factory(virus)

    results = GeographicalResult()

    input_data = InputData()

    for day_i in range(n_days):
        if day_i % 10 == 0:
            logging.info(f'day: {day_i}')

        for i, city_i in enumerate(cities):
            migrations_smeared = np.random.poisson(input_data.get_migration_row(i))

            for j, city_j in enumerate(cities):
                if i == j:
                    continue

                health_states, interaction_multiplicities = city_i.get_travelling(migrations_smeared[j])

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

        for city in cities:
            city.next_day()

        for city in cities:
            results.add_result(city.to_dict())

    return results
